"""Refresh Jev answers for a frozen example benchmark. Makes paid external API calls.

Requires --confirm-live, caps attempted requests, never retries, and preserves full
responses. Resolve jev-latest on the first answer, then request that reported version.
The recorded local-model predictions are reused, not silently regenerated.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from jev.data import Example, normalize_target, read_jsonl, sha256_file
from jev.jev_api import api_key
from jev.serialization import dumps

if __package__:
    from .benchmark_cached_teacher import NoRedirect, fidelity, metrics
else:
    from benchmark_cached_teacher import NoRedirect, fidelity, metrics

API_URL = "https://api.typesafe.ai/v1/systemone"


def teacher_probabilities(question, answer):
    if answer.get("type") != question.type:
        raise ValueError("teacher answer primitive mismatch")
    if question.type == "noul":
        yes = answer["noul"]
        if type(yes) not in (int, float):
            raise ValueError("invalid teacher Noul value")
        vector = [1 - yes, yes]
    else:
        if set(answer["probabilities"]) != set(question.answer_keys):
            raise ValueError("teacher answer labels differ from the supplied rubric")
        vector = [answer["probabilities"][key] for key in question.answer_keys]
    # API rounding can leave .99 total mass; preserve the unmodified response too.
    return normalize_target(vector, len(question.answer_keys))


def resolved_version(reported, pinned=None):
    if not isinstance(reported, str) or not re.fullmatch(r"jev-\d+\.\d+\.\d+", reported):
        raise ValueError("response did not identify a supported versioned Jev model; stop before more calls")
    if pinned is not None and reported != pinned:
        raise ValueError("Jev version changed during collection")
    return reported


def load_frozen_run(path, max_requests):
    if max_requests < 1:
        raise ValueError("max-requests must be positive")
    manifest = json.loads((path / "manifest.json").read_text())
    if sha256_file(path / "cases.jsonl") != manifest["cases_sha256"]:
        raise ValueError("case hash differs from the frozen manifest")
    cases = list(read_jsonl(path / "cases.jsonl"))
    predictions = list(read_jsonl(path / "predictions.jsonl"))
    if not 1 <= len(cases) <= max_requests or len(cases) != len(predictions):
        raise ValueError("incomplete source run or request budget exceeded")
    if len({c["case"] for c in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    for case, row in zip(cases, predictions):
        example = Example.from_dict(case["example"])
        if (case["case"], example.id, example.question.answer_keys) != (row["case"], row["id"], row["answer_keys"]):
            raise ValueError("source cases and predictions are not aligned")
        if case["request"]["state"] != example.state or case["request"]["questions"] != {"q": asdict(example.question)}:
            raise ValueError("request does not match the preserved teacher input")
        if row["gold_label"] != example.provenance["gold_label"]:
            raise ValueError("recorded gold label changed")
        if row["response"]["model"] != manifest["server"]["id"]:
            raise ValueError("source local-model identity mismatch")
    return manifest, cases, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=Path("data/runs/benchmark-examples-v2"))
    parser.add_argument("--teacher", default="jev-latest")
    parser.add_argument("--max-requests", type=int, default=24)
    parser.add_argument("--confirm-live", action="store_true", help="authorize external, potentially billable Jev requests")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.confirm_live:
        parser.error("requires --confirm-live; this command makes paid external API calls")
    if args.teacher != "jev-latest" and not re.fullmatch(r"jev-\d+\.\d+\.\d+", args.teacher):
        parser.error("choose jev-latest or an explicit jev-X.Y.Z version")
    if args.out_dir.exists():
        parser.error("output already exists; collection never overwrites or automatically resumes")
    original, cases, local_rows = load_frozen_run(args.source_run, args.max_requests)
    key = api_key()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": API_URL, "source_run": str(args.source_run),
        "source_hashes": {name: sha256_file(args.source_run / name)
                          for name in ("manifest.json", "cases.jsonl", "predictions.jsonl", "report.json")},
        "initial_requested_model": args.teacher, "max_attempted_requests": args.max_requests,
        "cases": len(cases), "retry_policy": "none; stop on first failure",
        "pinning_policy": "resolve first response then require and request the same explicit version",
        "local_model": original["server"], "script_sha256": sha256_file(__file__),
        "metrics_helper_sha256": sha256_file(Path(__file__).with_name("benchmark_cached_teacher.py")),
        "limitations": "Same tiny, familiar-task diagnostic with historical input truncation. Original Qwen predictions reused. Not training data or a final test.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows, pinned = [], None if args.teacher == "jev-latest" else args.teacher
    with (args.out_dir / "requests.jsonl").open("x") as journal, (args.out_dir / "responses.jsonl").open("x") as raw, (args.out_dir / "predictions.jsonl").open("x") as predictions:
        for case, prior in zip(cases, local_rows):
            requested = pinned or args.teacher
            payload = {**case["request"], "model": requested}
            timestamp = datetime.now(timezone.utc).isoformat()
            journal.write(dumps({"case": case["case"], "requested_at_utc": timestamp, "request": payload}) + "\n")
            journal.flush()
            request = urllib.request.Request(API_URL, data=dumps(payload).encode(), headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            })
            start = time.perf_counter()
            # No retry, redirect, alternate endpoint, or fallback-model call.
            with opener.open(request, timeout=60) as response:
                body = json.load(response)
                headers = {name: response.headers[name] for name in ("Date", "X-Request-ID")
                           if name in response.headers}
            elapsed = (time.perf_counter() - start) * 1000
            raw.write(dumps({"case": case["case"], "requested_model": requested,
                             "requested_at_utc": timestamp, "elapsed_ms": elapsed,
                             "response_headers": headers, "response": body}) + "\n")
            raw.flush()  # Preserve responses even if validation below fails.
            pinned = resolved_version(body.get("model"), pinned)
            if set(body["answers"]) != {"q"}:
                raise ValueError("unexpected teacher answer IDs")
            question = Example.from_dict(case["example"]).question
            vector = teacher_probabilities(question, body["answers"]["q"])
            row = {key: prior[key] for key in ("case", "id", "source", "primitive", "answer_keys", "gold_label", "local_probabilities")}
            row.update({"cached_teacher_probabilities": prior["teacher_probabilities"],
                        "teacher_probabilities": vector, "elapsed_ms": elapsed,
                        "requested_model": requested, "response": body})
            rows.append(row)
            predictions.write(dumps(row) + "\n")
            predictions.flush()
            print(f"{case['case']}: {pinned}, {elapsed:.0f} ms", flush=True)

    def summarize(group):
        return {"qwen": metrics(group, "local_probabilities"),
                "fresh_jev": metrics(group, "teacher_probabilities"),
                "cached_jev": metrics(group, "cached_teacher_probabilities"),
                "qwen_vs_fresh_jev": fidelity(group)}

    report = {
        "resolved_model": pinned, "requests_attempted": len(cases), "completed": len(rows),
        "overall": summarize(rows),
        "by_primitive": {p: summarize([r for r in rows if r["primitive"] == p])
                         for p in sorted({r["primitive"] for r in rows})},
        "cached_to_fresh": {
            "argmax_changes": sum(max(range(len(r["teacher_probabilities"])), key=r["teacher_probabilities"].__getitem__)
                                  != max(range(len(r["cached_teacher_probabilities"])), key=r["cached_teacher_probabilities"].__getitem__) for r in rows),
            "max_probability_change": max(abs(a - b) for r in rows
                                          for a, b in zip(r["teacher_probabilities"], r["cached_teacher_probabilities"])),
        },
        "jev_http_latency_ms": {"median": statistics.median(r["elapsed_ms"] for r in rows),
                                "min": min(r["elapsed_ms"] for r in rows),
                                "max": max(r["elapsed_ms"] for r in rows)},
        "usage": {name: sum(r["response"].get("usage", {}).get(name, 0) for r in rows)
                  for name in ("input_tokens", "output_tokens")},
        "metric_policy": "Normalized distributions; first-index argmax; NLL floor 1e-12; summed multiclass Brier; Score uses weighted zero-based indices. No ECE significance claim.",
        "latency_note": "Fresh serial HTTPS calls include WAN/TLS overhead; not simultaneous with local measurements or a model-compute speed comparison.",
        "manifest_sha256": sha256_file(args.out_dir / "manifest.json"),
    }
    # Ensure no nonfinite aggregate sneaks into a completed report.
    if not all(math.isfinite(report["overall"][model][metric])
               for model in ("qwen", "fresh_jev", "cached_jev") for metric in ("accuracy", "nll", "brier")):
        raise ValueError("invalid aggregate metric")
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
