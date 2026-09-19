"""Benchmark an existing loopback server against cached Jev distributions; no live API calls.

Select examples before inference, exclude known local training/development/calibration
matches, and save requests, responses, server identity, and descriptive metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from jev.data import content_key, load_examples, normalize_target, sha256_file
from jev.provenance import environment_identity
from jev.serialization import dumps


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("benchmark requests must not redirect away from the local endpoint")


def document_text(state):
    """Recognize only the same one-document wrappers as the canonical matcher."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        if set(state) == {"document"} and isinstance(state["document"], str):
            return state["document"]
        ticket = state.get("ticket")
        if set(state) == {"ticket"} and isinstance(ticket, dict):
            if set(ticket) == {"channel", "body"} and isinstance(ticket["body"], str):
                return ticket["body"]
    if isinstance(state, list) and len(state) == 1 and isinstance(state[0], dict):
        message = state[0]
        if set(message) == {"role", "content"} and isinstance(message["content"], str):
            return message["content"]
    return None


def overlap_keys(example):
    keys = example.leakage_keys.copy()
    text = document_text(example.state)
    if text is not None:
        # Historical teacher collection truncated at 1,500 Python characters.
        keys.add(content_key(text[:1500]))
    return keys


def select_examples(examples, blocked, per_primitive=8, seed=42):
    if per_primitive < 2 or per_primitive % 2:
        raise ValueError("per-primitive must be positive, even, and at least two")
    quotas = {
        ("cached:ag_news", "choice"): per_primitive // 2,
        ("cached:dbpedia", "choice"): per_primitive // 2,
        ("cached:imdb", "noul"): per_primitive,
        ("cached:yelp_stars", "score"): per_primitive,
    }
    eligible = [e for e in examples if not overlap_keys(e) & blocked]
    selected, seen = [], set()
    for (source, primitive), count in quotas.items():
        candidates = sorted(
            (e for e in eligible if (e.source, e.question.type) == (source, primitive)),
            key=lambda e: hashlib.sha256(f"{seed}:{e.id}".encode()).hexdigest(),
        )
        picked = 0
        for example in candidates:
            keys = overlap_keys(example)
            if keys & seen:
                continue
            selected.append(example)
            seen.update(keys)
            picked += 1
            if picked == count:
                break
        if picked != count:
            raise ValueError(f"not enough disjoint examples for {source}/{primitive}")
    return selected, len(examples) - len(eligible)


def local_probabilities(question, answer):
    if answer.get("type") != question.type:
        raise ValueError("answer primitive mismatch")
    if question.type == "noul":
        yes = answer["noul"]
        if type(yes) not in (int, float):
            raise ValueError("invalid Noul value")
        probs = [1 - yes, yes]
    else:
        if set(answer["probabilities"]) != set(question.answer_keys):
            raise ValueError("answer label mismatch")
        probs = [answer["probabilities"][key] for key in question.answer_keys]
    normalized = normalize_target(probs, len(question.answer_keys))
    if abs(sum(probs) - 1) > 1e-5:
        raise ValueError("local distribution is not normalized")
    if question.type == "choice":
        winners = {key for key, p in zip(question.answer_keys, probs) if p == max(probs)}
        if answer.get("choice") not in winners:
            raise ValueError("Choice answer does not select a maximum-probability option")
    if question.type == "score":
        expected = sum(i * p for i, p in enumerate(normalized))
        value = answer.get("score")
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value - expected) > 1e-5:
            raise ValueError("Score answer differs from its weighted distribution")
        if answer.get("legend") != dict(zip(question.answer_keys, question.criteria)):
            raise ValueError("Score legend differs from the supplied rubric")
    return normalized


def metrics(rows, field):
    correct, losses, briers = [], [], []
    for row in rows:
        p, label = row[field], row["gold_label"]
        correct.append(max(range(len(p)), key=p.__getitem__) == label)
        losses.append(-math.log(max(p[label], 1e-12)))
        briers.append(sum((v - float(i == label)) ** 2 for i, v in enumerate(p)))
    result = {
        "n": len(rows), "correct": sum(correct),
        "accuracy": statistics.mean(correct), "nll": statistics.mean(losses),
        "brier": statistics.mean(briers),
        "zero_gold_probability_count": sum(r[field][r["gold_label"]] == 0 for r in rows),
    }
    if all(r["primitive"] == "score" for r in rows):
        result["score_mae"] = statistics.mean(
            abs(sum(i * p for i, p in enumerate(r[field])) - r["gold_label"])
            for r in rows
        )
    return result


def fidelity(rows):
    js, tv, agreement = [], [], []
    for row in rows:
        p, q = row["local_probabilities"], row["teacher_probabilities"]
        mixture = [(a + b) / 2 for a, b in zip(p, q)]
        js.append(0.5 * sum(v * math.log(v / m) for vector in (p, q)
                            for v, m in zip(vector, mixture) if v > 0))
        tv.append(sum(abs(a - b) for a, b in zip(p, q)) / 2)
        agreement.append(max(range(len(p)), key=p.__getitem__)
                         == max(range(len(q)), key=q.__getitem__))
    result = {"argmax_matches": sum(agreement), "argmax_agreement": statistics.mean(agreement),
              "mean_js_nats": statistics.mean(js), "mean_total_variation": statistics.mean(tv)}
    noul = [r for r in rows if r["primitive"] == "noul"]
    if noul:
        result["noul_mean_absolute_probability_difference"] = statistics.mean(
            abs(r["local_probabilities"][1] - r["teacher_probabilities"][1]) for r in noul
        )
    return result


def http_json(url, payload=None):
    request = urllib.request.Request(
        url, data=None if payload is None else dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    # A local benchmark must not send states through an environment-configured proxy
    # or follow a redirect to an external provider.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8399")
    parser.add_argument("--adapter", type=Path, default=Path("adapters/qwen3-0.6b-structured-v1-lr1e-5"))
    parser.add_argument("--reference", type=Path, default=Path("data/experiments/teacher-matched/teacher_test.jsonl"))
    parser.add_argument("--exclude", nargs="+", type=Path, default=[
        Path(f"data/kev-v1/{split}.jsonl") for split in ("train", "development", "calibration")
    ])
    parser.add_argument("--per-primitive", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.per_primitive < 2 or args.per_primitive % 2:
        parser.error("--per-primitive must be even and at least two")
    url = urllib.parse.urlsplit(args.base_url)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1"):
        parser.error("only a loopback HTTP server is allowed; this script makes no live Jev calls")
    if url.path not in ("", "/") or url.query or url.fragment or url.username or url.password:
        parser.error("base URL must contain only a loopback host and optional port")
    if args.out_dir.exists():
        parser.error("output directory already exists; choose a fresh path")

    excluded_hashes, blocked = {}, set()
    for path in args.exclude:
        excluded_hashes[str(path)] = sha256_file(path)
        for example in load_examples(path):
            blocked.update(overlap_keys(example))
    training = json.loads((args.adapter / "training_manifest.json").read_text())
    for key in ("train_sha256", "val_sha256"):
        if training[key] not in excluded_hashes.values():
            parser.error(f"exclusions do not cover the adapter's {key}")
    reference = load_examples(args.reference)
    selected, excluded = select_examples(reference, blocked, args.per_primitive, args.seed)
    for e in selected:
        if e.target_origin != "teacher-soft" or type(e.provenance.get("gold_label")) is not int:
            parser.error("reference requires teacher distributions and recorded gold labels")
        if not 0 <= e.provenance["gold_label"] < len(e.target):
            parser.error("gold label outside answer range")

    base = args.base_url.rstrip("/")
    discovery = http_json(base + "/v1/models")
    if len(discovery["data"]) != 1:
        parser.error("expected exactly one served model")
    model = discovery["data"][0]
    if model["adapter_sha256"] != sha256_file(args.adapter / "adapters.safetensors"):
        parser.error("server is not using the expected adapter")
    if model["backbone"] != training["arguments"]["model"]:
        parser.error("server backbone differs from adapter training backbone")

    cases = [{"case": f"B{i:02}", "example": e.to_dict(), "request": {
        "model": "jev-latest", "state": e.state, "questions": {"q": asdict(e.question)},
    }} for i, e in enumerate(selected, 1)]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "cases.jsonl").write_text("".join(dumps(c) + "\n" for c in cases))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "SHA256(seed:id) ascending; equal primitive quotas; Choice split equally by source; no prediction-based selection",
        "seed": args.seed, "per_primitive": args.per_primitive,
        "reference_path": str(args.reference), "reference_sha256": sha256_file(args.reference),
        "excluded_file_hashes": excluded_hashes, "reference_examples": len(reference),
        "overlap_excluded": excluded, "selected": len(cases),
        "matching": "canonical IDs/groups/normalized content plus known-wrapper 1500-character prefixes; not fuzzy/pretraining decontamination",
        "cases_sha256": sha256_file(args.out_dir / "cases.jsonl"),
        "script_sha256": sha256_file(__file__), "environment": environment_identity(),
        "base_url": base, "server": model,
        "teacher_versions": sorted({e.provenance["teacher_resolved_model"] for e in selected
                                    if e.provenance.get("teacher_resolved_model")}),
        "unversioned_teacher_examples": sum(not e.provenance.get("teacher_resolved_model") for e in selected),
        "limitations": "Historical cached teacher, rounded probabilities and truncated inputs; tiny in-family diagnostic, not current Jev parity or calibration certification. No reserved Kev test opened.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    with (args.out_dir / "predictions.jsonl").open("x") as handle:
        for case, example in zip(cases, selected):
            start = time.perf_counter()
            response = http_json(base + "/v1/systemone", case["request"])
            elapsed = time.perf_counter() - start
            if response["model"] != model["id"] or set(response["answers"]) != {"q"}:
                raise ValueError("server identity or answer IDs changed during benchmark")
            row = {
                "case": case["case"], "id": example.id, "source": example.source,
                "primitive": example.question.type, "answer_keys": example.question.answer_keys,
                "gold_label": example.provenance["gold_label"],
                "teacher_probabilities": example.target,
                "local_probabilities": local_probabilities(example.question, response["answers"]["q"]),
                "elapsed_ms": elapsed * 1000, "response": response,
            }
            rows.append(row)
            handle.write(dumps(row) + "\n")
            handle.flush()
            print(f"{case['case']}: {example.question.type}, {elapsed * 1000:.0f} ms", flush=True)
    if http_json(base + "/v1/models") != discovery:
        raise ValueError("server configuration changed during benchmark")

    def summarize(group):
        return {"local": metrics(group, "local_probabilities"),
                "cached_jev": metrics(group, "teacher_probabilities"), "fidelity": fidelity(group)}

    report = {
        "overall": summarize(rows),
        "by_primitive": {p: summarize([r for r in rows if r["primitive"] == p])
                         for p in ("choice", "noul", "score")},
        "local_http_latency_ms": {"median": statistics.median(r["elapsed_ms"] for r in rows),
                                  "min": min(r["elapsed_ms"] for r in rows),
                                  "max": max(r["elapsed_ms"] for r in rows)},
        "latency_note": "One request per example, serial, no explicit warmup, existing loaded server; no teacher latency measurement.",
        "metric_note": "First-index argmax ties; NLL clips at 1e-12, teacher rounding can create zeros; multiclass Brier is not divided by label count; Score MAE uses probability-weighted zero-based index. No ECE claim from this tiny set.",
        "manifest_sha256": sha256_file(args.out_dir / "manifest.json"),
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
