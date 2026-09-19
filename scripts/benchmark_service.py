"""Launch an owned local server and measure HTTP latency, load rejection, and memory.

Synthetic repeated requests, not model quality or a production capacity certificate.
Each case has new rubrics, so its first request misses contextual-prior caches.
"""

from __future__ import annotations

import argparse
import http.client
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from jev.provenance import environment_identity

ROOT = Path(__file__).resolve().parents[1]


def fixture(case, count, repeats, options):
    state = {
        "category": "category_1",
        "messages": [
            {
                "role": "customer",
                "content": "The package arrived broken. I need a refund. Please explain the return process.",
            }
        ]
        * repeats,
    }
    questions = {}
    for i in range(count):
        suffix = f" Benchmark fixture {case}, assessment {i}."
        kind = i % 3
        if kind == 0:
            q = {
                "type": "choice",
                "instructions": "Select the category field in the state." + suffix,
                "criteria": {f"category_{k}": None for k in range(options)},
            }
        elif kind == 1:
            q = {
                "type": "noul",
                "instructions": "Does the customer explicitly request a refund?"
                + suffix,
            }
        else:
            q = {
                "type": "score",
                "instructions": "How frustrated is the customer?" + suffix,
                "criteria": ["Calm", "Somewhat frustrated", "Very angry"],
            }
        questions[str(i)] = q
    return {"state": state, "questions": questions}


def request(port, method, path, payload=None):
    start = time.perf_counter()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        body = json.dumps(payload).encode() if payload is not None else None
        connection.request(
            method, path, body=body, headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        data = json.loads(response.read())
        return {
            "status": response.status,
            "seconds": time.perf_counter() - start,
            "response": data,
        }
    except (OSError, http.client.HTTPException) as exc:
        return {
            "status": type(exc).__name__,
            "seconds": time.perf_counter() - start,
            "response": None,
        }
    finally:
        connection.close()


def checked(port, method, path, payload=None):
    result = request(port, method, path, payload)
    if result["status"] != 200:
        raise RuntimeError(f"{path}: {result}")
    return result


def vector(answer):
    return (
        list(answer["probabilities"].values())
        if "probabilities" in answer
        else [1 - answer["noul"], answer["noul"]]
    )


def idle_metrics(port):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = checked(port, "GET", "/metrics")["response"]
        if result["active"] == 0 and result["queue_depth"] == 0:
            return result
        time.sleep(0.01)
    raise TimeoutError("worker did not become idle")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--readout", choices=["letters-v1", "candidate-v1"])
    ap.add_argument(
        "--precision", choices=["native", "float16", "float32"], default="native"
    )
    ap.add_argument(
        "--execution-mode", choices=["independent", "shared"], default="independent"
    )
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--temperature")
    ap.add_argument("--question-counts", type=int, nargs="+", default=[3, 12])
    ap.add_argument("--state-repeats", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--options", type=int, nargs="+", default=[3, 26])
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--queue-capacity", type=int, default=8)
    ap.add_argument("--max-connections", type=int, default=16)
    ap.add_argument("--tolerance", type=float, default=2e-5)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if any(
        n < 1
        for n in [
            args.requests,
            *args.question_counts,
            *args.state_repeats,
            *args.options,
            *args.concurrency,
        ]
    ):
        ap.error("counts must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    ready_path = args.out_dir.resolve() / "ready.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/serve.py"),
        "--model",
        args.model,
        "--port",
        "0",
        "--ready-file",
        str(ready_path),
        "--precision",
        args.precision,
        "--execution-mode",
        args.execution_mode,
        "--queue-capacity",
        str(args.queue_capacity),
        "--max-connections",
        str(args.max_connections),
    ]
    for flag, value in [
        ("adapter", args.adapter),
        ("readout", args.readout),
        ("temperature", args.temperature),
    ]:
        if value:
            command += ["--" + flag, str(value)]
    if args.calibrate:
        command += ["--calibrate"]
    report = {
        "environment": environment_identity(),
        "arguments": {**vars(args), "out_dir": str(args.out_dir)},
        "command": command,
        "cases": [],
        "notes": [
            "Closed-loop repeated synthetic requests, one new connection per request.",
            "First requests use new rubrics; warm samples use identical payloads.",
            "Latency quantiles are over successful requests; rejection counts are separate.",
            "Memory is cumulative high water since server startup, not incremental per-case allocation.",
            "Only the first Choice is checked alone versus its multi-question request.",
        ],
    }
    with (args.out_dir / "server.log").open("x") as log:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + 120
        while True:
            if child.poll() is not None:
                raise RuntimeError("server startup failed; see server.log")
            if time.monotonic() >= deadline:
                raise TimeoutError("server startup timed out")
            if ready_path.exists():
                try:
                    ready = json.loads(ready_path.read_text())
                    break
                except json.JSONDecodeError:
                    pass
            time.sleep(0.05)
        report["server"] = ready
        port = ready["port"]
        for case, (count, repeats, options, concurrency) in enumerate(
            itertools.product(
                args.question_counts, args.state_repeats, args.options, args.concurrency
            )
        ):
            payload = fixture(case, count, repeats, options)
            first = checked(port, "POST", "/v1/systemone", payload)
            solo = checked(
                port,
                "POST",
                "/v1/systemone",
                {
                    "state": payload["state"],
                    "questions": {"renamed": payload["questions"]["0"]},
                },
            )
            drift = max(
                abs(a - b)
                for a, b in zip(
                    vector(first["response"]["answers"]["0"]),
                    vector(solo["response"]["answers"]["renamed"]),
                )
            )
            warm = checked(port, "POST", "/v1/systemone", payload)
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                samples = list(
                    pool.map(
                        lambda _: request(port, "POST", "/v1/systemone", payload),
                        range(args.requests),
                    )
                )
            elapsed = time.perf_counter() - start
            ok = [sample["seconds"] for sample in samples if sample["status"] == 200]
            row = {
                "questions": count,
                "state_repeats": repeats,
                "options": options,
                "concurrency": concurrency,
                "requests": args.requests,
                "status_counts": dict(
                    Counter(str(sample["status"]) for sample in samples)
                ),
                "first_ms": first["seconds"] * 1000,
                "first_input_tokens": first["response"]["usage"]["input_tokens"],
                "warm_input_tokens": warm["response"]["usage"]["input_tokens"],
                "p50_ms": float(np.percentile(ok, 50) * 1000) if ok else None,
                "p95_ms": float(np.percentile(ok, 95) * 1000) if ok else None,
                "successful_requests_per_second": len(ok) / elapsed,
                "successful_questions_per_second": len(ok) * count / elapsed,
                "first_choice_single_multi_max_drift": drift,
                "parity_pass": drift <= args.tolerance,
                "metrics": idle_metrics(port),
            }
            report["cases"].append(row)
            (args.out_dir / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                json.dumps({k: v for k, v in row.items() if k != "metrics"}), flush=True
            )
        # Invalid bodies must be rejected before model work.
        oversize = request(
            port,
            "POST",
            "/v1/systemone",
            {"state": "x" * 262144, "questions": payload["questions"]},
        )
        overquestions = request(port, "POST", "/v1/systemone", fixture(999, 33, 1, 3))
        overtokens = request(port, "POST", "/v1/systemone", fixture(999, 3, 150, 3))
        report["admission_probes"] = {
            "body": oversize["status"],
            "questions": overquestions["status"],
            "state_tokens": overtokens["status"],
        }
        report["complete"] = True
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if report["admission_probes"] != {
            "body": 413,
            "questions": 422,
            "state_tokens": 422,
        }:
            raise RuntimeError("admission probe failed")
    finally:
        forced = False
        if child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                forced = True
                child.kill()
                child.wait()
        report["shutdown"] = {"forced": forced, "exit_code": child.returncode}
        report["validation_pass"] = bool(
            report.get("complete")
            and not forced
            and child.returncode == 0
            and all(case["parity_pass"] for case in report["cases"])
            and report.get("admission_probes")
            == {"body": 413, "questions": 422, "state_tokens": 422}
            and all("500" not in case["status_counts"] for case in report["cases"])
        )
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["validation_pass"]:
        raise RuntimeError("benchmark validation/shutdown failed; see report.json")


if __name__ == "__main__":
    main()
