"""Numerical and runtime measurements, not a model-quality benchmark.

Each shared result is compared with independent execution at the SAME precision.
Cold model loading is excluded. Memory is MLX peak active allocation, not process RSS.
"""
import argparse
import gc
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from jev.engine import SystemOneEngine
from jev.schema import Question


def fixture(repetitions=1, count=3):
    state = {"ticket": {"messages": ["The package arrived damaged. Please refund it."] * repetitions}}
    questions = [
        Question("noul", "Does `ticket.messages` request a refund?"),
        Question("choice", "Which department fits?", {"returns": {"covers": ["refunds", "damaged products"]}, "sales": "New purchases", "account": None}),
        Question("score", "How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
    ]
    return state, {str(i): questions[i % 3] for i in range(count)}


def measure(engine, items, repeats):
    engine._score_batch(items)  # warm up; tolist synchronizes Metal
    mx.reset_peak_memory()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        probs = engine._score_batch(items)
        times.append(time.perf_counter() - start)
    return probs, {
        "p50_ms": float(np.percentile(times, 50) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000),
        "questions_per_second": len(items) / float(np.mean(times)),
        "peak_active_bytes": mx.get_peak_memory(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--precisions", nargs="+", default=["native", "float16", "float32"], choices=["native", "float16", "float32"])
    ap.add_argument("--counts", nargs="+", type=int, default=[3, 12])
    ap.add_argument("--state-repetitions", nargs="+", type=int, default=[1, 16])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    if min(args.counts + args.state_repetitions + [args.repeats]) < 1:
        ap.error("counts, lengths, and repeats must be positive")
    out = Path(args.out)
    if out.exists():
        ap.error("output exists")
    results = []
    for precision in args.precisions:
        engine = SystemOneEngine(args.model, adapter_path=args.adapter, precision=precision)
        for length in args.state_repetitions:
            for count in args.counts:
                state, questions = fixture(length, count)
                items = [engine._render(state, q) for q in questions.values()]
                engine.execution_mode = "independent"
                reference, single_stats = measure(engine, items, args.repeats)
                engine.execution_mode = "shared"
                shared, shared_stats = measure(engine, items, args.repeats)
                reverse = list(reversed(engine._score_batch(list(reversed(items)))))
                row = {
                    "precision": precision, "questions": count, "state_repetitions": length,
                    "max_prompt_tokens": max(len(engine.tokenizer.encode(p)) for p, _ in items),
                    "max_probability_drift": max(abs(a-b) for x,y in zip(reference, shared) for a,b in zip(x,y)),
                    "max_reorder_drift": max(abs(a-b) for x,y in zip(shared, reverse) for a,b in zip(x,y)),
                    "argmax_flips": sum(np.argmax(a) != np.argmax(b) for a,b in zip(reference,shared)),
                    "independent": single_stats, "shared": shared_stats,
                }
                row["argmax_flips"] = int(row["argmax_flips"])
                results.append(row)
                print(json.dumps(row), flush=True)
        del engine
        gc.collect()
        mx.clear_cache()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump({"arguments": vars(args), "platform": platform.platform(), "results": results,
                   "limitations": "Synthetic fixtures, warm engine latency (not HTTP); MLX active memory excludes RSS. Small repeat count is not a production p95 estimate."}, handle, indent=2)


if __name__ == "__main__":
    main()
