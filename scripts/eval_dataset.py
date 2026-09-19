"""Evaluate a canonical gold-labeled corpus and save per-example predictions.

    uv run python scripts/eval_dataset.py --model HuggingFaceTB/SmolLM2-135M \
        --data data/kev-v1/development.jsonl --out-dir data/evals/smoke --n 20

This first harness evaluates one question at a time. It is not a throughput test.
Teacher-soft fidelity and grouped bootstrap intervals are separate future work.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from jev.data import load_examples, sha256_file
from jev.engine import SystemOneEngine
from jev.metrics import expected_calibration_error
from jev.provenance import environment_identity, model_identity
from jev.rendering import RENDERER_VERSIONS, READOUT_VERSIONS, render_views, resolve_renderer, resolve_readout
from jev.serialization import dumps


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize empty predictions")
    return {
        "n": len(rows),
        "groups": len({r["group_id"] for r in rows}),
        "accuracy": float(np.mean([r["correct"] for r in rows])),
        "nll": float(np.mean([r["nll"] for r in rows])),
        "brier": float(np.mean([r["brier"] for r in rows])),
        "ece": expected_calibration_error(
            np.array([r["top1_probability"] for r in rows]),
            np.array([r["correct"] for r in rows]),
        ),
        "mean_top1_prob": float(np.mean([r["top1_probability"] for r in rows])),
        **({"score_mae": float(np.mean([r["score_error"] for r in rows]))}
           if all(r["primitive"] == "score" for r in rows) else {}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--precision", choices=["native", "float16", "float32"], default="native")
    parser.add_argument("--execution-mode", choices=["independent", "shared"], default="independent")
    parser.add_argument("--renderer", choices=RENDERER_VERSIONS)
    parser.add_argument("--readout", choices=READOUT_VERSIONS)
    parser.add_argument("--calibrate", action="store_true", help="contextual correction, not temperature fitting")
    parser.add_argument("--n", type=int, default=0, help="0 = all; otherwise seeded example subsample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n < 0:
        parser.error("n must be nonnegative")
    out = Path(args.out_dir)
    if out.exists():
        parser.error(f"output already exists: {out}")
    data_hash = sha256_file(args.data)
    examples = load_examples(args.data)
    if sha256_file(args.data) != data_hash:
        parser.error("input data changed while loading")
    adapter_hash = sha256_file(Path(args.adapter) / "adapters.safetensors") if args.adapter else None
    version = resolve_renderer(args.renderer, args.adapter)
    readout = resolve_readout(args.readout, args.adapter)
    # Reject unsupported supervision/readouts before loading weights.
    for example in examples:
        if example.target_origin != "gold" or sum(p == 1 for p in example.target) != 1:
            parser.error("outcome evaluation requires one-hot gold targets, not teacher distributions")
        render_views(example.state, example.question, version, readout)
    if args.n and args.n < len(examples):
        examples = random.Random(args.seed).sample(examples, args.n)
    engine = SystemOneEngine(args.model, adapter_path=args.adapter,
                             contextual_calibration=args.calibrate, renderer_version=version,
                             precision=args.precision, execution_mode=args.execution_mode, readout_version=readout)
    provenance = {"environment": environment_identity(), "backbone": model_identity(args.model)}
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    # Exclusive creation and a final report distinguish complete runs from partial
    # prediction files left by interruption. No existing run is silently reused.
    with (out / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for example in tqdm(examples, desc="evaluate"):
            answer = engine.ask(example.state, {"q": example.question})["q"]
            keys = example.question.answer_keys
            probs = ([1 - answer.noul, answer.noul] if answer.type == "noul" else
                     [answer.probabilities[key] for key in keys])
            label = example.target.index(1.0)
            predicted = max(range(len(probs)), key=probs.__getitem__)
            row = {
                "id": example.id, "group_id": example.group_id, "source": example.source,
                "primitive": example.question.type, "answer_keys": keys,
                "target": example.target, "probabilities": probs, "answer": asdict(answer),
                "correct": predicted == label, "top1_probability": max(probs),
                "nll": -math.log(max(probs[label], 1e-12)),
                "brier": sum((p - t) ** 2 for p, t in zip(probs, example.target)),
            }
            if answer.type == "score":
                row["score_error"] = abs(answer.score - label)
            rows.append(row)
            handle.write(dumps(row) + "\n")
            handle.flush()
    by_source, by_primitive = defaultdict(list), defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
        by_primitive[row["primitive"]].append(row)
    report = {
        "arguments": vars(args), "renderer_version": engine.renderer_version,
        "readout_version": engine.readout_version,
        "data_sha256": data_hash,
        "adapter_sha256": adapter_hash,
        **provenance,
        "overall_micro": summarize_rows(rows),
        "by_source": {key: summarize_rows(group) for key, group in sorted(by_source.items())},
        "by_primitive": {key: summarize_rows(group) for key, group in sorted(by_primitive.items())},
        "limitations": "Example-level point estimates; variants share groups; no confidence intervals. n subsamples examples, not groups.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
