"""Eval: run a model through the readout on a recast task and report calibration.

    uv run python scripts/eval_baseline.py --model mlx-community/SmolLM3-3B-Base-bf16 \
        --task sst2 --n 200 --calibrate [--adapter adapters/run1]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from tqdm import tqdm

from jev.engine import SystemOneEngine
from jev.metrics import summarize, summarize_binary
from jev.recast import TASKS, records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--calibrate", action="store_true", help="contextual calibration")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter directory")
    args = ap.parse_args()

    engine = SystemOneEngine(
        args.model, contextual_calibration=args.calibrate, adapter_path=args.adapter
    )
    task = TASKS[args.task]
    recs = list(records(task, n=args.n))

    rows, labels, scores = [], [], []
    for rec in tqdm(recs, desc=f"{args.task} ({args.model})"):
        answer = engine.ask(rec["state"], {"q": rec["question"]})["q"]
        if answer.type == "noul":
            rows.append(answer.noul)
        else:
            rows.append(list(answer.probabilities.values()))
            if answer.type == "score":
                scores.append(answer.score)
        labels.append(rec["label"])

    labels_arr = np.array(labels)
    if task.question_type == "noul":
        report = summarize_binary(np.array(rows), labels_arr)
    else:
        report = summarize(np.array(rows), labels_arr)
    if scores:
        report["score_mae"] = float(np.abs(np.array(scores) - labels_arr).mean())

    print(
        json.dumps(
            {
                "model": args.model,
                "adapter": args.adapter,
                "task": args.task,
                "calibrated": args.calibrate,
                **report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
