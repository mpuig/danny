"""Agreement with the real Jev on held-out states: the direct "works like Jev"
measure. Pulls Jev's distributions once (cached to disk), then compares a local
checkpoint's distributions: mean KL(ours ‖ Jev), argmax agreement, noul MAE.

    uv run python scripts/agreement_vs_jev.py --model ... --task sst2 --n 100 \
        [--adapter ...] [--calibrate]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from jev.engine import SystemOneEngine
from jev.jev_api import api_key, ask_jev
from jev.recast import TASKS, records


def jev_dist(task, answer: dict) -> list[float]:
    if task.question_type == "noul":
        p = float(answer["noul"])
        return [1.0 - p, p]
    probs = answer["probabilities"]
    if task.question_type == "score":
        return [float(probs[str(i)]) for i in range(len(task.criteria))]
    return [float(probs[name]) for name in task.criteria]


def local_dist(task, answer) -> list[float]:
    if answer.type == "noul":
        return [1.0 - answer.noul, answer.noul]
    return list(answer.probabilities.values())


def get_jev_answers(task, n: int) -> list[dict]:
    cache = Path(f"data/jev_answers_{task.name}_{n}.jsonl")
    if cache.exists():
        return [json.loads(l) for l in cache.open()]
    key = api_key()
    rows = []
    with cache.open("w") as f:
        for rec in tqdm(list(records(task, n=n)), desc=f"jev pulls ({task.name})"):
            answer = ask_jev(key, rec["state"], rec["question"])
            row = {"state": rec["state"], "label": rec["label"], "jev": answer}
            f.write(json.dumps(row) + "\n")
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    task = TASKS[args.task]
    jev_rows = get_jev_answers(task, args.n)
    engine = SystemOneEngine(
        args.model, contextual_calibration=args.calibrate, adapter_path=args.adapter
    )
    question = task.question()

    eps = 1e-9
    kls, agree, jev_acc, our_acc = [], 0, 0, 0
    for row in tqdm(jev_rows, desc="local model"):
        ours = np.array(local_dist(task, engine.ask(row["state"], {"q": question})["q"]))
        jev = np.array(jev_dist(task, row["jev"]))
        kls.append(float((ours * np.log((ours + eps) / (jev + eps))).sum()))
        agree += int(ours.argmax() == jev.argmax())
        jev_acc += int(jev.argmax() == row["label"])
        our_acc += int(ours.argmax() == row["label"])

    n = len(jev_rows)
    print(
        json.dumps(
            {
                "model": args.model,
                "adapter": args.adapter,
                "task": args.task,
                "calibrated": args.calibrate,
                "n": n,
                "mean_kl_ours_vs_jev": float(np.mean(kls)),
                "argmax_agreement_with_jev": agree / n,
                "our_accuracy": our_acc / n,
                "jev_accuracy": jev_acc / n,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
