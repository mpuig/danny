"""Measure option-order sensitivity: rerun each example with shuffled options
and compare the (un-shuffled) distributions. A perfectly robust model shows
flip_rate 0 and tv_distance 0.

    uv run python scripts/permutation_test.py --model ... --task ag_news --n 100
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
from tqdm import tqdm

from jev.engine import SystemOneEngine
from jev.recast import TASKS, records
from jev.schema import Question
from jev.rendering import RENDERER_VERSIONS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=[
        name for name, t in TASKS.items() if t.question_type == "choice"
    ])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--renderer", choices=RENDERER_VERSIONS, default=None)
    args = ap.parse_args()

    engine = SystemOneEngine(
        args.model, contextual_calibration=args.calibrate, adapter_path=args.adapter,
        renderer_version=args.renderer,
    )
    task = TASKS[args.task]
    rng = random.Random(args.seed)

    flips, tvs = 0, []
    for rec in tqdm(list(records(task, n=args.n)), desc="permutation"):
        q = rec["question"]
        names = list(q.criteria.keys())
        base = engine.ask(rec["state"], {"q": q})["q"].probabilities

        order = names[:]
        while order == names:
            rng.shuffle(order)
        shuffled_q = Question(
            type="choice",
            instructions=q.instructions,
            criteria={name: q.criteria[name] for name in order},
        )
        perm = engine.ask(rec["state"], {"q": shuffled_q})["q"].probabilities

        p = np.array([base[name] for name in names])
        s = np.array([perm[name] for name in names])
        flips += int(p.argmax() != s.argmax())
        tvs.append(0.5 * np.abs(p - s).sum())

    print(
        json.dumps(
            {
                "model": args.model,
                "adapter": args.adapter,
                "renderer_version": engine.renderer_version,
                "task": args.task,
                "calibrated": args.calibrate,
                "n": args.n,
                "argmax_flip_rate": flips / args.n,
                "mean_tv_distance": float(np.mean(tvs)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
