"""Build training/validation JSONL for the LoRA phase.

Each line: {"prompt": str, "labels": [" A", ...], "target": [p, ...], "task": str}
The prompt is rendered with the same templates the engine uses at inference
(noul routed through the choice template), phrasings are mixed for diversity,
and the target is the label distribution (one-hot for these datasets).

    uv run python scripts/build_data.py --per-task 2000 --val-per-task 200
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from jev.engine import _LETTERS, SystemOneEngine
from jev.recast import TASKS, TRAIN_TASKS, records
from jev.schema import Question


def render(rec: dict, rng: random.Random | None = None) -> dict:
    """Render a training row. With an rng, applies robustness augmentation:
    choice/noul options are shuffled (target remapped) so the model must read
    the option text rather than learn letter positions, and ~20% of choice
    descriptions are dropped. Score levels are ordered and never shuffled."""
    q = rec["question"]
    label = rec["label"]

    if q.type == "score":
        prompt = SystemOneEngine._score_prompt(rec["state"], q)
        n_opts = len(q.criteria)
    else:
        if q.type == "noul":
            q = SystemOneEngine._as_yes_no_choice(q)
        names = list(q.criteria.keys())
        if rng is not None:
            order = list(range(len(names)))
            rng.shuffle(order)
            label = order.index(label)
            criteria = {names[i]: q.criteria[names[i]] for i in order}
            if rec["question"].type == "choice" and rng.random() < 0.2:
                criteria = {name: None for name in criteria}
            q = Question(type="choice", instructions=q.instructions, criteria=criteria)
        prompt, _ = SystemOneEngine._choice_prompt(rec["state"], q)
        n_opts = len(q.criteria)

    target = [0.0] * n_opts
    target[label] = 1.0
    return {
        "prompt": prompt,
        "labels": [f" {_LETTERS[i]}" for i in range(n_opts)],
        "target": target,
        "task": rec["task"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=2000)
    ap.add_argument("--val-per-task", type=int, default=200)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    train_rows, val_rows = [], []
    for name in TRAIN_TASKS:
        task = TASKS[name]
        total = args.per_task + args.val_per_task
        recs = []
        # cycle phrasings across shuffled draws for question-wording diversity
        for i, rec in enumerate(
            records(task, split=task.train_split, n=total, seed=args.seed)
        ):
            rec = dict(rec, question=task.question(phrasing=i % len(task.phrasings)))
            recs.append(render(rec, rng=rng))
        rng.shuffle(recs)
        val_rows += recs[: args.val_per_task]
        train_rows += recs[args.val_per_task :]
        print(f"{name}: {len(recs) - args.val_per_task} train / {args.val_per_task} val")

    rng.shuffle(train_rows)
    for path, rows in ((out / "train.jsonl", train_rows), (out / "val.jsonl", val_rows)):
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
