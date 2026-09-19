"""Convert kev's frozen datasets (evals/public-pool-v4 et al.) into our training
JSONL. kev rows are Jev-shaped requests with per-question `label` fields.

Guards:
- excludes sst5 (shares Stanford Sentiment Treebank sentences with our held-out
  sst2 -> contamination) unless --include-sst5
- skips choice questions with > 26 options (v0 letter-label cap; banking77)

    uv run python scripts/convert_kev.py data/external/kev_pp4_train.jsonl \
        --out data/kev_train.jsonl
"""

from __future__ import annotations

import argparse
import json

from jev.engine import _LETTERS, SystemOneEngine
from jev.schema import Question

CONTAMINATING_SOURCES = {"sst5"}


def convert_question(state: str, spec: dict) -> dict | None:
    qtype = spec["type"]
    criteria = spec.get("criteria")
    label = spec["label"]

    if qtype == "choice":
        if len(criteria) > 26:
            return None
        names = list(criteria.keys())
        target_idx = names.index(label) if isinstance(label, str) else int(label)
        q = Question(type="choice", instructions=spec["instructions"], criteria=criteria)
        prompt, _ = SystemOneEngine._choice_prompt(state, q)
        n = len(names)
    elif qtype == "score":
        q = Question(type="score", instructions=spec["instructions"], criteria=list(criteria))
        prompt = SystemOneEngine._score_prompt(state, q)
        target_idx = int(label)
        n = len(criteria)
    else:  # noul: our render order is [no, yes]
        q = Question(type="noul", instructions=spec["instructions"], criteria=criteria)
        prompt, _ = SystemOneEngine._choice_prompt(state, SystemOneEngine._as_yes_no_choice(q))
        target_idx = 1 if label in (True, "yes", 1, "true") else 0
        n = 2

    target = [0.0] * n
    target[target_idx] = 1.0
    return {
        "prompt": prompt,
        "labels": [f" {_LETTERS[i]}" for i in range(n)],
        "target": target,
        "task": f"kev:{spec.get('src', '?')}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-sst5", action="store_true")
    args = ap.parse_args()

    kept, skipped_wide, skipped_contam = 0, 0, 0
    with open(args.out, "w") as out:
        for line in open(args.input):
            row = json.loads(line)
            for spec in row["questions"].values():
                if not args.include_sst5 and spec.get("src") in CONTAMINATING_SOURCES:
                    skipped_contam += 1
                    continue
                converted = convert_question(row["state"], spec)
                if converted is None:
                    skipped_wide += 1
                    continue
                out.write(json.dumps(converted) + "\n")
                kept += 1
    print(
        f"kept {kept}, skipped {skipped_wide} (>26 options), "
        f"{skipped_contam} (contaminating sources) -> {args.out}"
    )


if __name__ == "__main__":
    main()
