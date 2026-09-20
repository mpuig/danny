"""Filter teacher-overconfident synthetic rows and build a combined training corpus.

Drops synthetic examples whose generation cell matches --ambiguity and whose
teacher target max-probability is at or above --max-prob (the recorded suspect
for the rubric-NLL regression: the teacher answering near-certain on scenarios
designed to lack decisive evidence), then concatenates the base corpus with the
surviving rows. The intervention is deliberately minimal: other cells are
untouched regardless of confidence.

    uv run python scripts/filter_synthetic.py \
        --synthetic data/experiments/synthetic-v1/examples.jsonl \
        --base data/kev-v1/train.jsonl \
        --out data/experiments/synthetic-v1/train_kev_plus_synthfiltered.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from jev.data import sha256_file

# A confident answer that ITSELF expresses "cannot be determined" is correct
# behavior on a missing-evidence scenario, not teacher overconfidence.
UNKNOWN_OPTION = re.compile(
    r"unknown|cannot|can_not|insufficient|unclear|undetermin|not_determin|"
    r"indetermin|unable_to|no_evidence|not_enough|unverifiable|other|none",
    re.IGNORECASE,
)


def confident_in_unknown(row: dict) -> bool:
    question = row["question"]
    if question["type"] != "choice":
        return False  # noul/score have no explicit-unknown outcome
    options = list(question["criteria"])
    argmax = max(range(len(row["target"])), key=row["target"].__getitem__)
    return bool(UNKNOWN_OPTION.search(options[argmax]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ambiguity", default="missing_evidence")
    ap.add_argument("--max-prob", type=float, default=0.99)
    args = ap.parse_args()

    kept, dropped, exempted = [], [], []
    for line in open(args.synthetic):
        row = json.loads(line)
        cell = row["provenance"]["cell"]
        if cell["ambiguity"] == args.ambiguity and max(row["target"]) >= args.max_prob:
            if confident_in_unknown(row):
                exempted.append(row["id"])
                kept.append(line)
            else:
                dropped.append(row["id"])
        else:
            kept.append(line)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for line in open(args.base):
            handle.write(line)
        handle.writelines(kept)

    print(json.dumps({
        "dropped": len(dropped),
        "exempted_confident_in_unknown": len(exempted),
        "kept_synthetic": len(kept),
        "combined_rows": sum(1 for _ in out.open()),
        "criteria": {"ambiguity": args.ambiguity, "max_prob_at_or_above": args.max_prob},
        "synthetic_sha256": sha256_file(args.synthetic),
        "base_sha256": sha256_file(args.base),
        "out_sha256": sha256_file(out),
    }, indent=2))


if __name__ == "__main__":
    main()
