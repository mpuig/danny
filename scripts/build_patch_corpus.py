"""Assemble the decision-26 patch corpus from teacher-scored patch batches.

Inputs are canonical teacher-scored examples (score_scenarios + evidence pairs).
Applies, in order:
  1. ordinality exclusion for Score rows (audit_ordinality verdicts required);
  2. the filter-v2 overconfidence rule on missing-evidence cells;
  3. the evidence-removal analogue: drop REMOVED-side rows where the teacher is
     >= --max-prob confident in a substantive (non-unknown) option — supervision
     that contradicts "confidence collapses when evidence leaves";
  4. leakage check against every partition the target adapter has seen.

    uv run python scripts/build_patch_corpus.py \
        --scored data/experiments/patch-v1/score_examples.jsonl \
                 data/experiments/patch-v1/evidence_examples.jsonl \
        --ordinality-verdicts data/experiments/patch-v1/ordinality_verdicts.jsonl \
        --reference data/experiments/synthfiltered-corpus \
        --out data/experiments/patch-v1/patch_corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev.data import load_examples, sha256_file
from filter_synthetic import UNKNOWN_OPTION, rubric_key


def confident_in_substantive(example, threshold: float) -> bool:
    top = max(example.target)
    if top < threshold:
        return False
    if example.question.type != "choice":
        return True  # noul/score have no explicit-unknown escape
    options = list(example.question.criteria)
    argmax = max(range(len(example.target)), key=example.target.__getitem__)
    return not UNKNOWN_OPTION.search(options[argmax])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", nargs="+", required=True)
    ap.add_argument("--ordinality-verdicts", required=True)
    ap.add_argument("--reference", required=True,
                    help="partition dir whose train/dev/calibration/test must stay disjoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-prob", type=float, default=0.99)
    args = ap.parse_args()

    nonordinal = set()
    for line in open(args.ordinality_verdicts):
        verdict = json.loads(line)
        if verdict["verdict"] in ("categorical", "mixed_unknown_tail"):
            nonordinal.add(verdict["rubric_key"])

    reference_keys: set[str] = set()
    for name in ("train", "development", "calibration", "test"):
        path = Path(args.reference) / f"{name}.jsonl"
        if path.exists():
            for example in load_examples(path):
                reference_keys |= example.leakage_keys

    kept, counts = [], {"nonordinal": 0, "overconfident_missing": 0,
                        "overconfident_removed": 0, "leaked": 0}
    for path in args.scored:
        for example in load_examples(path):
            cell = example.provenance["cell"]
            generator = example.provenance["generator"]
            if example.question.type == "score" and rubric_key(
                {"instructions": example.question.instructions,
                 "criteria": example.question.criteria}) in nonordinal:
                counts["nonordinal"] += 1
                continue
            if (cell.get("ambiguity") == "missing_evidence"
                    and confident_in_substantive(example, args.max_prob)):
                counts["overconfident_missing"] += 1
                continue
            removed_side = (generator.get("pair_kind") == "evidence_removal"
                            and example.id.endswith("-b"))
            if removed_side and confident_in_substantive(example, args.max_prob):
                counts["overconfident_removed"] += 1
                continue
            own = {"example:" + example.id, "group:" + example.group_id}
            if (example.leakage_keys - own) & reference_keys:
                counts["leaked"] += 1
                continue
            kept.append(example)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for example in kept:
            handle.write(json.dumps(example.to_dict()) + "\n")

    print(json.dumps({
        "kept": len(kept),
        "dropped": counts,
        "sources": {p: sha256_file(p) for p in args.scored},
        "out_sha256": sha256_file(out),
    }, indent=2))


if __name__ == "__main__":
    main()
