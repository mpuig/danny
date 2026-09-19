"""Freeze synthetic held-out rubrics with explicit provenance and leakage checks.

Small diagnostic suite only: hand-authored labels need independent review and
cannot establish population-level generalization. Never use it for training/calibration.
"""

import argparse
import json
from pathlib import Path

from jev.data import (
    Example,
    assert_disjoint,
    load_examples,
    sha256_file,
    partition_summary,
)
from jev.schema import Question
from jev.serialization import dumps, loads


def build(fixture: Path) -> list[Example]:
    spec = loads(fixture.read_text())
    digest = sha256_file(fixture)
    rows = []
    for family in spec["families"]:
        question = Question(**family["question"])
        for index, (text, label) in enumerate(family["cases"]):
            if type(label) is not int or not 0 <= label < len(question.answer_keys):
                raise ValueError("invalid fixture label")
            identity = f"rubric-v1:{family['name']}:{index}"
            rows.append(
                Example(
                    id=identity,
                    group_id=f"rubric-v1:{family['name']}:{index // 3}",
                    source=f"rubric:{family['name']}",
                    state={"ticket": {"message": text}},
                    question=question,
                    target=[
                        float(i == label) for i in range(len(question.answer_keys))
                    ],
                    target_origin="gold",
                    provenance={
                        "fixture_sha256": digest,
                        "author": "project synthetic fixture",
                        "review_status": "requires independent review",
                        "split": "rubric-holdout",
                        "limitations": spec["description"],
                    },
                )
            )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fixture", type=Path, default=Path("tests/fixtures/heldout_rubrics.json")
    )
    ap.add_argument(
        "--against",
        nargs="+",
        default=[
            "data/kev-v1/train.jsonl",
            "data/kev-v1/development.jsonl",
            "data/kev-v1/calibration.jsonl",
        ],
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists():
        ap.error("output directory exists")
    rows = build(args.fixture)
    assert_disjoint({"holdout": rows, **{p: load_examples(p) for p in args.against}})
    args.out_dir.mkdir(parents=True, exist_ok=False)
    out = args.out_dir / "rubrics.jsonl"
    out.write_text("".join(dumps(row.to_dict()) + "\n" for row in rows))
    report = {
        "fixture_sha256": sha256_file(args.fixture),
        "data_sha256": sha256_file(out),
        "overlap_checked": {p: sha256_file(p) for p in args.against},
        **partition_summary(rows),
        "policy": "Frozen diagnostic holdout. No training, temperature fitting, or tuning against these cases.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
