"""Score the examples/ requests across systems against authored intended answers.

Correctness (examples/expected.json): noul = thresholded P(yes) at 0.5 matches;
choice = argmax option in the acceptable set; score = argmax level in the
acceptable set. 24 hand-labeled questions across 8 requests - an illustration
of relative quality, not a benchmark. Outputs under examples/outputs/<arm>/
are committed verbatim so this table is reproducible.

    uv run python scripts/score_examples.py [--per-question]
"""

import argparse
import json
from pathlib import Path

ARMS = [
    ("base-minicpm2b", "2B base (no FT)"),
    ("volume-qwen0.6b", "0.6B FT"),
    ("ft-minicpm2b", "2B FT"),
    ("q8-minicpm2b", "2B FT q8 (served)"),
    ("jev-1.13.0", "Jev 1.13.0"),
]


def correct(answer: dict, expected) -> bool:
    kind = answer["type"]
    if kind == "noul":
        return (answer["noul"] >= 0.5) == bool(expected)
    top = max(answer["probabilities"], key=answer["probabilities"].get)
    if kind == "score":
        return int(top) in expected
    return top in expected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-question", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1] / "examples"
    expected = {
        k: v for k, v in json.load(open(root / "expected.json")).items()
        if not k.startswith("_")
    }
    header = "| Example | " + " | ".join(label for _, label in ARMS) + " |"
    print(header)
    print("|" + "---|" * (len(ARMS) + 1))
    totals = {arm: [0, 0] for arm, _ in ARMS}
    for name in sorted(expected):
        cells = []
        for arm, _ in ARMS:
            answers = json.load(open(root / "outputs" / arm / f"{name}.json"))["answers"]
            marks = {
                qid: correct(answers[qid], want)
                for qid, want in expected[name].items()
            }
            totals[arm][0] += sum(marks.values())
            totals[arm][1] += len(marks)
            cells.append(f"{sum(marks.values())}/{len(marks)}")
            if args.per_question:
                for qid, ok in marks.items():
                    if not ok:
                        print(f"<!-- {arm} misses {name}.{qid} -->")
        print(f"| [{name}](examples/{name}.md) | " + " | ".join(cells) + " |")
    print(
        "| **Total** | "
        + " | ".join(f"**{c}/{n}**" for c, n in (totals[a] for a, _ in ARMS))
        + " |"
    )


if __name__ == "__main__":
    main()
