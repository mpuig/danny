"""Data-quality gate for teacher-scored synthetic examples.

Checks, in order: canonical validation, leakage against the kev-v1 partitions and
the frozen rubric holdout, diversity coverage against the generation plan, teacher
softness (overall and by ambiguity cell), and generator/teacher argmax agreement.
Prints a JSON report and a PASS/FAIL verdict with the reasons.

    uv run python scripts/audit_synthetic.py data/experiments/synthetic-v1/examples.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from jev.data import content_key, load_examples
from jev.serialization import dumps

KEV_PARTITIONS = ("train", "development", "calibration", "test")
RUBRIC_FIXTURE = Path("tests/fixtures/heldout_rubrics.json")

GATES = {
    "validation": "all rows load as canonical examples with unique ids",
    "leakage": "zero content/group overlap with kev-v1 partitions and rubric holdout",
    "coverage": "every planned axis value is populated",
    "softness": "more than 25% of rows have teacher max-prob < 0.99",
    "agreement": "generator/teacher argmax agreement within 0.60-0.97",
}


def leakage_reference_keys(kev_dir: Path) -> set[str]:
    keys: set[str] = set()
    for name in KEV_PARTITIONS:
        path = kev_dir / f"{name}.jsonl"
        if path.exists():
            for example in load_examples(path):
                keys |= example.leakage_keys
    if RUBRIC_FIXTURE.exists():
        fixture = json.loads(RUBRIC_FIXTURE.read_text())
        for family in fixture.get("families", []):
            for case in family.get("cases", []):
                text = case[0]
                keys.add(content_key(text))
                keys.add(content_key({"ticket": {"message": text}}))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("examples")
    ap.add_argument("--kev-dir", default="data/kev-v1")
    args = ap.parse_args()

    failures: list[str] = []
    examples = load_examples(args.examples)  # validates every row, unique ids
    n = len(examples)

    # leakage
    reference = leakage_reference_keys(Path(args.kev_dir))
    leaks = [e.id for e in examples if (e.leakage_keys - {"example:" + e.id, "group:" + e.group_id}) & reference]
    if leaks:
        failures.append(f"leakage: {len(leaks)} rows overlap reference data (e.g. {leaks[:3]})")

    # coverage
    def tally(key: str) -> dict:
        return dict(Counter(e.provenance["cell"].get(key) for e in examples))

    coverage = {k: tally(k) for k in
                ("primitive", "length", "state_format", "style", "ambiguity", "register")}
    plan = {"primitive": 3, "length": 3, "state_format": 4, "style": 6, "ambiguity": 3, "register": 5}
    for axis, expected in plan.items():
        present = {k for k, v in coverage[axis].items() if k and v > 0}
        if len(present) < expected:
            failures.append(f"coverage: axis {axis} has {len(present)}/{expected} values")
    pair_rows = sum(1 for e in examples if e.id.endswith(("-a", "-b")))
    wide_rows = sum(1 for e in examples
                    if e.question.type == "choice" and len(e.question.criteria) > 10)
    long_chars = sorted(len(dumps(e.state)) for e in examples)
    if pair_rows == 0:
        failures.append("coverage: no contrastive pairs")
    if wide_rows == 0:
        failures.append("coverage: no wide-choice rows")

    # teacher softness
    max_probs = [max(e.target) for e in examples]
    soft_fraction = sum(1 for p in max_probs if p < 0.99) / n
    if soft_fraction <= 0.25:
        failures.append(f"softness: only {soft_fraction:.1%} rows below 0.99 max-prob")
    by_ambiguity: dict[str, list[float]] = defaultdict(list)
    for e, p in zip(examples, max_probs):
        by_ambiguity[e.provenance["cell"]["ambiguity"]].append(p)
    softness = {
        amb: {
            "n": len(vals),
            "mean_max_prob": round(sum(vals) / len(vals), 3),
            "at_or_above_0.99": round(sum(1 for v in vals if v >= 0.99) / len(vals), 3),
        }
        for amb, vals in sorted(by_ambiguity.items())
    }

    # generator/teacher agreement
    def generator_argmax(e) -> int:
        expected = e.provenance["generator"]["expected"]
        if e.question.type == "choice":
            return list(e.question.criteria).index(expected) if expected in e.question.criteria else -1
        return int(expected)

    agree_flags = [max(range(len(e.target)), key=e.target.__getitem__) == generator_argmax(e)
                   for e in examples]
    agreement = sum(agree_flags) / n
    if not 0.60 <= agreement <= 0.97:
        failures.append(f"agreement: {agreement:.3f} outside [0.60, 0.97]")
    agreement_by_ambiguity = {
        amb: round(sum(f for e, f in zip(examples, agree_flags)
                       if e.provenance["cell"]["ambiguity"] == amb)
                   / len(vals), 3)
        for amb, vals in sorted(by_ambiguity.items())
    }

    report = {
        "examples": n,
        "groups": len({e.group_id for e in examples}),
        "coverage": coverage,
        "pair_rows": pair_rows,
        "wide_choice_rows": wide_rows,
        "state_chars": {"min": long_chars[0], "median": long_chars[n // 2], "max": long_chars[-1]},
        "soft_fraction_below_0.99": round(soft_fraction, 3),
        "teacher_softness_by_ambiguity": softness,
        "generator_teacher_agreement": round(agreement, 3),
        "agreement_by_ambiguity": agreement_by_ambiguity,
        "leaked_rows": len(leaks),
        "verdict": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
