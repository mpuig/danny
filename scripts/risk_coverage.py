"""Risk-coverage analysis over saved predictions: the workflow-usefulness view.

For confidence thresholds, reports coverage (fraction of questions automated at
or above the threshold), risk (error rate among automated ones), and the
accuracy of what remains for humans. This is the number a deployment actually
budgets on: "automate X% of decisions at Y% error". Uses top-1 probability as
the confidence signal; per-primitive breakdown included since thresholds should
be set per primitive and per cost model, not globally.

    uv run python scripts/risk_coverage.py data/runs/final-test/scaled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev.data import read_jsonl


def curve(rows: list[dict], thresholds: list[float]) -> list[dict]:
    n = len(rows)
    points = []
    for t in thresholds:
        automated = [r for r in rows if r["top1_probability"] >= t]
        if not automated:
            points.append({"threshold": t, "coverage": 0.0, "risk": None})
            continue
        errors = sum(1 for r in automated if not r["correct"])
        points.append({
            "threshold": t,
            "coverage": round(len(automated) / n, 4),
            "risk": round(errors / len(automated), 4),
            "automated": len(automated),
            "errors": errors,
        })
    return points


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions_dir")
    ap.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9,0.95,0.99")
    args = ap.parse_args()

    rows = list(read_jsonl(Path(args.predictions_dir) / "predictions.jsonl"))
    thresholds = [float(x) for x in args.thresholds.split(",")]

    report = {"n": len(rows), "overall": curve(rows, thresholds), "by_primitive": {}}
    for primitive in sorted({r["primitive"] for r in rows}):
        subset = [r for r in rows if r["primitive"] == primitive]
        report["by_primitive"][primitive] = {"n": len(subset), "curve": curve(subset, thresholds)}
    report["caveat"] = ("Risk is measured against this dataset's labels under its "
                       "distribution; thresholds do not transfer to shifted workloads "
                       "(see the rubric-temperature finding). Confidence = top-1 "
                       "probability, not a probability of correctness.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
