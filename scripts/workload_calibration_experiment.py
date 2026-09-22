"""Per-workload temperature fitting on spent reserved-v2 predictions (decision 31).

Analysis-only: no model is loaded or selected. For each (model, workload, k),
resample k fit rows, fit a scalar temperature, and score the workload's
remaining rows under four arms: raw, the in-family serving temperatures,
pooled-OOD (one temperature over the union of the resample's fit samples), and
per-workload. Reports means and percentile intervals over resamples, paired
per-resample deltas, and coverage/error operating points together.

    uv run python scripts/workload_calibration_experiment.py \
        --out data/runs/workload-calibration-v1/report.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from jev.calibration import fit_temperature, temperature_scale
from jev.data import read_jsonl, sha256_file
from jev.metrics import expected_calibration_error
from jev.provenance import environment_identity
from jev.serialization import loads

MODELS = {
    "minicpm-q8": {
        "predictions": "data/runs/reserved-v2/q8-raw",
        "serving_temperature": "data/runs/quant-v1/temperature-q8.json",
    },
    "qwen-0.6b": {
        "predictions": "data/runs/reserved-v2/qwen-raw",
        "serving_temperature": "data/runs/filtered-v2/temperature.json",
    },
}
THRESHOLDS = (0.9, 0.95)


def scaled_rows(rows, temperature):
    return [temperature_scale(r["probabilities"], temperature) for r in rows]


def evaluate(prob_rows, rows):
    """Metrics over variable-width one-hot rows."""
    top = np.array([max(p) for p in prob_rows])
    correct = np.array(
        [int(np.argmax(p) == r["target"].index(1.0)) for p, r in zip(prob_rows, rows)]
    )
    p_true = np.array(
        [p[r["target"].index(1.0)] for p, r in zip(prob_rows, rows)]
    )
    brier = float(
        np.mean(
            [
                sum((pi - ti) ** 2 for pi, ti in zip(p, r["target"]))
                for p, r in zip(prob_rows, rows)
            ]
        )
    )
    out = {
        "n": len(rows),
        "nll": float(np.mean(-np.log(np.maximum(p_true, 1e-12)))),
        "brier": brier,
        "ece": expected_calibration_error(top, correct.astype(float)),
        "accuracy": float(correct.mean()),
    }
    for t in THRESHOLDS:
        mask = top >= t
        out[f"coverage@{t}"] = float(mask.mean())
        out[f"errors@{t}"] = (
            float(1 - correct[mask].mean()) if mask.any() else None
        )
    return out


def serving_temperature_for(row, fits):
    return fits[row["primitive"]]["temperature"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fit-sizes", type=int, nargs="+", default=[25, 50, 100])
    ap.add_argument("--resamples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("fresh output path required")

    report = {
        "decision": 31,
        "environment": environment_identity(),
        "arguments": {**vars(args), "out": str(args.out)},
        "inputs": {},
        "models": {},
        "limitations": (
            "Post-hoc analysis of spent reserved-v2 predictions; a scalar temperature "
            "cannot change accuracy or fix rank errors. Operating points must be read "
            "as coverage/error pairs. Demonstration fits are never served."
        ),
    }

    for model, spec in MODELS.items():
        pred_dir = Path(spec["predictions"])
        rows = list(read_jsonl(pred_dir / "predictions.jsonl"))
        run_report = loads((pred_dir / "report.json").read_text())
        if run_report["arguments"].get("temperature"):
            raise ValueError(f"{model}: predictions must be raw")
        serving = loads(Path(spec["serving_temperature"]).read_text())
        report["inputs"][model] = {
            "predictions_sha256": sha256_file(pred_dir / "predictions.jsonl"),
            "data_sha256": run_report["data_sha256"],
            "serving_temperature_sha256": sha256_file(spec["serving_temperature"]),
        }
        by_workload = defaultdict(list)
        for row in rows:
            by_workload[row["source"].removeprefix("reserved-v2:")].append(row)

        model_out = {"workloads": {}, "fitted_temperatures": {}}
        for k in args.fit_sizes:
            for workload in sorted(by_workload):
                pool = by_workload[workload]
                if k >= len(pool) - 20:
                    continue
                arm_metrics = defaultdict(list)
                deltas = []  # per-resample NLL: per_workload - serving
                temps = []
                for r_i in range(args.resamples):
                    # string seeds hash deterministically in random.Random,
                    # unlike tuple/str __hash__ which is salted per process
                    rng = random.Random(f"{args.seed}:{model}:{workload}:{k}:{r_i}")
                    fit_idx = set(rng.sample(range(len(pool)), k))
                    fit_rows = [pool[i] for i in fit_idx]
                    eval_rows = [p for i, p in enumerate(pool) if i not in fit_idx]
                    # pooled arm shares the SAME per-resample draw across workloads
                    pooled_fit = []
                    for other, other_pool in by_workload.items():
                        rng_o = random.Random(f"{args.seed}:{model}:{other}:{k}:{r_i}")
                        idx_o = set(rng_o.sample(range(len(other_pool)), min(k, len(other_pool) - 20)))
                        pooled_fit += [other_pool[i] for i in idx_o]
                    t_local = fit_temperature(fit_rows)["temperature"]
                    t_pooled = fit_temperature(pooled_fit)["temperature"]
                    temps.append(t_local)
                    arms = {
                        "raw": [r["probabilities"] for r in eval_rows],
                        "serving": [
                            temperature_scale(
                                r["probabilities"],
                                serving_temperature_for(r, serving["fits"]),
                            )
                            for r in eval_rows
                        ],
                        "pooled": scaled_rows(eval_rows, t_pooled),
                        "per_workload": scaled_rows(eval_rows, t_local),
                    }
                    resample = {
                        arm: evaluate(probs, eval_rows) for arm, probs in arms.items()
                    }
                    for arm, metrics in resample.items():
                        arm_metrics[arm].append(metrics)
                    deltas.append(
                        resample["per_workload"]["nll"] - resample["serving"]["nll"]
                    )

                def agg(values):
                    a = np.array([v for v in values if v is not None], dtype=float)
                    if not len(a):
                        return None
                    return {
                        "mean": float(a.mean()),
                        "interval_95": [
                            float(np.percentile(a, 2.5)),
                            float(np.percentile(a, 97.5)),
                        ],
                    }

                summary = {
                    arm: {
                        metric: agg([m[metric] for m in ms])
                        for metric in ms[0]
                        if metric != "n"
                    }
                    for arm, ms in arm_metrics.items()
                }
                d = np.array(deltas)
                summary["nll_delta_per_workload_minus_serving"] = {
                    "mean": float(d.mean()),
                    "interval_95": [
                        float(np.percentile(d, 2.5)),
                        float(np.percentile(d, 97.5)),
                    ],
                    "interval_excludes_zero": bool(
                        np.percentile(d, 97.5) < 0 or np.percentile(d, 2.5) > 0
                    ),
                }
                model_out["workloads"].setdefault(workload, {})[f"k={k}"] = summary
                model_out["fitted_temperatures"].setdefault(workload, {})[
                    f"k={k}"
                ] = agg(temps)
        report["models"][model] = model_out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({m: v["fitted_temperatures"] for m, v in report["models"].items()}, indent=1))


if __name__ == "__main__":
    main()
