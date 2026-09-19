"""Paired outcome comparison with group-resampled confidence intervals.

Bootstrap uses whole groups (not independent questions). These intervals quantify
sampling variation within this dataset, not task-family generalization or label validity.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from jev.data import load_examples, read_jsonl, sha256_file
from jev.serialization import loads


def paired(first: Path, second: Path, samples=2000, seed=42) -> dict:
    reports = [loads((p / "report.json").read_text()) for p in (first, second)]
    if reports[0]["data_sha256"] != reports[1]["data_sha256"]:
        raise ValueError("prediction datasets differ")
    rows = [list(read_jsonl(p / "predictions.jsonl")) for p in (first, second)]
    maps = [{r["id"]: r for r in group} for group in rows]
    if (
        any(len(m) != len(r) for m, r in zip(maps, rows))
        or maps[0].keys() != maps[1].keys()
    ):
        raise ValueError("predictions must have unique, identical example IDs")
    groups = {}
    for identity, a in maps[0].items():
        b = maps[1][identity]
        if any(
            a[k] != b[k] for k in ("target", "answer_keys", "group_id", "primitive")
        ):
            raise ValueError("paired labels/order/group/primitive mismatch")
        values = []
        for r in (a, b):
            p = r["probabilities"]
            target = r["target"]
            label = target.index(1.0)
            if (
                len(p) != len(target)
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p)
                or abs(sum(p) - 1) > 1e-5
            ):
                raise ValueError("invalid probability vector")
            values.append(
                np.array(
                    [
                        float(np.argmax(p) == label),
                        -math.log(max(p[label], 1e-12)),
                        sum((x - y) ** 2 for x, y in zip(p, target)),
                    ]
                )
            )
        total = groups.setdefault(a["group_id"], np.zeros(4))
        total[:3] += values[1] - values[0]
        total[3] += 1
    group_values = np.array(list(groups.values()))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(samples, len(groups)))
    sums = group_values[draws].sum(axis=1)
    delta = sums[:, :3] / sums[:, 3:]
    observed = group_values.sum(axis=0)
    return {
        "first": str(first),
        "second": str(second),
        "examples": len(rows[0]),
        "groups": len(groups),
        "data_sha256": reports[0]["data_sha256"],
        "bootstrap_samples": samples,
        "seed": seed,
        "delta_second_minus_first": {
            name: {
                "estimate": float(observed[i] / observed[3]),
                "group_bootstrap_95_interval": np.quantile(
                    delta[:, i], [0.025, 0.975]
                ).tolist(),
            }
            for i, name in enumerate(("accuracy", "nll", "brier"))
        },
        "first_metrics": reports[0]["overall_micro"],
        "second_metrics": reports[1]["overall_micro"],
    }


def teacher_fidelity(run: Path, targets: Path) -> dict:
    examples = {e.id: e for e in load_examples(targets)}
    rows = list(read_jsonl(run / "predictions.jsonl"))
    if len(rows) != len(examples) or {r["id"] for r in rows} != set(examples):
        raise ValueError(
            "teacher targets and predictions must cover identical IDs exactly once"
        )
    divergences = []
    agreements = []
    noul_errors = []
    for row in rows:
        e = examples[row["id"]]
        if (
            row["answer_keys"] != e.question.answer_keys
            or row["primitive"] != e.question.type
        ):
            raise ValueError("teacher label order/primitive mismatch")
        p = row["probabilities"]
        q = e.target
        mixture = [(a + b) / 2 for a, b in zip(p, q)]
        js = 0.5 * sum(
            v * math.log(v / m)
            for vector in (p, q)
            for v, m in zip(vector, mixture)
            if v > 0
        )
        divergences.append(js)
        agreements.append(np.argmax(p) == np.argmax(q))
        if e.question.type == "noul":
            noul_errors.append(abs(p[1] - q[1]))
    return {
        "teacher_target_sha256": sha256_file(targets),
        "n": len(rows),
        "mean_js_nats": float(np.mean(divergences)),
        "argmax_agreement": float(np.mean(agreements)),
        "noul_mae": float(np.mean(noul_errors)) if noul_errors else None,
        "teacher_versions": sorted(
            {
                e.provenance["teacher_resolved_model"]
                for e in examples.values()
                if e.provenance.get("teacher_resolved_model")
            }
        ),
        "unversioned_examples": sum(
            e.provenance.get("teacher_resolved_model") is None
            for e in examples.values()
        ),
        "caveat": "Agreement with these cached distributions, not proof of outcome correctness or current Jev fidelity.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first", type=Path, required=True)
    ap.add_argument("--second", type=Path, required=True)
    ap.add_argument("--teacher-targets", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=2000)
    args = ap.parse_args()
    if args.out.exists() or args.samples < 1:
        ap.error("fresh output and positive sample count required")
    result = paired(args.first, args.second, args.samples)
    if args.teacher_targets:
        result["teacher_fidelity_first"] = teacher_fidelity(
            args.first, args.teacher_targets
        )
        result["teacher_fidelity_second"] = teacher_fidelity(
            args.second, args.teacher_targets
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
