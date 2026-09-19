"""Fit per-primitive temperatures using the declared calibration partition only.

Requires complete canonical outcome predictions, verifies IDs/labels/hashes, rejects
train/development/test predictions and recursive fitting on already scaled outputs.
A calibration file is pinned to weights, tokenizer files, renderer, readout, precision,
and execution policy. The untouched test split is never evaluated here.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from jev.calibration import fit_temperature, prediction_config
from jev.data import assert_disjoint, load_examples, read_jsonl, sha256_file
from jev.serialization import loads


def fit(predictions_dir: Path, data_manifest: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("calibration output already exists")
    manifest = loads(data_manifest.read_text())
    calibration_path = data_manifest.parent / "calibration.jsonl"
    report = loads((predictions_dir / "report.json").read_text())
    digest = sha256_file(calibration_path)
    if (
        digest != manifest["files"]["calibration.jsonl"]["sha256"]
        or report["data_sha256"] != digest
    ):
        raise ValueError(
            "predictions must come from the declared calibration partition"
        )
    if report["arguments"].get("temperature"):
        raise ValueError(
            "fit on raw predictions, not already temperature-scaled outputs"
        )
    partitions = {"calibration": load_examples(calibration_path)}
    for split in ("train", "development"):
        path = data_manifest.parent / f"{split}.jsonl"
        if sha256_file(path) != manifest["files"][path.name]["sha256"]:
            raise ValueError(f"{split} file changed since partition preparation")
        partitions[split] = load_examples(path)
    assert_disjoint(partitions)
    training = report.get("adapter_training")
    if report["adapter_sha256"] and not training:
        raise ValueError(
            "adapter training provenance is required before fitting calibration"
        )
    if training:
        if (
            training["train_sha256"] != manifest["files"]["train.jsonl"]["sha256"]
            or training["val_sha256"]
            != manifest["files"]["development.jsonl"]["sha256"]
        ):
            raise ValueError(
                "model training/development hashes do not match this partition manifest"
            )
    examples = {e.id: e for e in partitions["calibration"]}
    predictions = list(read_jsonl(predictions_dir / "predictions.jsonl"))
    if len(predictions) != len(examples) or {r["id"] for r in predictions} != set(
        examples
    ):
        raise ValueError("calibration predictions must cover each example exactly once")
    grouped = defaultdict(list)
    for row in predictions:
        e = examples[row["id"]]
        if (
            row["target"] != e.target
            or row["answer_keys"] != e.question.answer_keys
            or row["primitive"] != e.question.type
        ):
            raise ValueError(
                "prediction labels/primitive/order do not match canonical examples"
            )
        grouped[row["primitive"]].append(row)
    result = {
        "format_version": 1,
        "method": "per-primitive-temperature-v1",
        "prediction_config": prediction_config(report),
        "data_sha256": digest,
        "data_manifest_sha256": sha256_file(data_manifest),
        "predictions_sha256": sha256_file(predictions_dir / "predictions.jsonl"),
        "fits": {
            primitive: fit_temperature(rows)
            for primitive, rows in sorted(grouped.items())
        },
        "limitations": "Outcome calibration on the declared partition, not a guarantee under task/domain shift. Confidence statistics are separate.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions-dir", required=True, type=Path)
    ap.add_argument("--data-manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    try:
        result = fit(args.predictions_dir, args.data_manifest, args.out)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        ap.exit(1, f"fit_calibration: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
