"""Prepare a provenance-preserving, group-disjoint Kev corpus without model loading.

    uv run python scripts/prepare_data.py --out-dir data/kev-v1

The supplied Kev test partition stays test-only. Development and calibration are
new partitions drawn from TRAIN groups, not the absent upstream partitions.
Existing outputs are never overwritten. No downloads or API calls are made.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from jev.data import (
    FORMAT_VERSION, MATCHING_POLICY, kev_examples, partition_summary, sha256_file,
    split_examples, verify_kev_file,
)
from jev.rendering import STRUCTURED_V1
from jev.serialization import dumps, loads


def prepare(args) -> dict:
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError(f"output already exists: {out}; choose a new directory")
    manifest_hash = sha256_file(args.manifest)
    manifest = loads(Path(args.manifest).read_text())
    if sha256_file(args.manifest) != manifest_hash:
        raise ValueError("input manifest changed while loading")
    train_rows, train_hash = verify_kev_file(args.train, manifest, "train")
    test_rows, test_hash = verify_kev_file(args.test, manifest, "test")
    train, train_skips = kev_examples(train_rows, train_hash, max_options=args.max_options)
    test, test_skips = kev_examples(test_rows, test_hash, max_options=args.max_options)
    partitions, removed = split_examples(
        train, test, seed=args.seed, development_fraction=args.development_fraction,
        calibration_fraction=args.calibration_fraction,
    )
    report = {
        "format_version": FORMAT_VERSION,
        "recommended_renderer": STRUCTURED_V1,
        "matching_policy": MATCHING_POLICY,
        "split_policy": "reserve supplied test; derive development/calibration from training connected groups",
        "seed": args.seed,
        "development_fraction": args.development_fraction,
        "calibration_fraction": args.calibration_fraction,
        "max_options": args.max_options,
        "excluded_sources": ["sst5"],
        "inputs": {
            "train": {"path": str(args.train), "sha256": train_hash},
            "test": {"path": str(args.test), "sha256": test_hash},
            "manifest": {"path": str(args.manifest), "sha256": manifest_hash},
        },
        "upstream_manifest": manifest,
        "skipped_questions": {"train": train_skips, "test": test_skips},
        "test_overlap_removed_training_ids": removed,
        "files": {},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".jev-prepare-", dir=out.parent))
    try:
        for split, examples in partitions.items():
            path = staging / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for example in examples:
                    handle.write(dumps(example.to_dict()) + "\n")
            report["files"][path.name] = {
                "sha256": sha256_file(path), **partition_summary(examples),
            }
        (staging / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        # A racing creator must not be replaced, even if its directory is empty.
        out.mkdir()
        try:
            for path in staging.iterdir():
                path.rename(out / path.name)
        except BaseException:
            shutil.rmtree(out)
            raise
    finally:
        shutil.rmtree(staging)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/external/kev_pp4_train.jsonl")
    parser.add_argument("--test", default="data/external/kev_pp4_test.jsonl")
    parser.add_argument("--manifest", default="data/external/kev_pp4_manifest.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--development-fraction", type=float, default=0.1)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--max-options", type=int, default=26,
                        help="readout admission limit; canonical schema supports up to 255")
    args = parser.parse_args()
    try:
        report = prepare(args)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"prepare_data: {exc}\n")
    print(json.dumps({
        "output": args.out_dir, "files": report["files"],
        "skipped_questions": report["skipped_questions"],
        "test_overlap_removed": len(report["test_overlap_removed_training_ids"]),
    }, indent=2))


if __name__ == "__main__":
    main()
