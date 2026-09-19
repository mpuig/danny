"""Prepare a small, matched-source readout pilot; this is not a matched-compute study."""

import argparse
import json
import random
from pathlib import Path

from mlx_lm.utils import load_tokenizer

from jev.data import load_examples, partition_summary, sha256_file
from jev.provenance import model_path
from jev.rendering import CANDIDATE_READOUT, LETTER_READOUT, render_views
from jev.serialization import dumps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="data/kev-v1/train.jsonl")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--per-primitive", type=int, default=150)
    ap.add_argument("--max-seq", type=int, default=768)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    if args.out_dir.exists() or min(args.per_primitive, args.max_seq) < 1:
        ap.error("fresh output and positive sizes required")
    tokenizer = load_tokenizer(model_path(args.model))
    examples = load_examples(args.train)
    rng = random.Random(args.seed)
    selected = []
    excluded = []
    for primitive in ("choice", "noul", "score"):
        candidates = [e for e in examples if e.question.type == primitive]
        rng.shuffle(candidates)
        kept = []
        for e in candidates:
            longest = max(
                len(tokenizer.encode(prompt))
                for readout in (LETTER_READOUT, CANDIDATE_READOUT)
                for prompt, _ in render_views(e.state, e.question, readout=readout)
            )
            if longest <= args.max_seq:
                kept.append(e)
            else:
                excluded.append(e.id)
            if len(kept) == args.per_primitive:
                break
        if len(kept) != args.per_primitive:
            ap.error(f"not enough admitted {primitive} examples")
        selected.extend(kept)
    selected.sort(key=lambda e: e.id)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    path = args.out_dir / "train.jsonl"
    path.write_text("".join(dumps(e.to_dict()) + "\n" for e in selected))
    report = {
        "arguments": {**vars(args), "out_dir": str(args.out_dir)},
        "input_sha256": sha256_file(args.train),
        "output_sha256": sha256_file(path),
        "excluded_examined_ids": excluded,
        "selection": "seeded per-primitive sample from training, admitted under both prompt factorizations",
        **partition_summary(selected),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
