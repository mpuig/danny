"""Freeze identical examples admitted by every compared model's tokenizer.

No truncation, model weights, downloads, or teacher API calls. Models must be cached.
This consumes train/development only, never calibration/test. The tokenizer-length
intersection is deliberate selection; report exclusions rather than hiding them.
"""
import argparse
import json
from pathlib import Path

from mlx_lm.utils import load_tokenizer

from jev.data import assert_disjoint, load_examples, partition_summary, sha256_file
from jev.provenance import model_identity, model_path
from jev.rendering import label_token_ids, render
from jev.serialization import dumps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--train", default="data/kev-v1/train.jsonl")
    ap.add_argument("--development", default="data/kev-v1/development.jsonl")
    ap.add_argument("--max-seq", type=int, default=768)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    if out.exists() or args.max_seq < 1:
        ap.error("choose a fresh output directory and positive max-seq")
    tokenizers = {name: load_tokenizer(model_path(name)) for name in args.models}
    source = {name: load_examples(getattr(args, name)) for name in ("train", "development")}
    assert_disjoint(source)
    selected, exclusions = {}, {}
    for split, examples in source.items():
        selected[split], exclusions[split] = [], []
        for example in examples:
            prompt, labels = render(example.state, example.question)
            lengths = {}
            for name, tokenizer in tokenizers.items():
                label_token_ids(tokenizer, labels)
                lengths[name] = len(tokenizer.encode(prompt))
            if max(lengths.values()) <= args.max_seq:
                selected[split].append(example)
            else:
                exclusions[split].append({"id": example.id, "source": example.source, "tokens": lengths})
        if not selected[split]:
            ap.error(f"no admitted {split} examples")
    report = {"arguments": vars(args), "renderer_version": "structured-v1",
              "models": [model_identity(name) for name in args.models],
              "inputs": {s: sha256_file(getattr(args,s)) for s in source}, "exclusions": exclusions, "files": {}}
    out.mkdir(parents=True, exist_ok=False)
    for split, examples in selected.items():
        path = out / f"{split}.jsonl"
        path.write_text("".join(dumps(e.to_dict()) + "\n" for e in examples))
        report["files"][path.name] = {"sha256": sha256_file(path), **partition_summary(examples)}
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"files": report["files"], "excluded": {k:len(v) for k,v in exclusions.items()}}, indent=2))


if __name__ == "__main__":
    main()
