"""LoRA fine-tuning with the readout-matched loss.

The loss is cross-entropy between the target distribution and the softmax over
the *label tokens only* at the prompt's last position, matching the raw inference
readout rather than contextual correction. Cross-entropy is a proper scoring rule;
finite-data and out-of-domain calibration still require measurement.
mlx_lm's built-in text-completion loss doesn't fit; this loop replaces it.

    uv run python scripts/build_data.py --per-task 2000 --val-per-task 200
    uv run python scripts/train_lora.py --model HuggingFaceTB/SmolLM2-135M \
        --out adapters/smollm2-135m
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from jev.data import assert_training_disjoint, load_training_rows, sha256_file
from jev.rendering import RENDERER_VERSIONS, label_token_ids
from jev.provenance import environment_identity, model_identity

LORA_PARAMS = {
    "rank": 16,
    "scale": 20.0,
    "dropout": 0.0,
    "keys": [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ],
}


def tokenize_rows(source_rows: list[dict], tokenizer, max_seq: int) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    for source in source_rows:
        row = dict(source)
        row["label_ids"] = label_token_ids(tokenizer, row["labels"])
        tokens = tokenizer.encode(row["prompt"])
        if not tokens:
            raise ValueError(f"empty tokenized prompt: {row['id']}")
        if len(tokens) > max_seq:
            skipped += 1
            continue
        row["tokens"] = tokens
        rows.append(row)
    if not rows:
        raise ValueError("no examples remain after max-seq filtering")
    return rows, skipped


def make_batches(rows: list[dict], batch_size: int, pad_id: int, rng: random.Random):
    """Length-sorted batches (less padding), shuffled batch order."""
    rows = sorted(rows, key=lambda r: len(r["tokens"]))
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    rng.shuffle(batches)
    max_k = max(len(r["target"]) for r in rows)
    for batch in batches:
        max_len = max(len(r["tokens"]) for r in batch)
        inputs, last_idx, label_ids, label_mask, targets = [], [], [], [], []
        for r in batch:
            t = r["tokens"]
            inputs.append(t + [pad_id] * (max_len - len(t)))
            last_idx.append(len(t) - 1)
            k = len(r["label_ids"])
            label_ids.append(r["label_ids"] + [0] * (max_k - k))
            label_mask.append([1.0] * k + [0.0] * (max_k - k))
            targets.append(r["target"] + [0.0] * (max_k - k))
        yield (
            mx.array(inputs),
            mx.array(last_idx),
            mx.array(label_ids),
            mx.array(label_mask),
            mx.array(targets),
        )


def loss_fn(model, inputs, last_idx, label_ids, label_mask, targets):
    logits = model(inputs)  # (B, L, V)
    b, seq_len, vocab = logits.shape
    at_last = logits.reshape(b * seq_len, vocab)[last_idx + mx.arange(b) * seq_len]
    lab = mx.take_along_axis(at_last, label_ids, axis=1).astype(mx.float32)
    lab = lab + (1.0 - label_mask) * -1e9
    logp = lab - mx.logsumexp(lab, axis=1, keepdims=True)
    return -(targets * logp).sum() / b


def evaluate(model, rows, batch_size, pad_id) -> float:
    rng = random.Random(0)
    total, count = 0.0, 0
    for batch in make_batches(rows, batch_size, pad_id, rng):
        total += loss_fn(model, *batch).item() * batch[0].shape[0]
        count += batch[0].shape[0]
    return total / max(count, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", required=True, help="adapter output directory")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = no cap")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq", type=int, default=768)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--renderer", choices=RENDERER_VERSIONS, default=None,
                    help="inferred from canonical (v1) or legacy (v0) data if omitted")
    args = ap.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.max_seq < 1 or args.max_steps < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        ap.error("batch-size, epochs, max-seq, and lr must be positive; max-steps must be nonnegative")
    out = Path(args.out)
    if out.exists():
        ap.error(f"adapter output already exists: {out}; choose a new directory")

    # Validate supervision and splits before allocating a model or training it.
    input_hashes = {"train": sha256_file(args.train), "val": sha256_file(args.val)}
    source_train = load_training_rows(args.train, args.renderer)
    source_val = load_training_rows(args.val, args.renderer)
    if input_hashes != {"train": sha256_file(args.train), "val": sha256_file(args.val)}:
        ap.error("input data changed while loading")
    versions = {r["renderer_version"] for r in source_train + source_val}
    if len(versions) != 1:
        ap.error("training and validation must use the same renderer")
    renderer_version = versions.pop()
    assert_training_disjoint(source_train, source_val)
    print(f"renderer: {renderer_version}")

    mx.random.seed(args.seed)
    model, tokenizer = load(args.model)
    provenance = {"environment": environment_identity(), "backbone": model_identity(args.model)}
    pad_id = tokenizer.eos_token_id or 0

    model.freeze()
    num_layers = len(model.layers)
    linear_to_lora_layers(model, num_layers, LORA_PARAMS)
    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"LoRA on {num_layers} layers, {n_trainable / 1e6:.2f}M trainable params")

    train_rows, train_skipped = tokenize_rows(source_train, tokenizer, args.max_seq)
    val_rows, val_skipped = tokenize_rows(source_val, tokenizer, args.max_seq)
    print(f"{len(train_rows)} train rows, {len(val_rows)} val rows; "
          f"skipped {train_skipped}/{val_skipped} over {args.max_seq} tokens")

    optimizer = optim.Adam(learning_rate=args.lr)
    step_fn = nn.value_and_grad(model, loss_fn)
    rng = random.Random(args.seed)

    initial_val_loss = evaluate(model, val_rows, args.batch_size, pad_id)
    print(f"initial val loss: {initial_val_loss:.4f}")

    step, ema, t0, examples_seen = 0, None, time.time(), 0
    for epoch in range(args.epochs):
        for batch in make_batches(train_rows, args.batch_size, pad_id, rng):
            loss, grads = step_fn(model, *batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            step += 1
            examples_seen += batch[0].shape[0]
            val = loss.item()
            ema = val if ema is None else 0.95 * ema + 0.05 * val
            if step % 25 == 0:
                rate = step / (time.time() - t0)
                print(f"step {step}  loss(ema) {ema:.4f}  {rate:.2f} it/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    training_seconds = time.time() - t0
    final_val_loss = evaluate(model, val_rows, args.batch_size, pad_id)
    if not math.isfinite(final_val_loss):
        raise ValueError("nonfinite final validation loss; refusing to save unusable adapter")
    print(f"final val loss: {final_val_loss:.4f}")

    out.mkdir(parents=True, exist_ok=False)
    mx.save_safetensors(
        str(out / "adapters.safetensors"),
        dict(tree_flatten(model.trainable_parameters())),
    )
    (out / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": num_layers,
                "lora_parameters": LORA_PARAMS,
                "model": args.model,
                "renderer_version": renderer_version,
            }
        )
    )
    (out / "training_manifest.json").write_text(json.dumps({
        "arguments": vars(args), "renderer_version": renderer_version,
        "train_sha256": input_hashes["train"], "val_sha256": input_hashes["val"],
        "train_rows": len(train_rows), "val_rows": len(val_rows),
        "train_skipped": train_skipped, "val_skipped": val_skipped,
        "steps": step, "examples_seen": examples_seen,
        "initial_val_loss": initial_val_loss, "final_val_loss": final_val_loss,
        "training_seconds": training_seconds, "peak_active_bytes": mx.get_peak_memory(),
        **provenance,
    }, indent=2) + "\n")
    print(f"saved adapters -> {out}")


if __name__ == "__main__":
    main()
