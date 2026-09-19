"""LoRA fine-tuning with the readout-matched loss.

The loss is cross-entropy between the target distribution and the softmax over
the *label tokens only* at the prompt's last position — exactly what the engine
reads at inference. Cross-entropy is a proper scoring rule, so with honest
(eventually soft) targets this trains toward calibrated probabilities.
mlx_lm's built-in text-completion loss doesn't fit; this loop replaces it.

    uv run python scripts/build_data.py --per-task 2000 --val-per-task 200
    uv run python scripts/train_lora.py --model HuggingFaceTB/SmolLM2-135M \
        --out adapters/smollm2-135m
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

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


def load_rows(path: str, tokenizer, max_seq: int) -> list[dict]:
    rows = []
    skipped = 0
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            tokens = tokenizer.encode(row["prompt"])
            if len(tokens) > max_seq:
                skipped += 1
                continue
            row["tokens"] = tokens
            row["label_ids"] = [
                tokenizer.encode(l, add_special_tokens=False)[0] for l in row["labels"]
            ]
            rows.append(row)
    if skipped:
        print(f"{path}: skipped {skipped} rows over {max_seq} tokens")
    return rows


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
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    pad_id = tokenizer.eos_token_id or 0

    model.freeze()
    num_layers = len(model.layers)
    linear_to_lora_layers(model, num_layers, LORA_PARAMS)
    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"LoRA on {num_layers} layers, {n_trainable / 1e6:.2f}M trainable params")

    train_rows = load_rows(args.train, tokenizer, args.max_seq)
    val_rows = load_rows(args.val, tokenizer, args.max_seq)
    print(f"{len(train_rows)} train rows, {len(val_rows)} val rows")

    optimizer = optim.Adam(learning_rate=args.lr)
    step_fn = nn.value_and_grad(model, loss_fn)
    rng = random.Random(args.seed)

    print(f"initial val loss: {evaluate(model, val_rows, args.batch_size, pad_id):.4f}")

    step, ema, t0 = 0, None, time.time()
    for epoch in range(args.epochs):
        for batch in make_batches(train_rows, args.batch_size, pad_id, rng):
            loss, grads = step_fn(model, *batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            step += 1
            val = loss.item()
            ema = val if ema is None else 0.95 * ema + 0.05 * val
            if step % 25 == 0:
                rate = step / (time.time() - t0)
                print(f"step {step}  loss(ema) {ema:.4f}  {rate:.2f} it/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    print(f"final val loss: {evaluate(model, val_rows, args.batch_size, pad_id):.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
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
            }
        )
    )
    print(f"saved adapters -> {out}")


if __name__ == "__main__":
    main()
