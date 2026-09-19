# How to Train

## 0. Setup

```bash
uv sync                     # installs mlx-lm, datasets, numpy, tqdm
```
Models download from Hugging Face on first use. For Jev distillation, put
`TYPESAFE_API_KEY=...` in `.env` at the repo root (gitignored).

## 1. Build training data (gold labels, one-hot)

```bash
uv run python scripts/build_data.py --per-task 2000 --val-per-task 200
# -> data/train.jsonl (8000 rows), data/val.jsonl (800 rows)
```

Recasts the train splits of the four training tasks (ag_news, dbpedia, imdb, yelp_stars —
see `src/jev/recast.py`) into rows of `{prompt, labels, target, task}`. The prompt is
rendered with the *same templates the engine uses at inference*. Augmentation (decision
log #12): instruction phrasings cycle per example; choice/noul option order is shuffled
with the target remapped; ~20% of choice examples drop option descriptions. Score levels
are ordered and never shuffled. States are capped at 1500 chars.

## 2. Pull soft targets from Jev (optional, recommended)

```bash
uv run python scripts/distill_from_jev.py --per-task 500 --out data/distill_train.jsonl
# resumable after interruptions:
uv run python scripts/distill_from_jev.py --per-task 500 --out data/distill_train.jsonl --resume
```

Same JSONL format, but `target` is Jev's probability distribution (noul → `[1−p, p]`;
choice/score reordered to our option order). Jev's raw answer is kept under `"jev"` and
the gold label under `"gold_label"` for agreement analysis. Uses a seed distinct from
build_data so the sets differ. ~1.5 requests/s observed.

## 2b. External data: kev's frozen datasets (optional)

kev commits checksummed, Jev-shaped datasets in-repo (`evals/public-pool-v4` and
`decision-v1/v2` with train/calibration/dev/test splits). Convert with:

```bash
uv run python scripts/convert_kev.py data/external/kev_pp4_train.jsonl --out data/kev_train.jsonl
```

The converter **excludes sst5** (shares SST sentences with our held-out sst2 —
contamination) and skips >26-option questions (banking77; needs the v1 reserved-token
tier). Yield: 11,000 questions over 10 sources incl. BoolQ (reading comprehension) and
MNLI (reasoning) — dimensions our own recast set lacks. Nimble publishes only its
curation *pipeline* + hashes, not data; jeff has none.

## 3. Train LoRA

```bash
# gold one-hot:
uv run python scripts/train_lora.py --model mlx-community/SmolLM3-3B-Base-bf16 \
    --out adapters/smollm3-3b --batch-size 4 --max-steps 800 --lr 1e-5

# Jev soft targets:
uv run python scripts/train_lora.py --model mlx-community/SmolLM3-3B-Base-bf16 \
    --train data/distill_train.jsonl --val data/val.jsonl \
    --out adapters/smollm3-3b-distill --batch-size 4 --epochs 2 --lr 1e-5
```

The loss is CE between the target distribution and the softmax over the **label tokens
only** at each prompt's last position — identical to the inference readout, soft-target
capable (decision log #8). Batches are length-sorted (less padding), right-padded (safe
under causal attention; logits taken at each row's true last token), mixed option counts
handled with label masking.

**Hyperparameters that matter:**

| | 135M | 3B |
|---|---|---|
| learning rate | 1e-4 OK | **1e-5** (1e-4 is destructive — see decision #9) |
| batch size | 8 | 4 (36 GB machine) |
| throughput | ~2.7 it/s | ~0.6 it/s |
| LoRA | rank 16, scale 20, q/k/v/o projections, all layers | same |

Adapters save in mlx-lm's standard format (`adapters.safetensors` + `adapter_config.json`)
and load anywhere via `--adapter` / `SystemOneEngine(..., adapter_path=...)`.

**Operational cautions:** one 3B training run takes 25–45 min; don't overlap heavy GPU
jobs — an earlier run was killed by the harness under system memory pressure at step
800/1000, losing the unsaved adapter. Consider periodic checkpointing if runs get longer.

## 4. Reference val losses (sanity anchors)

- 135M, one-hot, lr 1e-4, 400 steps: val 1.99 → 1.03
- 3B, one-hot, lr 1e-5, 800 steps: val 0.856 → **0.349**
- 3B, lr 1e-4 (destructive): val 0.856 → 1.59

## 5. Serving a trained model

```bash
uv run python scripts/serve.py --model mlx-community/SmolLM3-3B-Base-bf16 \
    --adapter adapters/smollm3-3b --calibrate --port 8399
```
Then point any Typesafe SDK at it: `TYPESAFE_BASE_URL=http://127.0.0.1:8399`.
