# Training the MLX prototype

Two rendering versions are supported: `legacy-v0` for historical prompts/adapters,
and `structured-v1` for canonical structured examples. Read [Data](DATA.md) before
mixing sources. Old adapters do not establish performance on the new format;
renderer mismatches are rejected rather than silently applying v1 to v0 weights.

## 1. Setup

```bash
uv sync
```

MLX is the primary backend for both training and serving on Apple Silicon. Models
download from Hugging Face on first use. Only live Jev collection requires
`TYPESAFE_API_KEY` in the environment or the repository's `.env` file.

## 2. Recommended: canonical Kev data and structured-v1

```bash
# Only if this output directory has not already been prepared:
uv run python scripts/prepare_data.py --out-dir data/kev-v1

uv run python scripts/train_lora.py --model mlx-community/SmolLM3-3B-Base-bf16 \
    --train data/kev-v1/train.jsonl --val data/kev-v1/development.jsonl \
    --out adapters/smollm3-3b-structured-v1 --batch-size 4 --max-steps 800 --lr 1e-5
```

The preparer verifies source checksums, preserves provenance/structured inputs,
reserves test groups, and derives separate development/calibration partitions.
It refuses existing output directories. The trainer renders canonical records
with `structured-v1` by default, checks train/development leakage before loading
weights, validates/normalizes targets, and rejects non-single-token or colliding
labels. `--renderer legacy-v0` can render canonical data for a controlled legacy
ablation; it does not alter the canonical files. Run that ablation through
`eval_dataset.py`: v0 direct calls retain Python representations, whereas the
historical HTTP server flattens object state. Only v1 guarantees a shared structured
serialization contract across training, direct inference, and HTTP.

The trainer records `renderer_version` in adapter metadata, and arguments, data
hashes, retained/skipped counts, optimizer steps, and example exposures in
`training_manifest.json`. Both Python shuffling and MLX initialization are seeded.
Existing adapter directories are refused rather than overwritten.

This is still joint letter-token readout, not new primitive-specific heads.
Option-order augmentation for canonical examples, a per-level Score model,
temperature fitting, and periodic checkpoints remain future work.

Use only `development.jsonl` for model selection. `calibration.jsonl` is reserved
for future calibration fitting, and `test.jsonl` for frozen final evaluation.
No full structured-v1 3B training run has been completed by the implementation
smoke tests; those tests train the 135M on four fixture examples for two steps.

## 3. Historical recast gold-label data

```bash
uv run python scripts/build_data.py --per-task 2000 --val-per-task 200
# Writes data/train.jsonl (8,000 rows) and data/val.jsonl (800 rows).
```

This command **overwrites** those files. Use `--out-dir data/recast-v0` to preserve
existing artifacts, then pass the resulting paths explicitly to the trainer.

The four source tasks are ag_news, dbpedia, imdb, and yelp_stars. The builder draws
from source training splits, cycles three phrasings per task, and partitions the
selected rows into train/validation. The row format is:

```json
{"prompt": "...", "labels": [" A", " B"], "target": [0.0, 1.0], "task": "imdb"}
```

Current augmentation:

- Choice and Noul option order is shuffled, with targets remapped.
- Roughly 20% of Choice examples drop descriptions.
- Score levels remain ordered.
- Validation rows are rendered with augmentation too, not only canonical wording.

States are truncated to **1,500 characters** while keeping the original labels.
That can remove decisive evidence. The trainer separately skips prompts longer
than `--max-seq` (default 768 tokens). Neither operation is an evidence-preserving
truncation policy; report exclusions and revise this for the next data pipeline.

The builder does not pin source revisions or preserve source IDs. A distinct seed
is not a global split guarantee when other datasets are added.

## 4. Historical Kev prompt conversion

The existing converter accepts the downloaded Kev request-shaped rows:

```bash
uv run python scripts/convert_kev.py data/external/kev_pp4_train.jsonl \
    --out data/kev_train.jsonl
uv run python scripts/convert_kev.py data/external/kev_pp4_test.jsonl \
    --out data/kev_test.jsonl
```

These overwrite converted outputs. With the audited files, they yield 11,000 and
1,048 question rows. SST-5 and choices above 26 options are skipped. The converted
rows are compatible with the trainer's input format, but lack source/group metadata,
option-shuffling augmentation, and deliberate structured-state serialization.

**Do not use `kev_test.jsonl` as training, validation, or temperature-fitting data.**
Do not combine Kev training with the current recast validation: four exact states
overlap. Kev calibration/development files are listed by its manifest but absent
locally. Both local Nimble files are invalid downloads, not usable JSONL.
See [Data](DATA.md) for the inventory and checks.

## 5. Collect Jev soft targets (legacy-format experiment)

```bash
uv run python scripts/distill_from_jev.py --per-task 500 --out data/distill_train.jsonl
# Resume only with the same seed, configuration, and dataset ordering:
uv run python scripts/distill_from_jev.py --per-task 500 --out data/distill_train.jsonl --resume
```

This calls a paid external API. Check terms before using resulting data or weights
beyond experimentation. Without `--resume`, it overwrites the output file.

The collector stores the teacher distribution as `target`, the original label as
`gold_label`, and the answer as `jev`. Noul targets become `[1-p_yes, p_yes]`;
Choice and Score targets are reordered to the local labels. It does not apply the
gold builder's option shuffling or description dropout.

Current limitations:

- `jev-latest` is hard-coded, and the actual response model version is not stored.
- Resume uses per-task row counts, not record IDs; configuration changes are unsafe.
- Targets are not validated/normalized. Four existing rows sum to 0.99.
- Six existing teacher-training states occur in `data/val.jsonl`.
- Source IDs and a disjoint teacher validation partition are missing.

The historical distilled adapter used `data/val.jsonl`, so that validation loss is
not strictly held out. The updated trainer now rejects that known overlap and
normalizes near-unit targets. Before another distillation run, prepare disjoint
partitions and migrate source provenance rather than bypassing the check.
The collector still writes legacy prompts, so it cannot currently produce v1
teacher-training data without an explicit migration.

For a causal comparison, use the **same examples, augmentation, step budget, and
splits** for gold, teacher-argmax, teacher-soft, and mixed-target runs. The existing
8,000-row gold corpus versus 2,000-row soft corpus does not isolate soft-target value.

## 6. Reproduce a recast-only v0 LoRA run

Example for the recast-only v0 baseline, using a fresh adapter directory:

```bash
uv run python scripts/train_lora.py --model mlx-community/SmolLM3-3B-Base-bf16 \
    --train data/train.jsonl --val data/val.jsonl \
    --out adapters/smollm3-3b-recast-v0 --batch-size 4 --max-steps 800 --lr 1e-5
```

The loss is cross-entropy over **label-token logits only**, at each prompt's last
position, against a one-hot or soft target. It matches the raw inference readout,
not the optional contextual correction. Proper scoring rules encourage honest
probabilities in expectation under suitable assumptions; they do not guarantee
calibration under finite data, teacher errors, limited capacity, or task shift.
This is supervised fine-tuning, not a claimed reproduction of private RLCD.

Batches are length-sorted, then shuffled by batch. Right padding is safe for the
causal readout. Masks support different option counts. Unlike the optimized batched
inference path, training currently computes the full sequence vocabulary logits.

### Historical settings, not universal optima

| Setting | 135M smoke model | 3B research model |
|---|---|---|
| Learning rate | 1e-4 | 1e-5 |
| Batch size | 8 | 4 on the 36 GB development machine |
| Reported throughput | ~2.7 iterations/s | ~0.6 iterations/s |
| LoRA | Rank 16, scale 20, q/k/v/o attention projections, all layers | Same |

One 3B run at 1e-4 worsened validation loss; that does not prove the rate is always
destructive. Recorded losses were 0.856 → 0.349 for 3B at 1e-5 / 800 steps, and
0.856 → 1.59 at 1e-4. A 135M run reported 1.99 → 1.03. These are historical notes,
not results rerun in the documentation audit.

`--max-steps` limits optimizer steps, not dataset size. With batch size 4, 800 steps
process at most 3,200 example presentations, even when the corpus contains 8,000
rows. Record actual exposure counts when comparing experiments.

### Operational and reproducibility limits

Adapters use mlx-lm's `adapters.safetensors` and `adapter_config.json` format.
The latter now includes the renderer version. Saving happens only at the end;
there is no optimizer checkpoint/resume support.
Do not overlap heavy training jobs on a memory-limited machine. Earlier 3B runs
were reported to take 25–45 minutes, with one interrupted before saving.

The trainer now seeds Python batch order and MLX initialization, checks label
encodings, validates targets/split overlap, and writes a training manifest.
It does not yet pin/hash the base weights and tokenizer revision, record all
software/hardware versions, or support periodic/resumable checkpoints. Numerical
determinism across devices and dependency versions is not guaranteed.

## 7. Serve an adapter

```bash
uv run python scripts/serve.py --model mlx-community/SmolLM3-3B-Base-bf16 \
    --adapter adapters/smollm3-3b-recast-v0 --port 8399
```

For a v1 run, replace the adapter path with `adapters/smollm3-3b-structured-v1`.
The engine selects the adapter's renderer automatically; older metadata without a
renderer version selects v0. Add `--calibrate` only as an explicitly evaluated
contextual-correction variant. The server is still a development HTTP implementation; see
[Architecture](ARCHITECTURE.md) for its compatibility and serving limits.
