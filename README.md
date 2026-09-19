# jev (open-source System One model)

An open reimplementation of the ideas behind [Typesafe's Jev](https://docs.typesafe.ai/introduction):
a model that answers typed questions (**choice / score / noul**) about a text `state` with
**calibrated probability distributions** — no text generation, no parsing. Built on
**Apple MLX**. Drop-in compatible with the official Typesafe SDKs.

## Documentation

- [docs/OVERVIEW.md](docs/OVERVIEW.md) — what this is, goals, competitive landscape, status
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — logit readout, engine design, calibration, server, roadmap tiers
- [docs/DECISIONS.md](docs/DECISIONS.md) — decision log with the evidence behind each choice
- [docs/TRAINING.md](docs/TRAINING.md) — data building, Jev distillation, LoRA loop, hyperparameters
- [docs/EVALUATION.md](docs/EVALUATION.md) — metrics, protocol, commands, results tables

## Quickstart

```bash
uv sync

# one Jev-shaped request, all three primitives:
uv run python scripts/demo_request.py --model HuggingFaceTB/SmolLM2-135M

# calibration eval (accuracy, ECE, Brier, NLL) on a recast dataset:
uv run python scripts/eval_baseline.py --model mlx-community/SmolLM3-3B-Base-bf16 \
    --task sst2 --n 200 --calibrate --adapter adapters/smollm3-3b

# serve a Jev-compatible API and hit it with the official TypeScript SDK:
uv run python scripts/serve.py --model mlx-community/SmolLM3-3B-Base-bf16 \
    --adapter adapters/smollm3-3b --calibrate --port 8399
cd tests/ts && npm install && node test.ts
```

## Layout

```
src/jev/
  schema.py       Jev-compatible request/response types
  engine.py       logit-readout engine: shared-prefix KV caching, contextual calibration
  metrics.py      ECE, Brier, NLL (the success criteria)
  recast.py       6 public datasets recast to state/question/label (2 held out)
scripts/
  demo_request.py     one request, all three primitives
  eval_baseline.py    calibration eval on any task/checkpoint
  permutation_test.py option-order sensitivity
  build_data.py       training data + shuffling augmentation
  distill_from_jev.py Jev soft-target puller (resumable)
  train_lora.py       LoRA with readout-matched CE loss (soft-target capable)
  serve.py            Jev-compatible API server (POST /v1/systemone)
tests/ts/          official @typesafe-ai/sdk run against our server and the real API
docs/              project documentation
```

## Headline result so far

SmolLM3-3B + LoRA + contextual calibration, on tasks **never seen in training** (n=200):
sst2 **0.925 acc / 0.068 ECE** (untuned: 0.695 / 0.088); tweet_emotion ECE
**0.205 → 0.134** at equal accuracy — calibration transfers to unseen questions.

Requires `TYPESAFE_API_KEY` in `.env` only for the distillation puller and `--with-jev`
SDK tests. `.env`, `data/`, `adapters/` are gitignored.
