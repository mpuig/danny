# jev — a local System One model and runtime

An open project inspired by [TypeSafe's Jev](https://docs.typesafe.ai/introduction).
The goal is a **small fine-tuned model**, built on an existing backbone such as
SmolLM or Qwen, with task-specific architectural changes and efficient local
inference. **Apple MLX is the primary training and serving backend**; a Rust
runtime is an option if profiling justifies it.

The model should answer narrow, request-defined questions about structured state
using **Choice / Score / Noul** probabilities, without generating text. Code owns
workflow logic; the model supplies semantic judgments and useful uncertainty.

## Current status

The repository contains a **Python/Apple MLX research prototype**, not a
production-ready decision service. It implements restricted-token readout, LoRA training, Jev
soft-target collection, evaluation scripts, and a development HTTP server.

- SmolLM3-3B is the current research backbone; SmolLM2-135M is a smoke-test model.
- The API supports the basic Jev request/answer shapes and has a TypeScript SDK
  smoke test. **Full behavioral and API compatibility is not established.**
- Calibration and unfamiliar-rubric generalization are research goals, not guarantees.
- Primitive-specific modeling and a bounded, tested MLX service are planned.
  Portable export and a Rust runtime are later, evidence-gated options.

See [Architecture](docs/ARCHITECTURE.md) for current behavior and compatibility gaps,
and [Roadmap](docs/ROADMAP.md) for the model and serving plan.

## Quickstart: current MLX prototype

Requires an Apple Silicon environment supported by MLX. Models download from
Hugging Face on first use.

```bash
uv sync

# One request with all three primitives; no adapter required.
uv run python scripts/demo_request.py --model HuggingFaceTB/SmolLM2-135M

# Untuned baseline. --calibrate enables optional contextual bias correction.
uv run python scripts/eval_baseline.py --model mlx-community/SmolLM3-3B-Base-bf16 \
    --task sst2 --n 200 --calibrate

# Development server; keep running and use a second terminal for the SDK test.
uv run python scripts/serve.py --model HuggingFaceTB/SmolLM2-135M --port 8399
```

```bash
# Requires Node with TypeScript type-stripping support.
cd tests/ts && npm install && node test.ts
```

Add `--adapter adapters/smollm3-3b` to the 3B evaluation/server commands only after
training or obtaining that matching adapter. Weights and datasets are not bundled
in git. See [Training](docs/TRAINING.md).

## Evidence so far

Historical 3B experiments report SST-2 accuracy of **0.925** with gold-label LoRA
and contextual correction (n=200). SST-2 shares its question and criteria with
the IMDB training task: this is **cross-dataset transfer**, not unseen-question
validation. A distilled adapter reports tweet-emotion ECE **0.065**, versus
**0.134** for gold-label LoRA, but the runs differ in data and lack uncertainty
intervals. See [Evaluation](docs/EVALUATION.md) for all results and limitations.

The local data audit found valid Kev train/test downloads, validation overlap
across training sources, and two Nimble files containing only `404: Not Found`.
Do not combine the datasets before resolving the issues in [Data](docs/DATA.md).

## Documentation

- [Overview](docs/OVERVIEW.md) — goals, scope, evidence, and sources
- [Architecture](docs/ARCHITECTURE.md) — current implementation and target MLX service
- [Data](docs/DATA.md) — external inventory, integrity checks, and split risks
- [Training](docs/TRAINING.md) — executable v0 recipe and its limitations
- [Evaluation](docs/EVALUATION.md) — metrics, commands, historical results, and required tests
- [Roadmap](docs/ROADMAP.md) — staged model and runtime implementation plan
- [Decisions](docs/DECISIONS.md) — experiment history and revised decisions

`TYPESAFE_API_KEY` is needed only for live Jev calls: distillation, uncached
agreement evaluation, and SDK tests with `--with-jev`. `.env`, `data/`, and
`adapters/` are gitignored. Check upstream data/model licenses and TypeSafe's terms
before distributing data or distilled weights.
