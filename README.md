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
production-ready decision service. It implements versioned readouts, LoRA training,
matched experiments, fitted temperatures, and a bounded local HTTP service.

- Actual structured-v1 SmolLM2-135M and Qwen3-0.6B training runs are complete;
  historical SmolLM3-3B results remain separately labeled.
- The API supports the basic Jev request/answer shapes and has a TypeScript SDK
  smoke test. **Full behavioral and API compatibility is not established.**
- Calibration and unfamiliar-rubric generalization are research goals, not guarantees.
- `structured-v1` rendering preserves JSON and primitive identity. Historical
  adapters automatically use `legacy-v0`; incompatible overrides are rejected.
- Canonical data, split/provenance checks, grouped comparisons, independent Score
  levels, and up-to-255-option candidate readout are implemented and tested.
- The local service has bounded admission/caches, deadlines, model discovery, and
  a model-owning worker. See [Serving](docs/SERVING.md) for limits and caveats.
- Portable export and a Rust runtime are later, evidence-gated options.

See [Architecture](docs/ARCHITECTURE.md) for current behavior and compatibility gaps,
and [Roadmap](docs/ROADMAP.md) for the model and serving plan.

## Quickstart: current MLX prototype

Requires an Apple Silicon environment supported by MLX. Models download from
Hugging Face on first use.

```bash
uv sync

# One request with all three primitives; no adapter required.
uv run python scripts/demo_request.py --model HuggingFaceTB/SmolLM2-135M

# Historical v0 baseline. New bare-model calls otherwise default to structured-v1.
uv run python scripts/eval_baseline.py --model mlx-community/SmolLM3-3B-Base-bf16 \
    --task sst2 --n 200 --calibrate --renderer legacy-v0

# Development server; keep running and use a second terminal for the SDK test.
uv run python scripts/serve.py --model HuggingFaceTB/SmolLM2-135M --port 8399
```

```bash
# Requires Node with TypeScript type-stripping support.
cd tests/ts && npm install && node test.ts
```

Add `--adapter adapters/smollm3-3b` to the 3B evaluation/server commands only after
training or obtaining that matching adapter. Weights and datasets are not bundled
in git. See [Training](docs/TRAINING.md). For the trained SmolLM and selected Qwen
adapters, fish-compatible commands, side-by-side servers, and request/output examples,
use [Serving](docs/SERVING.md).

## Start a structured-v1 experiment

```bash
# Validates hashes and derives group-disjoint train/development/calibration/test.
# Choose a fresh output directory; existing datasets are never overwritten.
uv run python scripts/prepare_data.py --out-dir data/kev-v1

# A new external-data evaluator saves per-example predictions and a run report.
uv run python scripts/eval_dataset.py --model HuggingFaceTB/SmolLM2-135M \
    --data data/kev-v1/development.jsonl --out-dir data/evals/v1-smoke --n 20

# Fast tests; GPU/model tests are opt-in (see Evaluation).
uv run python -m unittest discover -s tests/python -v
```

The audited local preparation produced 8,769 training, 1,128 development, 1,103
calibration, and 1,048 test questions. Do not mix these with the old corpora without
cross-source leakage checks. See [Training](docs/TRAINING.md) for the v1 LoRA command.

## Evidence so far

On the same 1,128-question development set, structured-v1 SmolLM2-135M reached
**67.0% accuracy**; Qwen3-0.6B at LR 1e-5 reached **81.3%** (NLL **0.460**, or
**0.448** with calibration-partition temperature fitting). Qwen scored **76.2%** on
42 synthetic rubric cases, but those labels need independent review. Candidate
wide-Choice predictions were near chance; calibration did not reliably transfer to
new rubrics. See [Experiments](docs/EXPERIMENTS.md) for controls and limitations.

Historical 3B experiments report SST-2 accuracy of **0.925** with gold-label LoRA
and contextual correction (n=200). SST-2 shares its question and criteria with
the IMDB training task: this is **cross-dataset transfer**, not unseen-question
validation. A distilled adapter reports tweet-emotion ECE **0.065**, versus
**0.134** for gold-label LoRA, but the runs differ in data and lack uncertainty
intervals. See [Evaluation](docs/EVALUATION.md) for all results and limitations.

The local data audit found valid Kev train/test downloads, validation overlap
across training sources, and two Nimble files containing only `404: Not Found`.
The new preparer uses only the verified Kev files; it does not mix in those invalid
or overlapping legacy artifacts. See [Data](docs/DATA.md).

## Documentation

- [Overview](docs/OVERVIEW.md) — goals, scope, evidence, and sources
- [Architecture](docs/ARCHITECTURE.md) — current implementation and compatibility gaps
- [Serving](docs/SERVING.md) — SmolLM/Qwen launch commands, requests, options, and bounded deployment
- [Experiments](docs/EXPERIMENTS.md) — controlled model/readout/calibration experiments
- [Data](docs/DATA.md) — external inventory, integrity checks, and split risks
- [Training](docs/TRAINING.md) — v1 experiments, calibration, and historical v0 recipes
- [Evaluation](docs/EVALUATION.md) — metrics, commands, historical results, and required tests
- [Roadmap](docs/ROADMAP.md) — staged model and runtime implementation plan
- [Decisions](docs/DECISIONS.md) — experiment history and revised decisions

`TYPESAFE_API_KEY` is needed only for live Jev calls: distillation, uncached
agreement evaluation, and SDK tests with `--with-jev`. `.env`, `data/`, and
`adapters/` are gitignored. Check upstream data/model licenses and TypeSafe's terms
before distributing data or distilled weights.
