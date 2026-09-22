# danny — a local System One model and runtime

> System 1 is "a machine for jumping to conclusions." — Daniel Kahneman.
> danny is that machine, taught to jump carefully.

An open learning project inspired by [TypeSafe's Jev](https://docs.typesafe.ai/introduction).
The goal is a **small fine-tuned model**, built on an existing backbone such as
SmolLM, Qwen, or MiniCPM, with task-specific changes and efficient local
inference. **Apple MLX is the primary training and serving backend**; a Rust
runtime is an option if profiling justifies it.

## Why "danny"

Jev takes its name from the [Jevons paradox](https://en.wikipedia.org/wiki/Jevons_paradox):
make a resource cheap enough and consumption explodes. Jev's product category is
the "System One" model — Daniel Kahneman's name, from
[*Thinking, Fast and Slow*](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow),
for the fast, automatic, intuitive judgment system. Colleagues called Kahneman
**Danny**. This project is a System 1 built in the open — fine-tuned from open
models, calibrated on purpose, honest about what it knows — so it borrows the
name of the person who described the faculty it tries to be. The naming also
follows the clone ecosystem's convention of casual human names (jeff, kev).
The internal Python package keeps the name `jev` for wire-compatibility clarity:
the served API is Jev's request/answer shape.

The model should answer narrow, request-defined questions about structured state
using **Choice / Score / Noul** probabilities, without generating text. Code owns
workflow logic; the model supplies semantic judgments and useful uncertainty.

## Quickstart

Requires **Apple Silicon** (MLX) and [uv](https://docs.astral.sh/uv/).

**Models on HuggingFace:**
[mpuig/danny-qwen3-0.6b](https://huggingface.co/mpuig/danny-qwen3-0.6b)
(volume tier — the adapter also ships in this repo) ·
[mpuig/danny-minicpm5-2b-q8](https://huggingface.co/mpuig/danny-minicpm5-2b-q8)
(quality tier, 2.5 GB). Each carries its identity-bound `temperature.json`.

```bash
git clone https://github.com/mpuig/danny && cd danny
uv sync

# volume tier (0.6B). The adapter and temperature file ship in this repo;
# the Qwen base model (~1.2 GB) downloads from HuggingFace on first run.
uv run python scripts/serve.py --model Qwen/Qwen3-0.6B \
  --adapter adapters/qwen3-0.6b-structured-v1-synthfiltered-rps \
  --temperature release/temperature-qwen3-0.6b.json --port 8399

# ask three typed questions in one request:
curl -s -X POST http://127.0.0.1:8399/v1/systemone \
  -H "Content-Type: application/json" -d '{
  "state": "I was charged twice for my flight and nobody answers the phone.",
  "questions": {
    "refund":      {"type": "noul",   "instructions": "Does the customer want money back?"},
    "route":       {"type": "choice", "instructions": "Which queue should handle this?",
                    "criteria": {"billing": "Payment issues.", "support": "Everything else."}},
    "frustration": {"type": "score",  "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm.", "Annoyed.", "Angry."]}}}'
```

Every question gets a typed answer with probabilities — nothing is generated
or parsed:

```json
{"model": "Qwen/Qwen3-0.6B@…",
 "answers": {
   "refund":      {"noul": 0.93},
   "route":       {"choice": "billing", "probabilities": {"billing": 0.91, "support": 0.09}, "confidence": 0.82},
   "frustration": {"score": 1.7, "probabilities": {"0": 0.05, "1": 0.21, "2": 0.74}, "…": "…"}},
 "usage": {"input_tokens": 385, "output_tokens": 0}}
```

The **[quality tier](https://huggingface.co/mpuig/danny-minicpm5-2b-q8)**
(MiniCPM5-2B fused to 8-bit — +6.8 accuracy points over the 0.6B on the fresh
reserved test, CI [+4.5, +9.2]) is a 2.5 GB download:

```bash
hf download mpuig/danny-minicpm5-2b-q8 --local-dir models/danny-minicpm5-2b-q8
uv run python scripts/serve.py --model models/danny-minicpm5-2b-q8 \
  --temperature release/temperature-minicpm5-2b-q8.json --port 8399
```

Details in `release/MODEL_CARD_minicpm5-2b-q8.md`. Per-workload calibration —
fitting temperatures to *your* traffic from ~100 labeled decisions — is one
call: `POST /v1/calibrations` (see [docs/SERVING.md](docs/SERVING.md)).

**Not affiliated with or endorsed by TypeSafe.** "Jev-compatible" describes the
request/answer shape (the official TypeScript SDK runs against this server
unchanged in smoke tests), not certified behavioral equivalence. Training data
targets include outputs from a pinned Jev API version; this is disclosed in the
model cards. Code is MIT-licensed; base models keep their own licenses.

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

## Explore the research prototype

Beyond the served tiers, the repo is a full experimental workbench. Everything
below also requires Apple Silicon; models download from Hugging Face on first use.

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
training or obtaining that matching adapter. Datasets and full-weight models are
not bundled in git; the selected LoRA adapters (both tiers plus experiment
controls) are. See [Training](docs/TRAINING.md). For launch commands for every
local adapter, side-by-side servers, and request/output examples, use
[Serving](docs/SERVING.md).

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

The headline numbers, each from a reserved test the model never influenced and
spent exactly once: the frozen **0.6B volume tier reached 77.6% / ECE 0.049**
on the untouched in-family test (development: 82.0% / 0.020 — dev numbers steer
experiments and are optimistic by construction); the **MiniCPM-q8 quality tier
beat it by +6.8 accuracy points, CI [+4.5, +9.2]**, on a fresh out-of-family
gold test, with better NLL and Brier. Quantization to 8-bit was quality-free
(zero argmax flips, median 1.61x serving speedup). The most important negative
result is also measured: **in-family calibration does not survive distribution
shift** (confident errors 14-19% at t>=0.9 out-of-family vs ~2% in-family) —
which is why the runtime ships per-workload temperature fitting, whose
resampled evaluation cut that to 3-4%. An earlier 42-case rubric benchmark was
**retired after retraction** — it swung on seed alone. Full protocols,
confidence intervals, and every retraction: [Experiments](docs/EXPERIMENTS.md)
and [Decisions](docs/DECISIONS.md).

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

- [Example benchmark](BENCHMARK.md) — 24 local Qwen requests versus fresh Jev 1.13.0 and historical cached answers; a small diagnostic, not full parity evidence
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
agreement evaluation, and SDK tests with `--with-jev`. `.env`, datasets under
`data/`, and full-weight models are gitignored; the selected LoRA adapters,
corpus manifests, and release temperature artifacts are tracked. Check upstream
data/model licenses and TypeSafe's terms before distributing data or distilled
weights.
