# Bounded local MLX serving

`scripts/serve.py` now separates HTTP handling from a single inference worker. The
worker loads the model and owns every MLX operation; HTTP threads only parse,
queue, and serialize requests. Independent execution remains the numerical default.

## Run SmolLM and Qwen side by side

Run these commands from the repository root, in separate terminals. They work in
**fish** as well as Bash; no heredocs or shell-specific variable assignments are needed.

**Terminal 1 — SmolLM on port 8399:**

```fish
uv run python scripts/serve.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --adapter adapters/smollm2-135m-structured-v1 \
  --port 8399
```

**Terminal 2 — Qwen on port 8400:**

```fish
uv run python scripts/serve.py \
  --model Qwen/Qwen3-0.6B \
  --adapter adapters/qwen3-0.6b-structured-v1-synthfiltered-rps \
  --port 8400
```

Use the **`synthfiltered-rps` Qwen adapter** (selected and frozen, decision 25):
82.0% development accuracy, the family's best development NLL (0.428) and ECE
(0.020), with the held-out synthetic gains retained. Its fitted temperature file
is `data/runs/filtered-v2/temperature.json`. The `lr1e-5` kev-only run (81.3%)
remains the experiment control. These results do not
guarantee correct answers on your tickets; see
[Experiments](EXPERIMENTS.md) for the evaluation scope.

Wait for a startup JSON line containing `"ready": true`. Leave each server running;
use another terminal to send requests. Press **Ctrl+C** in a server terminal to stop
that process. If a port is already occupied, stop its server or choose another port.

Each process loads one model/adapter pair. Choose the **URL/port** to select a server;
a request's `model` field does not load or route to another model. Both servers can
accept `"model": "jev-latest"` as a compatibility alias, but neither runs Jev.
They return their actual model identity.

Both processes share the GPU and consume additional memory. Simultaneous work can
change latency; the isolated benchmarks do not guarantee the same speed under contention.

Weights and temperature files are local experiment artifacts, not bundled in git.
Base models download from Hugging Face when needed; adapters must already exist or
be trained using [Training](TRAINING.md). To require cached base-model files, prefix
a launch command with `env HF_HUB_OFFLINE=1`.

### Check both servers

```fish
curl -sS http://127.0.0.1:8399/v1/models | jq
curl -sS http://127.0.0.1:8400/v1/models | jq
curl -sS http://127.0.0.1:8400/health | jq
curl -sS http://127.0.0.1:8400/metrics | jq
```

## Send a request and format the response

This fish-compatible example sends all three question types to Qwen. Change `8400`
to `8399` to send the same request to SmolLM. No API key is required locally.

```fish
curl -sS http://127.0.0.1:8400/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "jev-latest",
    "state": {
      "ticket_message": "I was charged twice. Please refund the duplicate charge."
    },
    "questions": {
      "refund_requested": {
        "type": "noul",
        "instructions": "Does ticket_message explicitly request a refund?"
      },
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this ticket?",
        "criteria": {
          "billing": "Payments, charges, and refunds.",
          "technical": "Technical problems.",
          "sales": "Questions about purchasing."
        }
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Annoyed", "Angry"]
      }
    }
  }' | jq
```

For reusable payloads, save just the JSON object above as `request.json`, then run:

```fish
curl -sS http://127.0.0.1:8400/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @request.json | jq '.answers'
```

A JSON file also avoids shell-quoting problems when text contains apostrophes.
Fish does not support Bash's `<<'JSON'` heredoc syntax.

Replace the final formatter in either command with one of these:

```fish
# Indented JSON, including model identity and usage:
jq

# Only answers:
jq '.answers'

# Round numbers for display only; retain raw values for decisions/evaluation:
jq '.answers | walk(if type == "number" then (. * 1000 | round) / 1000 else . end)'

# Without jq:
uv run python -m json.tool
```

`noul` is the probability of yes. `choice` is the highest-probability option.
`score` is a weighted mean of zero-based levels, so this example uses a 0–2 scale.
`confidence` is a derived concentration statistic, not the probability of correctness;
a Score confidence of zero is valid. `output_tokens: 0` is expected for this runtime.
Matching Jev's core response shape does not establish equivalent judgments or full
API compatibility; see [Architecture](ARCHITECTURE.md).

### Official SDK example

[`tests/ts/test.ts`](../tests/ts/test.ts) sends a request and prints rounded summaries.
From the repository root, test either URL without changing the script:

```fish
cd tests/ts
npm install
env LOCAL_JEV_URL=http://127.0.0.1:8399 node test.ts
env LOCAL_JEV_URL=http://127.0.0.1:8400 node test.ts
cd ../..
```

Do not add `--with-jev` for local testing; that option makes live external API calls.
[`scripts/demo_request.py`](../scripts/demo_request.py) contains another payload,
but running it loads a model directly rather than calling these HTTP servers.

[Example benchmark](../BENCHMARK.md) compares 24 local Qwen requests with cached
answers and fresh Jev 1.13.0 responses. Its cached-reference runner makes no external
calls; a separate live-refresh command requires `--confirm-live` and incurs API usage.

## Other model and serving options

The commands below are **alternative launches**. Stop an existing process before
reusing its port, or choose an unused port.

### Available local adapter variants

| Variant | Matching backbone | Adapter path | Intended use |
|---|---|---|---|
| SmolLM v1 | `HuggingFaceTB/SmolLM2-135M` | `adapters/smollm2-135m-structured-v1` | Smaller trained baseline |
| Qwen v1, filtered synth+RPS | `Qwen/Qwen3-0.6B` | `adapters/qwen3-0.6b-structured-v1-synthfiltered-rps` | Selected configuration, frozen (decision 25) |
| Qwen v1, LR 1e-5 | `Qwen/Qwen3-0.6B` | `adapters/qwen3-0.6b-structured-v1-lr1e-5` | kev-only control |
| Qwen v1, LR 5e-5 | `Qwen/Qwen3-0.6B` | `adapters/qwen3-0.6b-structured-v1` | Regression/control run, not the recommended adapter |
| SmolLM candidate pilot | `HuggingFaceTB/SmolLM2-135M` | `adapters/smollm2-135m-readout-pilot-candidate-v1` | Experimental independent Score levels and wide Choice |
| SmolLM 3B gold | `mlx-community/SmolLM3-3B-Base-bf16` | `adapters/smollm3-3b` | Historical legacy-v0 model |
| SmolLM 3B distilled | `mlx-community/SmolLM3-3B-Base-bf16` | `adapters/smollm3-3b-distill` | Historical legacy-v0 distillation experiment |
| MiniCPM v1, filtered synth+RPS | `openbmb/MiniCPM5-2B-Base` | `adapters/minicpm5-2b-structured-v1-synthfiltered-rps` | Scale-up reference (experiments §11) |
| MiniCPM patch1 | `openbmb/MiniCPM5-2B-Base` | `adapters/minicpm5-2b-sfr-patch1` | Experimental record of the null patch pass (decision 26); not a replacement |

### Fused and quantized MiniCPM weights

`models/minicpm5-2b-sfr-fused` (BF16, 4.7 GB), `-q8` (2.5 GB), and `-q4` (1.3 GB)
are full-weight variants of the MiniCPM reference adapter, launched with `--model
models/minicpm5-2b-sfr-q8` and **no `--adapter`**. They are git-ignored artifacts;
rebuild with `mlx_lm fuse` on the reference adapter, then `mlx_lm convert -q
--q-bits 8|4`.

**q8 is the recommended quality-tier serving configuration** (decision 28):
quality-free versus BF16, median 1.61x faster, with its own fitted temperatures.

```fish
uv run python scripts/serve.py --model models/minicpm5-2b-sfr-q8 \
  --temperature data/runs/quant-v1/temperature-q8.json --port 8399
```

The temperature file is pinned to the fused-q8 weight hashes (sha256
0115533118f2ad2eb88e1a019dac1fb5bdc8179516c01b8e41c055331cff712c, reported by
`/v1/models`); it fails closed against any other weights, including BF16 and q4.
If the file is missing, rebuild it: eval_dataset on
`data/experiments/synthfiltered-corpus/calibration.jsonl` with the q8 model (raw),
then `fit_calibration.py` with that predictions dir and the corpus manifest. q4
has no temperature artifact and earns no tier (slower than q8, lower quality —
experiments §13).

The letter-readout pilot and four `smollm2-135m-teacher-matched-*` adapters are also
local experiment controls, not better validated replacements for the selected Qwen.
A larger backbone is not automatically a better configuration. No structured-v1
3B adapter has been trained in these experiments. Do not pair an adapter with a
different backbone merely by changing `--model`.

To try the historical 3B gold adapter:

```fish
uv run python scripts/serve.py \
  --model mlx-community/SmolLM3-3B-Base-bf16 \
  --adapter adapters/smollm3-3b \
  --port 8401
```

Use `adapters/smollm3-3b-distill` instead for its distilled counterpart. Legacy
adapters select legacy rendering automatically, including historical state flattening
and entropy confidence. Do not force `--renderer structured-v1` on them.

To run an **untuned baseline**, omit `--adapter`:

```fish
uv run python scripts/serve.py --model Qwen/Qwen3-0.6B --port 8402
```

### Candidate readout: experimental, not a quality upgrade

```fish
uv run python scripts/serve.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --adapter adapters/smollm2-135m-readout-pilot-candidate-v1 \
  --port 8401
```

This adapter selects `candidate-v1` from its metadata. Score descriptions are scored
independently, then normalized; Choice supports up to 255 options within token/view
budgets. Each candidate costs a binary evaluation. The pilot scored **0/30** on the
77-option holdout, with nearly uniform probabilities. Treat it as a research option.
You cannot switch a letter-trained adapter to candidate readout by adding a flag.

### Optional fitted temperatures

Use `--temperature` to load a fitted probability-scaling artifact:

```fish
uv run python scripts/serve.py \
  --model Qwen/Qwen3-0.6B \
  --adapter adapters/qwen3-0.6b-structured-v1-synthfiltered-rps \
  --temperature data/runs/followup-v1/qwen3-temperature.json \
  --port 8400
```

The corresponding SmolLM file is `data/runs/followup-v1/smollm2-temperature.json`,
for `adapters/smollm2-135m-structured-v1`. Both artifacts were fitted with native
precision, independent execution, microbatch size four, and **no `--calibrate`**.

Temperature scaling improved in-family negative log-likelihood, not every metric
or unfamiliar rubric. It does not change the most likely label, so it cannot repair
a wrong yes/no classification at a 0.5 threshold. The weighted-mean Score can change.
Artifacts must match weights/tokenizer, renderer, readout, precision, execution,
microbatch size, and contextual correction. Older fixed-size artifacts imply four views.

`--calibrate` is a **different operation**: it divides out content-free question
priors. It adds work for uncached rubrics, can change selected answers, and is not a
guaranteed accuracy/calibration improvement. The temperature files above cannot be
combined with it; a matching artifact would need to be fitted separately.

### Precision and shared-prefix execution

```fish
uv run python scripts/serve.py \
  --model Qwen/Qwen3-0.6B \
  --adapter adapters/qwen3-0.6b-structured-v1-synthfiltered-rps \
  --precision float32 --execution-mode shared \
  --port 8400
```

This is an experimental performance configuration, not the default recommendation.
FP32 shared execution reduced measured sibling-dependent drift, used more memory,
and improved latency on some fixtures but not others. Native and FP16 shared paths
showed larger drift. Do not attach the native/independent temperature artifact to
this launch. Validate quality and fit matching calibration before relying on it.

### Configuration reference

```fish
uv run python scripts/serve.py --help
```

| Flag | Choices / behavior |
|---|---|
| `--model`, `--adapter` | Base model and matching local adapter; omit adapter for an untuned baseline |
| `--renderer` | `structured-v1` or `legacy-v0`; normally selected from adapter metadata |
| `--readout` | `letters-v1` or `candidate-v1`; override must match the adapter |
| `--precision` | `native` (default), `float16`, `float32` |
| `--execution-mode` | `independent` (default), `shared` |
| `--temperature` | Path to a matching fitted temperature artifact |
| `--calibrate` | Optional content-free contextual correction, separate from temperature scaling |
| `--confidence` | `typesafe-fb52b10` (v1 default) or `entropy-v0` (legacy default); changes confidence, not probabilities |
| `--port` | Default 8399; use different ports for multiple processes; 0 selects a free port |
| `--host`, `--allow-remote` | Loopback by default; remote binding requires explicit consent and has no built-in authentication/TLS |
| `--ready-file` | Write startup metadata, including the selected port, to a fresh file |

## Default operating envelope

| Resource | Default | Flag |
|---|---:|---|
| Concurrent HTTP handlers | 16 | `--max-connections` |
| Waiting inference jobs | 8, plus one active job | `--queue-capacity` |
| JSON request body | 262,144 bytes | `--max-body-bytes` |
| Header/body read deadline | 10 seconds total, not reset by trickling bytes | `--io-timeout` |
| Queue + inference deadline | 30 seconds | `--request-timeout` |
| Questions | 32 | `--max-questions` |
| State tokens | 2,048 | `--max-state-tokens` |
| Tokens in any rendered prompt | min(4,096, backbone context limit) | `--max-prompt-tokens` |
| Rendered tokens across one request | 1,000,000 | `--max-request-tokens` |
| Model views across one request | 512 | `--max-views` |
| Answer width | 26 for letters; up to 255 for candidate readout | `--max-options` (cannot lift the letter cap) |
| Shared-execution microbatch | 4 views | `--max-batch-size` (not HTTP concurrency) |
| Cached question priors | 128 entries | `--max-prior-cache-entries` |
| Cached tokenized prompts | 64 entries | `--max-token-cache-entries` |
| MLX free-buffer allocator cache reclamation threshold | 512 MiB, process-wide | `--mlx-cache-limit-mb` |

View/token budgets include uncached content-free correction work. A cold corrected
request can therefore exceed a budget that a warm request fits. Each scoring stage
checks the remaining budget before its forwards; a late correction-stage rejection
can occur after raw scoring. Python caches use bounded LRU eviction; capacity zero
disables storage. KV caches are per scoring call, not persistent request histories.
Shared prefix KV is still physically copied, bounded by the microbatch and context
limits. MLX reclaims cache excess on the next allocation, so a telemetry snapshot
can exceed its configured threshold. This is not paged attention, a hard allocator
ceiling, or a total process-RAM reservation.

All ceilings are configurable through `--help`. Larger settings are not certified
safe merely because a backbone advertises a longer context. Account for model
weights, precision, working tensors, allocator cache, and other applications.

## Per-workload calibration

The out-of-family calibration failure is measured (experiments §14) and so is
the repair (§15): a scalar temperature fitted on ~100 labeled decisions from a
workload's own traffic. The server supports that fit directly.

**Fit:** `POST /v1/calibrations` with
`{"workload": "<name>", "examples": [{"state", "question", "label"}, ...]}`.
Labels are gold answers: the option key for choice, the level index for score,
true/false for noul. The server scores every example through the exact serving
path (raw plus contextual correction, before any temperature), fits per-primitive
temperatures, registers the result under the workload name, and returns the
artifact. Between 25 (`--min-calibration-examples`) and 256
(`--max-calibration-examples`) examples per request; ~100 per primitive is the
evidence-backed recommendation. **A fit runs on the single model thread and
blocks other requests for its duration** (`--fit-timeout`, default 120 s): run
fits off-peak.

**Use:** add `"calibration": "<name>"` to a `/v1/systemone` request. The
workload's temperatures replace the global ones for the primitives they cover;
uncovered primitives keep the global temperature. The response echoes
`{"calibration": {"workload", "sha256"}}`. An unknown name is a 422 — fail
closed, never silently uncalibrated.

**Diagnostics are the feature, not a footnote.** Each fit reports per primitive:
the temperature, NLL and top-1 ECE before/after (in-sample, therefore
optimistic), and a verdict — `apply` (|log T| is material), `neutral`
(temperature ~1, calibration already fine), `structural_warning` (temperature ~1
but residual ECE > 0.10: the miscalibration is not scalar; widen escalation
instead of trusting the fit), `degenerate` (bound hit or an all-correct
zero-NLL plateau; never applied), or `insufficient_examples` (never applied).

**Persistence:** the server stores fits in memory only (bounded,
`--max-workload-calibrations`, default 16). Save the returned artifact and
reload it at startup with `--workload-calibration FILE` (repeatable). Loading
verifies the artifact's integrity hash and its full prediction-config identity
against the running configuration — a fit from different weights, renderer,
precision, or execution mode is refused. `GET /v1/models` lists registered
workloads and their hashes.

Temperatures fitted on one workload's sample are valid for that workload only,
and in-sample diagnostics overstate held-out quality; the resampled evidence is
experiments §15. A temperature never changes the argmax answer.

## Protocol and failure policy

- `POST /v1/systemone`: typed answers, actual artifact/configuration identity, local
  scoring-token usage. `jev-latest` is accepted as a compatibility alias, not echoed
  as the loaded model identity. Unknown model IDs are rejected.
- `GET /v1/models`: backbone revision, adapter and implementation hashes, rendering,
  readout, confidence, precision, temperature identity, and effective limits.
- `GET /health`, `/metrics`: readiness, queue/counter snapshots, bounded-cache sizes,
  cumulative MLX peak active allocation and process peak RSS. These memory measures
  overlap; do not add them. Telemetry is sampled on the model thread after work.
- 422: JSON/schema/token/view validation. 413: body too large. 400/411/415: invalid
  HTTP framing or unsupported body encoding. 408: body read deadline. 503 with
  `Retry-After: 1`: queue saturation; connection saturation gets best-effort 503.
  504: inference/queue deadline. 500: generic internal failure, without traceback.
- Connections close after one response. Chunked/compressed bodies are unsupported.
  Body bytes, headers, prompts, and API keys are not logged by the handler.

Expired/cancelled queued jobs never execute. Running jobs check deadlines between
forwards. **An in-flight Metal kernel cannot be safely preempted by this thread
worker.** A timed-out caller may leave that one kernel finishing; its result is
discarded. This is not a hard real-time cancellation guarantee or a process supervisor.

SIGINT and SIGTERM initiate shutdown even when a background shell passed an ignored
SIGINT disposition. The benchmark verifies exit status and reports forced cleanup;
a successful HTTP run alone is not a passed lifecycle test.

The default binding is loopback. Non-loopback binding requires `--allow-remote`;
there is **no authentication, TLS, tenant isolation, or public-service certification**.
A reverse proxy alone does not establish model quality or safe automation policy.

## Verification

Worker tests cover ownership, connection/queue saturation, cancellation, expiry, request
copying, recovery, read deadlines, and error status codes. Engine tests cover
admission before raw scoring, aggregate correction budgets, bounded/disabled caches,
and cooperative deadlines. The actual trained 135M adapter passes the official
TypeScript SDK smoke test through this worker architecture; this checks response
contracts, not the correctness of its judgments.

Fresh real-model benchmark runs verified clean SIGTERM exits, and a lifecycle test
verifies inherited SIGINT handling. Selected Qwen also passed the SDK test with its
fitted temperature; 24 development predictions matched archived probabilities exactly.

The implementation closeout (`c7a1934`) passed all **67 Python tests** with model,
training, and candidate opt-ins enabled; Ruff, compile checks, the then-existing
20 CLI help commands, and Markdown link/fence/shell checks also passed. The later
cached-teacher benchmark and live refresh each add five unit tests, plus recorded
HTTP comparison results. These small diagnostics do not certify model parity.

See [Experiments](EXPERIMENTS.md) for measured performance and model limitations.
