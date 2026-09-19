# Bounded local MLX serving

`scripts/serve.py` now separates HTTP handling from a single inference worker. The
worker loads the model and owns every MLX operation; HTTP threads only parse,
queue, and serialize requests. Independent execution remains the numerical default.

```bash
uv run python scripts/serve.py --model HuggingFaceTB/SmolLM2-135M \
  --adapter adapters/smollm2-135m-structured-v1 --port 8399
curl http://127.0.0.1:8399/v1/models
curl http://127.0.0.1:8399/metrics
```

Weights are local experiment artifacts, not bundled with this repository. Optional
`--temperature FILE` must match the model, readout, renderer, precision, execution
policy, microbatch size, and contextual-correction setting used to fit it. Older
artifacts from the fixed-size evaluator imply a microbatch of four.

## Default operating envelope

| Resource | Default |
|---|---:|
| Concurrent HTTP handlers | 16 |
| Waiting inference jobs | 8, plus one active job |
| JSON request body | 262,144 bytes |
| Header/body read deadline | 10 seconds total, not reset by trickling bytes |
| Queue + inference deadline | 30 seconds |
| Questions | 32 |
| State tokens | 2,048 |
| Tokens in any rendered prompt | min(4,096, backbone context limit) |
| Rendered tokens across one request | 1,000,000 |
| Model views across one request | 512 |
| Answer width | 26 for letters; up to 255 for candidate readout |
| Shared-execution microbatch | 4 views |
| Cached question priors / tokenized prompts | 128 / 64 entries |
| MLX free-buffer allocator cache reclamation threshold | 512 MiB, process-wide |

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

Worker tests cover ownership, queue saturation, cancellation, expiry, request
copying, recovery, read deadlines, and error status codes. Engine tests cover
admission before raw scoring, aggregate correction budgets, bounded/disabled caches,
and cooperative deadlines. The actual trained 135M adapter passes the official
TypeScript SDK smoke test through this worker architecture; this checks response
contracts, not the correctness of its judgments.

See [Experiments](EXPERIMENTS.md) for measured performance and model limitations.
