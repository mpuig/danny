# Architecture & Design

## Core idea: readout, not generation

Every Jev primitive is a distribution over a small, request-defined label set. A decoder
LLM computes exactly that at each position — the next-token logits. So:

1. Render the question as a completion-style prompt ending just before the answer.
2. One forward pass; take the logits at the final position.
3. Softmax restricted to the label tokens (` A`, ` B`, …) → the answer distribution.

No autoregression, no sampling, no parsing. Any answer is "answer-shaped" by construction,
and every question in a request gets an answer.

## Data flow (`src/jev/engine.py`)

```
respond(request)                        # Jev-shaped dict in/out (+ usage tokens)
  └─ ask(state, questions)
       ├─ _render(state, q)             # prompt + label list per question
       ├─ _score_batch(items)           # shared-prefix KV cache scoring
       ├─ _apply_calibration(q, probs)  # optional contextual calibration
       └─ _to_answer(q, probs)          # ChoiceAnswer / ScoreAnswer / NoulAnswer
```

### Prompt templates

- **choice**: header + `State:` + question + lettered options (`A. name: description`)
  + `The best option is` → read ` A`, ` B`, …
- **score**: same shape with lettered *level descriptions*; the response maps letters back
  to level numbers. `score` = probability-weighted mean over levels (hence fractional).
- **noul**: rendered **through the choice template** as `A. no / B. yes` (see
  DECISIONS.md #5 — bare yes/no completion has a severe acquiescence bias). Internal
  distribution order is `[p_no, p_yes]`; the answer is `probs[1]`.

### Parallel question scoring (shared state, one batched pass)

Multi-question requests share the state. `_score_batch` tokenizes all prompts, finds the
longest **token-level** common prefix (≥ 8 tokens to bother; token-level matching avoids
BPE boundary bugs), runs it once into a KV cache, tiles that cache across the batch
dimension, and scores **all question suffixes in a single batched forward pass** —
reading the LM head only at each row's readout position (materializing the full
`(n, len, vocab)` logits tensor was the dominant cost). Each row's context is exactly
prefix + its own suffix, so questions stay strictly independent (the property Jev
guarantees and jeff trades away) and answers are bit-identical to sequential scoring.

Measured (SmolLM2-135M, M-series, under concurrent training load): 50 questions 0.29 s,
**100 questions 0.30 s** — near-flat, matching the flat-latency behavior probed on the
real Jev. Residual slope at 200+ questions: Python tokenization and the KV-replication
tax; both disappear in the v2 packed design.

### Contextual calibration (Zhao et al. 2021)

`contextual_calibration=True` estimates the model's label prior per question as its mean
answer distribution over content-free states (`"N/A"`, `""`, `"none"`), then divides the
prior out and renormalizes. Priors are cached per question. Measured effect: ag_news
0.75 → 0.80 acc, Brier 0.40 → 0.31 with zero training.

### Confidence

`1 − normalized entropy` of the distribution (choice/score only, matching Jev's API).
Pure post-processing.

## Serving (`scripts/serve.py`)

`POST /v1/systemone`, same wire format as `api.typesafe.ai`, so official SDKs work via
`baseURL` / `TYPESAFE_BASE_URL` override. Implementation constraints discovered:

- **Single-threaded** `HTTPServer`: MLX ops must run on the thread that owns the stream
  (`ThreadingHTTPServer` handler threads crash with `There is no Stream(cpu, 0)`).
- **`protocol_version = "HTTP/1.1"`**: Node's fetch (undici) rejects HTTP/1.0 responses.
- **Object states**: the JS SDK sends `state` as an object (`{document: "..."}`); the
  server flattens it to `key: value` lines.

This is a dev server; production serving would want a worker-queue design.

## Schema (`src/jev/schema.py`)

Dataclasses mirroring the Jev API. `Question` validates type, choice ≤ 26 options (v0
letter-label limit), score 2–10 levels. Answers serialize with `asdict` into the Jev
response shape.

## Roadmap tiers

- **v0 (current)**: stock backbone + LoRA, letter-token readout. Limits: 26 options.
- **v1**: reserved option tokens (`<opt_0>…<opt_254>`) → 255 options (this is very likely
  the origin of Jev's own 255 cap).
- **v2**: trained scoring head over option spans + packed questions with block-causal
  masking (kev-0.5b is a working reference implementation; the archerhume probe indicates
  this is Jev's actual architecture: additive token counts, flat latency at 100+ questions,
  listwise option effects). Adopt only when the eval shows a wall — see DECISIONS.md #1.
