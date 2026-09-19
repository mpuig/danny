# Architecture and compatibility

This page separates **current code**, **Jev's documented behavior**, and **planned
changes**. Reviewed 2026-09-19. MLX remains the primary training and serving backend.

## Current model: restricted-token readout

`src/jev/engine.py` loads an MLX language model, optionally with a LoRA adapter.
For each question it renders a prompt, reads final-position logits for ` A` through
` Z`, and applies softmax over those labels only. Application code serializes the
answer. There is no autoregressive answer generation.

This is a decision-model approximation, not a recovered implementation of Jev.
A vocabulary projection restricted to label rows already acts as a classifier head;
a separate head is an experiment, not automatically an improvement.

```text
respond(request)
  └─ ask(state, questions)
       ├─ _render(state, question)
       ├─ _score_batch(prompts)
       ├─ _apply_calibration(question, probabilities)  # optional bias correction
       └─ _to_answer(question, probabilities)
```

### Primitive rendering

| Primitive | Current computation | Important limitation |
|---|---|---|
| Choice | Whole list of named/described options; letter-token softmax | 26-option cap; position and label-token bias |
| Noul | Converted to Choice with `A=no`, `B=yes`; return `p_yes` | Original primitive identity is absent from the model prompt |
| Score | Whole list of lettered level descriptions; categorical softmax | Levels interact in one prompt, unlike the separate-level behavior described by Jev |

Score uses zero-based level indices: `score = sum(i * p_i)`. This expectation is
not an exact numeric measurement or a separate regression prediction.

**Noul identity:** a Noul and corresponding no/yes Choice can produce identical
prompts here. Jev's [limitations page](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
explicitly says their probabilities need not match. Teacher targets that differ
by primitive cannot be learned from identical inputs. Preserve a type marker or
distinct template before extending cross-primitive distillation.

**Score semantics:** the [Score docs](https://docs.typesafe.ai/primitives/score)
say each level is evaluated separately, without its number or neighboring levels.
Our joint-level readout differs. Compare it with a per-level scorer; the docs do
not disclose Jev's exact scoring, normalization, or attention implementation.

### Structured input is not yet implemented deliberately

The schema annotates instructions as strings. Non-string instructions and
criterion values can pass through at runtime, but f-strings render them using
Python representations. The server turns object state into `key: value` lines;
other non-string state uses `str(state)`.

This conversion is ambiguous. These distinct states both become `a: x\nb: y`:

```python
{"a": "x\nb: y"}
{"a": "x", "b": "y"}
```

A shared, versioned serializer is needed across training, evaluation, and serving.
It must preserve strings, nested fields, arrays, nulls, and option order. This is
important for instructions referring to paths such as `ticket.messages[0].text`.
Serialization alone does not make state immune to prompt injection.

## Current inference execution

`_score_batch` tokenizes every complete prompt, finds their longest common token
prefix, and uses caching when that prefix has at least eight tokens.

1. Encode the common prefix once into a KV cache.
2. **Physically repeat** each layer's cached keys and values across batch rows.
3. Right-pad the suffixes and process them together.
4. Apply the output head only to each row's last real hidden state.

With causal attention, those readout positions do not see later padding. Each
row sees the prefix and its own suffix, not another question. This provides the
intended information boundary, but numerical parity needs regression tests with
an explicit tolerance; bit-identical outputs are not a general guarantee.

Limitations:

- Cache memory scales with question count because the prefix is copied.
- Complete prompts are still tokenized separately.
- Choice/Noul and Score have different text before the state, so mixed requests
  can lose state-prefix sharing and fall back to sequential full-prompt scoring.
- New questions require three extra content-free evaluations when correction is on.
- There is no request-size bound, microbatch policy, or bounded prior cache.
- The `AttributeError` fallback may reuse an already-mutated cache if failure
  occurs after the optimized forward begins. It needs a safe capability check or
  fresh cache rather than a broad retry around mutable state.

Historical timing: 50 questions in 0.29 s and 100 in 0.30 s on SmolLM2-135M under
concurrent training load. This is not a controlled 3B benchmark or evidence that
100 questions cost the same as one. Packed attention is a possible optimization;
it does not automatically remove tokenization or question-to-state attention cost.

## Probabilities, correction, and confidence

### Contextual correction (`--calibrate`)

For each question, average distributions on `"N/A"`, `""`, and `"none"`, then:

```text
adjusted_i = p_i / max(prior_i, 1e-9)
output_i = adjusted_i / sum(adjusted)
```

This is contextual label-bias correction, commonly called contextual calibration.
It can help or hurt outcome calibration and can remove meaningful prior information.
Priors are cached by question; new rubrics incur additional work.

Training CE matches the **raw** label readout, not the corrected distribution.
Evaluate raw and corrected predictions separately. Temperature scaling on a disjoint
calibration partition is planned, not implemented.

### Confidence

Current Choice and Score confidence is `1 - normalized_entropy(probabilities)`.
It measures concentration, not the probability that an answer is correct.

This differs from the published
[TypeSafe adapter at revision fb52b103](https://github.com/typesafe-ai/system-one-adapter-python/blob/fb52b1030b7fc1f4f1cf39910afa5da54f9835e3/src/system_one_adapter/_utils/confidence_metrics.py):

- Choice uses `(p_max - 1/K) / (1 - 1/K)`, with a one-option special case.
- Score uses distance from the modal level relative to a uniform reference.

The [confidence docs](https://docs.typesafe.ai/confidence) describe a derived
statistic but do not specify an exact versioned formula. Align with a pinned
reference and verify live responses, accounting for rounding, before claiming
semantic compatibility. Do not transfer thresholds across definitions or primitives.

## API compatibility: current subset

| Contract | Current repository |
|---|---|
| `POST /v1/systemone`, answers keyed by question ID | Implemented for basic requests; IDs stay out of prompts |
| String/object/array state; structured instructions and criteria | Accepted unevenly; serialization and annotations need correction |
| Choice up to 255 options | 1–26 supported |
| Score with 2–10 levels and weighted mean | Implemented, with different model-side level handling |
| Noul probability, no separate confidence | Response shape implemented |
| Choice/Score confidence semantics | Different from the published adapter |
| Response identifies the actual model version | Echoes request `model`, or uses the loaded backbone name |
| Validation failure uses HTTP 422 | Selected exceptions return HTTP 400; validation is incomplete |
| `GET /v1/models` | Not implemented |
| Context limits and usage | No enforced token budget; local execution counters are not Jev billing semantics |

Jev 1.13's [model page](https://docs.typesafe.ai/models) specifies 64k total request
tokens and 32k for state plus the longest question. These are Jev's limits, not
limits established for our model or hardware. Some introductory docs round the
budget differently; use version-specific model documentation and contract tests.

`usage.input_tokens` currently counts scoring work, including uncached correction
priors; `output_tokens` is always zero. Jev can report nonzero output tokens without
autoregressive generation. A compatibility policy for usage still needs definition.

## Serving: MLX first

`scripts/serve.py` is a single-threaded `HTTPServer` with HTTP/1.1. Earlier experiments
encountered an MLX stream error in handler threads and client issues with HTTP/1.0.
Those observations motivated the current configuration, not universal claims about
MLX threading or Node's HTTP support.

The target is a bounded MLX inference worker with an HTTP front end, backpressure,
validated requests, safe error handling, and measured concurrency. Keep GPU work
on a controlled execution context. Implement microbatching and cache policies
before promising production latency or exposing the service beyond localhost.

Small architectural changes are part of the model goal: primitive-aware inputs,
possible specialized readouts, and efficient shared-state execution. Select changes
using controlled quality, memory, and latency experiments.

A Rust front end or full runtime is optional. First determine whether the bottleneck
is Python/tokenization, scheduling, GPU kernels, or model capacity. Rust alone does
not reduce transformer FLOPs or fix duplicated KV tensors. A second backend also
requires tokenizer, rendering, weights, and numerical-parity tests. See
[Roadmap](ROADMAP.md) for sequencing and acceptance criteria.
