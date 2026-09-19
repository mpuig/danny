# Architecture and compatibility

This page separates **current code**, **Jev's documented behavior**, and **planned
changes**. Reviewed 2026-09-19. MLX remains the primary training and serving backend.

## Current model: restricted-token readout

`src/jev/engine.py` loads an MLX language model, optionally with a LoRA adapter.
It uses the framework-independent `src/jev/rendering.py` to render prompts, reads
final-position logits for ` A` through ` Z`, and applies softmax over those labels
only. Application code serializes the answer. There is no answer-generation loop.

Bare models default to `structured-v1`. Adapters select their recorded
`renderer_version`; metadata-free historical adapters select `legacy-v0`.
`--renderer` permits an explicit choice but rejects mismatches with an adapter.
New training files can be canonical examples, rendered at load time. Legacy
pre-rendered files retain v0 prompts and cannot be silently upgraded.

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
| Noul | Binary letter readout `A=no`, `B=yes`; return `p_yes` | v1 retains a Noul type marker; legacy-v0 erases it |
| Score | Whole list of lettered level descriptions; categorical softmax | Levels interact in one prompt, unlike the separate-level behavior described by Jev |

Score uses zero-based level indices: `score = sum(i * p_i)`. This expectation is
not an exact numeric measurement or a separate regression prediction.

**Noul identity:** legacy-v0 can produce identical prompts for a Noul and a
corresponding no/yes Choice. Jev's
[limitations page](https://docs.typesafe.ai/model-jaggedness/jev-1.13) says their
probabilities need not match. `structured-v1` now preserves a type marker after
state, allowing different behavior to be learned. Existing v0 adapters have not
been retrained for this format.

**Score semantics:** the [Score docs](https://docs.typesafe.ai/primitives/score)
say each level is evaluated separately, without its number or neighboring levels.
Our joint-level readout differs. Compare it with a per-level scorer; the docs do
not disclose Jev's exact scoring, normalization, or attention implementation.

### Execution policy update

Independent scoring is now the default, preventing sibling questions from changing
the compute shape of a decision. Shared prefix caching requires
`--execution-mode shared`; `--precision float32` substantially reduces measured
drift. Native shared execution is experimental. Both paths project only the last
hidden position on supported backbones. See [Experiments](EXPERIMENTS.md) for
SmolLM/Qwen measurements; historical timing and projection results are not a parity
guarantee for the updated path.

### Versioned structured input

`src/jev/serialization.py` validates JSON, rejecting duplicate object keys,
nonfinite numbers, unsupported values, and excessive nesting. Schema entries
accept strings, objects, arrays, or null; state accepts strings, objects, or arrays.
The schema permits 255 Choice options; both current renderers explicitly limit
the letter readout to 26. Training and inference require distinct single-token labels.

In v1, state and every user-defined instruction/criterion are serialized as complete
JSON values, preserving nested fields, arrays, nulls, string boundaries, and option
order. The header and state prefix are identical across primitive types. Type-specific
material comes afterward, so mixed requests can share state computation.

Legacy rendering is retained for historical adapters, including the server's old
`key: value` flattening. That old path can collapse these states to the same text:

```python
{"a": "x\nb: y"}
{"a": "x", "b": "y"}
```

The v1 path keeps them distinct, with golden-prompt and HTTP regression tests.
This preserves input information; it is not a prompt-injection defense or proof
that an untuned model follows structured paths reliably.

## Current inference execution

`_score_batch` tokenizes every complete prompt, finds their longest common token
prefix, and uses caching when that prefix has at least eight tokens.

1. Encode the common prefix once into a KV cache.
2. **Physically repeat** each layer's cached keys and values across batch rows.
3. Right-pad the suffixes and process them together.
4. Apply the output head only to each row's last real hidden state.

With causal attention, those readout positions do not see later padding. Each
row sees the prefix and its own suffix, not another question. This provides the
intended information boundary, but bit-identical outputs are not a general guarantee.

New local SmolLM2-135M tests found **~0.03 probability drift** between native BF16
single/full-prefill and batched/cached scoring on a mixed structured request.
Converting the test model to FP32 reduced differences below `2e-5`. Tests use that
FP32 oracle for cache correctness and separately check native batch reordering.
The production path still uses the loaded model's precision: native single/batch
threshold stability remains an open issue, not a passed parity guarantee.

Limitations:

- Cache memory scales with question count because the prefix is copied.
- Complete prompts are still tokenized separately.
- Legacy Choice/Noul and Score templates have different text before state and
  can lose state-prefix sharing. structured-v1 fixes that prefix mismatch.
- New questions require three extra content-free evaluations when correction is on.
- There is no request-size bound, microbatch policy, or bounded prior cache.
- Architecture support remains limited. The optimized body/head path is now chosen
  before suffix execution; errors after execution starts propagate rather than
  retrying against a potentially mutated cache. A regression test covers this.

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
| String/object/array state; structured instructions and criteria | Deliberately serialized/validated in v1; historical conversion retained in v0 |
| Choice up to 255 options | 1–26 supported |
| Score with 2–10 levels and weighted mean | Implemented, with different model-side level handling |
| Noul probability, no separate confidence | Response shape implemented |
| Choice/Score confidence semantics | Different from the published adapter |
| Response identifies the actual model version | Returns the loaded backbone name, not a requested Jev alias; full adapter identity remains TODO |
| Validation failure uses HTTP 422 | Implemented for request/schema errors; production error handling remains incomplete |
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
