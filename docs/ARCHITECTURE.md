# Architecture and compatibility

This page separates **current code**, **Jev's documented behavior**, and **planned
changes**. Reviewed 2026-09-19. MLX remains the primary training and serving backend.

## Current model: restricted-token readout

`src/jev/engine.py` loads an MLX language model, optionally with a LoRA adapter.
It uses the framework-independent `src/jev/rendering.py` to render prompts, reads
final-position logits for label tokens and applies a restricted softmax. The default
`letters-v1` uses ` A` through ` Z`; `candidate-v1` combines binary candidate
judgments. Application code serializes answers; there is no answer-generation loop.

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
       ├─ render_views(state, question) + admission budgets
       ├─ _score_batch(prompts) + combine_views
       ├─ _apply_calibration(question, probabilities)  # optional bias correction
       ├─ temperature_scale(probabilities)             # optional fitted artifact
       └─ _to_answer(question, probabilities)
```

### Primitive rendering

| Primitive | Current computation | Important limitation |
|---|---|---|
| Choice | Joint letter softmax, or binary candidate views with all alternatives visible | Letters cap at 26; candidates support 255 but require multiple forwards |
| Noul | Binary letter readout `A=no`, `B=yes`; return `p_yes` | v1 retains a Noul type marker; legacy-v0 erases it |
| Score | Joint letter baseline, or independent description-only binary views normalized afterward | The normalization/objective is our design, not a disclosed Jev implementation |

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
The optional candidate readout implements description isolation; the default joint
readout does not. Adapters pin their readout and reject mismatches. The docs do not
disclose Jev's exact scoring, normalization, or attention implementation. Candidate
training uses weighted Bernoulli views; see [Experiments](EXPERIMENTS.md).

### Execution policy update

Independent scoring is now the default, preventing sibling questions from changing
the compute shape of a decision. Shared prefix caching requires
`--execution-mode shared`; `--precision float32` substantially reduces measured
drift. Native shared execution is experimental. Both paths project only the last
hidden position on the verified Llama/Qwen3/SmolLM3 families. Unverified families
retain their full framework wrapper and independent execution, preserving any
wrapper-specific logit transforms instead of bypassing them by attribute matching. See [Experiments](EXPERIMENTS.md) for
SmolLM/Qwen measurements; historical timing and projection results are not a parity
guarantee for the updated path.

### Versioned structured input

`src/jev/serialization.py` validates JSON, rejecting duplicate object keys,
nonfinite numbers, unsupported values, and excessive nesting. Schema entries
accept strings, objects, arrays, or null; state accepts strings, objects, or arrays.
The schema permits 255 Choice options; letter readout is limited to 26. Candidate
readout requires structured-v1. Training and inference require distinct single-token
labels. Invalid Unicode surrogates are rejected before tokenization.

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

Independent execution runs each prompt separately. In opt-in shared mode,
`_score_batch` processes at most four views at a time, finds their longest common
token prefix, and uses caching when that prefix has at least eight tokens:

1. Encode the common prefix once into a KV cache.
2. **Physically repeat** each layer's cached keys and values across batch rows.
3. Right-pad the suffixes and process them together.
4. Apply the output head only to each row's last real hidden state.

With causal attention, those readout positions do not see later padding. Each
row sees the prefix and its own suffix, not another question. This provides the
intended information boundary, but bit-identical outputs are not a general guarantee.

Measured native BF16 single/shared drift reached **0.031 on SmolLM and 0.060 on
Qwen**, including a Qwen argmax flip. FP32 substantially reduced drift on tested
fixtures. Default independent execution avoids sibling-dependent computation shapes;
it does not make native shared execution numerically safe. Tests retain the FP32
structural oracle and explicit native drift measurements.

Limitations:

- Prefix KV is physically copied within the bounded microbatch; no paged cache.
  Common prefixes are currently recomputed per microbatch, not reused across all chunks.
- Distinct complete prompts are tokenized separately; repeated prompts use a bounded
  token LRU, not a reusable state-embedding cache.
- Legacy Choice/Noul and Score templates have different text before state and
  can lose state-prefix sharing. structured-v1 fixes that prefix mismatch.
- New questions require three extra content-free evaluations when correction is on.
- Request/state/prompt/view budgets and bounded caches are implemented. These are
  local operating ceilings, not a guarantee against every model/device OOM.
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
Evaluate raw and corrected predictions separately. `scripts/fit_calibration.py` fits
per-primitive temperatures only on a declared disjoint calibration partition.
`--temperature FILE` verifies weights/tokenizer and inference configuration. It is
separate from `--calibrate`; neither guarantees calibration under distribution shift.

### Confidence

Structured-v1 defaults to the formulas in the published
[TypeSafe adapter at revision fb52b103](https://github.com/typesafe-ai/system-one-adapter-python/blob/fb52b1030b7fc1f4f1cf39910afa5da54f9835e3/src/system_one_adapter/_utils/confidence_metrics.py):

- Choice uses `(p_max - 1/K) / (1 - 1/K)`, with a one-option special case.
- Score uses distance from the modal level relative to a uniform reference.

The [confidence docs](https://docs.typesafe.ai/confidence) describe a derived
statistic but do not specify an exact versioned formula. Live Jev equivalence is
not verified. Legacy rendering defaults to `entropy-v0`, which can also be selected
explicitly. These measures are not correctness probabilities. Do not transfer
thresholds across definitions or primitives.

## API compatibility: current subset

| Contract | Current repository |
|---|---|
| `POST /v1/systemone`, answers keyed by question ID | Implemented for basic requests; IDs stay out of prompts |
| String/object/array state; structured instructions and criteria | Deliberately serialized/validated in v1; historical conversion retained in v0 |
| Choice up to 255 options | Candidate readout supports all 255; quality at large width is not established |
| Score with 2–10 levels and weighted mean | Joint and independent-level variants implemented |
| Noul probability, no separate confidence | Response shape implemented |
| Choice/Score confidence semantics | Pinned public adapter formulas by default in v1; no verified live parity |
| Response identifies the actual model version | Backbone/adapter/configuration/implementation fingerprint, not a Jev alias |
| Validation failure uses HTTP 422 | Implemented; body/framing, overload, timeout, and internal errors have separate statuses |
| `GET /v1/models` | Implemented, with local metadata and effective limits |
| Context limits and usage | Local token/view ceilings; scoring-work counters are not Jev billing semantics |

Jev 1.13's [model page](https://docs.typesafe.ai/models) specifies 64k total request
tokens and 32k for state plus the longest question. These are Jev's limits, not
limits established for our model or hardware. Some introductory docs round the
budget differently; use version-specific model documentation and contract tests.

`usage.input_tokens` currently counts scoring work, including uncached correction
priors; `output_tokens` is always zero. Jev can report nonzero output tokens without
autoregressive generation. A compatibility policy for usage still needs definition.

## Serving: MLX first

`scripts/serve.py` uses bounded HTTP threads and a bounded queue feeding one worker
that loads and owns the model. HTTP threads never execute MLX operations. This
addresses the earlier handler-thread stream failure without claiming arbitrary MLX
thread safety. HTTP/1.1 connections close after each response.

Admission budgets, LRU caches, a free-buffer allocator cap, cooperative deadlines,
backpressure, model discovery, and runtime telemetry are implemented. Running Metal
kernels cannot be force-preempted safely. There is no authentication or TLS; binding
remotely requires an explicit override. See [Serving](SERVING.md) for the operating
envelope and [Experiments](EXPERIMENTS.md) for measured results.

Small architectural changes are part of the model goal: primitive-aware inputs,
possible specialized readouts, and efficient shared-state execution. Select changes
using controlled quality, memory, and latency experiments.

A Rust front end or full runtime is optional. First determine whether the bottleneck
is Python/tokenization, scheduling, GPU kernels, or model capacity. Rust alone does
not reduce transformer FLOPs or fix duplicated KV tensors. A second backend also
requires tokenizer, rendering, weights, and numerical-parity tests. See
[Roadmap](ROADMAP.md) for sequencing and acceptance criteria.
