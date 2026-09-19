# Overview

Reviewed 2026-09-19 against the current code, local data, and the public Jev docs.

## Goal

Build a small, fine-tuned **System One model** on an existing pretrained backbone
such as SmolLM or Qwen. Make targeted architectural changes for typed decisions,
useful uncertainty, and efficient multi-question inference. Do not pretrain a new
foundation model or assume we have recovered Jev's private architecture.

**MLX is the primary training and serving backend on Apple Silicon.** The current
Python server is a development implementation. Rust remains an optional deployment
path if measured limits justify the extra backend and parity work.

The intended interface is:

> Structured evidence + a narrow, request-defined judgment → a typed probabilistic answer.

| Primitive | Intended question | Answer fields |
|---|---|---|
| Choice | Which supplied option fits best? | `choice`, `probabilities`, `confidence` |
| Score | Where does the state fit on a descriptive scale? | `score`, `probabilities`, `legend`, `confidence` |
| Noul | Is this statement true? | `noul`: probability of yes |

Application code owns arithmetic, control flow, authorization, and side effects.
Ask independent atomic questions together, including speculative questions that
some code paths will ignore. Make another call when earlier answers determine
new evidence or options. Question independence is not statistical independence.

## Success criteria

1. **Useful decisions with useful uncertainty.** Measure predictive quality,
   NLL/Brier, calibration, and how much work can be automated at a chosen error cost.
   Low calibration error alone is not success.
2. **Transfer to unfamiliar rubrics.** One model interprets new instructions and
   criteria at request time. Evaluate by held-out rubric and task family, not only
   by held-out document or dataset.
3. **Jev-like primitive behavior and API compatibility.** Preserve structured
   inputs, primitive identity, isolated questions, and typed responses. Distinguish
   wire compatibility, behavioral agreement, and correctness against outcomes.
4. **Efficient local serving.** Establish latency, memory, and throughput limits on
   the actual target hardware. Prefer improving MLX before adding another runtime.
5. **A reproducible recipe.** Version datasets, rendering, targets, weights, and
   evaluations. The current scripts do not yet capture all of that provenance.

## Scope and limitations

Start with English text and JSON state, narrow semantic judgments, and small
backbones. SmolLM3-3B is the current research model; SmolLM2-135M checks the pipeline.
Qwen is a candidate comparison, not an implemented second architecture.

Non-goals include free-form generation, multimodal inference, exact arithmetic,
and matching a much larger model's broad knowledge. Jev itself documents weaknesses
in counting, dates, indirection, adversarial state, and irrelevant long context.
Its reported benchmark scores do not identify its parameter count or prove an MoE
architecture.

Typed readout prevents invented output labels; it does not prevent wrong answers,
invalid input, runtime failures, or prompt injection. Successful valid requests
return every question ID, but this is not an unconditional availability guarantee.

## Why this approach

Restricted-logit readout avoids autoregressive answer generation and parsing.
A pretrained model supplies language understanding; fine-tuning adapts it to
request-defined decisions. Small architectural changes can preserve primitive
semantics and reduce inference work without replacing the entire backbone.

Token probabilities are not automatically outcome probabilities. Neither a base
checkpoint, proper-scoring-rule loss, teacher distillation, nor contextual bias
correction guarantees calibration after deployment. Base and instruction-tuned
checkpoints should be compared rather than excluding the latter categorically.

## Current evidence and status

Implemented: the MLX readout engine, LoRA training, recast data building, Kev
conversion, Jev target collection, recast evaluation, permutation testing,
teacher-agreement evaluation, and a development server with a TypeScript SDK smoke
test. This is not a comprehensive correctness or compatibility suite.

Gold-label and distilled 3B adapter files exist locally. Historical evaluation
results now include both; distillation is no longer merely "in flight".
[Evaluation](EVALUATION.md) preserves the numbers and their limitations.

The SST-2 result measures transfer from IMDB to another movie-review dataset:
the canonical question and criteria are identical. Tweet emotion provides a
held-out question relative to the four-task recast mix, but not broad evidence
of arbitrary-rubric generalization. There is no held-out Score family in that mix.

The new external data expands coverage to entailment, reading comprehension, and
question classification. It is not ready for blind concatenation: see
[Data](DATA.md) for invalid Nimble downloads, overlap, and missing split artifacts.

## Evidence sources and how to interpret them

Primary sources, reviewed 2026-09-19:

- [Introduction](https://docs.typesafe.ai/introduction),
  [building workflows](https://docs.typesafe.ai/concepts/how-to-build-with-system-one),
  and [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer).
- [Primitives](https://docs.typesafe.ai/primitives),
  [structured inputs](https://docs.typesafe.ai/primitives/advanced),
  [confidence](https://docs.typesafe.ai/confidence), and [HTTP API](https://docs.typesafe.ai/api).
- [Models](https://docs.typesafe.ai/models) and
  [Jev 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).
- [Launch article](https://typesafe.ai/blog/introducing-system-one-models-and-jev):
  explains the product and workflow evaluations; their model-consensus references
  are not observed-outcome calibration measurements.

Secondary sources and implementation references:

- [Archer Hume's probes, revised article](https://archerhume.com/posts/jevs-architecture-unmasked/?v=3):
  supports behavioral isolation and Choice option interactions. The head, attention
  implementation, backbone size, MoE routing, and exact training objective remain
  unconfirmed. Choice evidence must not be generalized to Score without testing.
- [SGNT's explanation](https://sgnt.ai/p/jev/): distinguishes a restricted-logit
  wrapper from a trained decision model; cross-project results are not controlled
  comparisons with this repository.
- [Kev](https://github.com/jaredpalmer/kev) and
  [Nimble](https://github.com/bespokelabsai/nimble) are data/design references, not
  evidence that one architecture or training recipe must be used here.

See [Architecture](ARCHITECTURE.md) for known divergences and
[Roadmap](ROADMAP.md) for the next implementation stages.
