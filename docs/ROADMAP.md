# Model and serving roadmap

Status: **recipe frozen (decision 25); reserved test spent (77.6% / ECE 0.049,
Experiments §9); scale-up phase open**.
Updated 2026-09-19. Acceptance gates below are not all met. The numbered phases
retain the full plan, including completed steps; the status summary identifies
current progress and outstanding evidence.

## Implementation status

- **Phase 1, partial:** verified, provenance-preserving Kev importer; canonical
  records; connected-group train/dev/calibration/test splits; hashes and leakage
  checks. A local corpus is prepared. External gold evaluation saves per-example
  predictions. Cross-source migration, teacher versioning, task-family holdouts,
  and independent human review remain open. Tokenizer/weight revisions, asset hashes,
  package versions, and source hashes are now recorded.
- **Phase 2, partial:** shared v1 JSON renderer; explicit primitive identity; legacy
  compatibility; strict label tokens; request validation; Python/HTTP/model smoke
  tests. Published adapter confidence formulas are pinned; independent execution
  avoids sibling-dependent shapes. Live Jev parity and broader backbone tests remain open.
- **Phase 3, experiments completed:** real matched SmolLM/Qwen runs, Qwen learning-rate
  control, candidate/letter pilot, 77-way quality test, 255-option contract test,
  fitted calibration, synthetic held-out rubrics, and matched cached-teacher regimes.
  Quality results are mixed; these are not all acceptance gates passed.
- **Phase 4, implemented/measured:** model-owning worker, bounded HTTP/queue admission,
  token/view/cache budgets, deadlines, discovery, telemetry, and HTTP benchmark tooling.
  Hard kernel cancellation, authentication, paged KV, and deployment certification remain open.
- **Phase 5, pending:** demonstrate workflow value and risk–coverage on independently
  reviewed frozen cases. No evidence currently requires a Rust rewrite.

The stability investigation found up to 0.031 drift on SmolLM2-135M and 0.060 on
Qwen3-0.6B, including a native Qwen argmax flip. Independent execution is now the
stable default; shared execution is opt-in. FP32 shared tests reduced drift below
0.00001 on the measured fixtures. See [Experiments](EXPERIMENTS.md). Production
serving acceptance still requires broader quality and load tests.

**Current decision point: serving tiers.** The MiniCPM scale-up won selection
(experiments §11), its patch pass was a clean null (§12), quantization is
measured (§13), and the decision-27 gate is closed: q8 with its own fitted
temperatures is the recommended quality-tier serving configuration (decision 28).
What remains for the tier decision proper: a fresh reserved split for MiniCPM
(no unbiased test exists for the 2B — the old partition was spent on the 0.6B),
and a documented latency/quality/footprint policy per tier — e.g., Qwen 0.6B BF16
for the volume tier, MiniCPM q8 for the quality tier — rather than a single
recommended adapter.

## Destination

A small fine-tuned model based on an existing SmolLM/Qwen-style backbone, with
targeted architectural changes for Jev-like decisions. **MLX is the main training
and serving option.** A Rust runtime is optional, justified by profiling rather
than a prerequisite for completion.

Keep three contracts separate:

- **API:** typed requests/responses, validation, model identity, and documented limits.
- **Model:** narrow judgments over unfamiliar instructions, criteria, and structured
  state; useful probabilities; primitive-specific behavior.
- **Runtime:** isolated questions, bounded memory, and measured efficient execution.

The goal is not exact reconstruction of Jev's private weights or architecture.
Do not intentionally reproduce its errors or promise its knowledge breadth in 3B.

## 1. Establish trustworthy data and a testable reference

- Validate external files before ingestion. Replace the two Nimble 404 bodies only
  when a valid source, schema, license, and split provenance are available.
- Define canonical records containing original structured state/question, primitive,
  source revision, record/group ID, target origin, and variant. Render at training
  time or store an explicit renderer version alongside derived prompts.
- Preserve Kev metadata and group its clean/permuted/none-option variants together.
  Exclude training overlap with every held-out partition across sources.
- Obtain missing calibration/development partitions or document a disjoint split
  from training groups. Keep Kev test reserved and SST-5 excluded for SST-2 evaluation.
- Validate teacher distributions; normalize rounded targets while retaining originals.
  Pin teacher versions and make pulls resumable by record identity.
- Add external-data evaluation, saved per-example outputs, and run manifests.

**Exit:** malformed inputs fail early; hashes/counts/targets are checked; all
training/dev/calibration/test groups are demonstrably separated under the declared
matching policy. Data-file names alone are not a split policy.

## 2. Preserve the input and primitive contract

- Implement one unambiguous serializer for structured state, instructions, and
  criteria, shared by data building, inference, and teacher comparisons.
- Preserve primitive identity: Noul must not silently become an indistinguishable
  Choice prompt. A type marker or distinct template is a minimal first change.
- Use a common state prefix across primitive types where possible. Type-specific
  processing belongs after the shared state if it is to remain reusable.
- Add strict label-token checks, finite-probability checks, and stronger schema
  validation. Align confidence with a pinned reference and test its semantics.
- Add tests for structured round trips, question IDs, padding, mixed primitives,
  batched/sequential parity, cache failure recovery, and adapter loading.

**Exit:** the new renderer has golden fixtures and a version. Tests preserve
information and primitive identity; changing question IDs has no effect. Old
adapters/results remain labeled as v0 rather than silently reused as validation.

## 3. Train and compare small decision-model architectures

Keep the restricted-token model as a baseline while testing targeted changes:

| Component | Initial experiment | Decision evidence |
|---|---|---|
| Noul | Type-aware binary readout; compare a dedicated scalar/binary head if needed | Outcome metrics and teacher fidelity, especially ambiguous inputs |
| Choice | Joint option-aware readout; compare slot projection or option-span scorer | Accuracy, NLL/Brier, option permutations, wide-option behavior, memory |
| Score | Current joint-level baseline versus shared per-level scorer | Held-out descriptive scales, ordinal errors, calibration, latency |
| Shared state | Reuse backbone state computation with isolated suffix branches | Numerical parity, quality, memory, mixed-request throughput |

A per-level Score scorer can condition on state, instructions, and one description,
then combine level scores into a distribution. Its normalization and training
objective are design choices to evaluate; the public docs do not reveal Jev's loss.
A pointer-style Choice scorer must still have access to joint option context when
handling relational alternatives such as "none of the above".

A 255-option limit is a compatibility target, not evidence of 255 reserved tokens
inside Jev. Compare reserved tokens, slot heads, and option-based scoring only when
expanding beyond v0. If using reserved tokens, account for embedding/output-row
training; the current attention-only LoRA recipe does not train those rows.

Training work:

- Cover multiple questions per state, unfamiliar rubrics, structured paths, evidence
  changes, absent evidence, and contrastive criteria. More topic/sentiment examples
  alone do not establish these abilities.
- Compare compatible small SmolLM/Qwen backbones under the same protocol. Base versus
  instruction-tuned is an experiment, not a categorical restriction.
- Compare gold, teacher-argmax, teacher-soft, and mixed targets on matched examples,
  augmentation, and optimizer exposure. Keep outcome accuracy separate from fidelity.
- Retain implemented MLX seeding, data/configuration manifests, tokenizer/backbone
  hashes and revisions, and environment details. Add periodic/resumable checkpoints.
- Fit any temperature parameters only on a separate calibration partition.

**Exit:** a frozen task-family/rubric holdout suite shows useful predictive quality
and uncertainty, with intervals and failure analysis. Architectural changes earn
their complexity through quality or efficiency gains, not resemblance to a diagram.

## 4. Build a bounded MLX service

Start profiling and a minimal worker prototype alongside model work; do not wait
for all research experiments before testing the serving design.

- Keep inference on a controlled MLX execution context; put request handling behind
  a bounded queue with backpressure, timeouts, and safe error handling.
- Bound state/question/option tokens, batch size, and calibration-cache growth.
  Define limits for the selected backbone and machine, not Jev's server hardware.
- Avoid repeated state tokenization and full-sequence vocabulary projection where
  safe. Microbatch suffixes to avoid an unbounded prefix-KV replication cost.
- Evaluate genuine shared-prefix or packed attention only with kernels that realize
  the expected memory/compute savings. A dense block mask alone is not enough.
- Serve actual model/adapter identity, model discovery, validation errors, and a
  documented usage-accounting policy. Keep local-only defaults until hardened.
- Package tokenizer/rendering/readout/calibration configuration with weights and
  numerical reference fixtures, even if MLX remains the only runtime.

**Exit:** report end-to-end p50/p95 latency, throughput, and peak memory on the target
Apple Silicon hardware. Sweep state length, question and option counts, mixed types,
concurrency, and cold/warm correction caches. Verify parity with the reference path
within predeclared tolerances. No "100 questions cost one" claim without measurements.

## 5. Prove workflow value, then consider another runtime

Build one representative workflow, such as support triage or candidate extraction.
Code handles deterministic rules, permissions, and actions; the model handles atomic
judgments. Evaluate action correctness, review rate, risk–coverage, and asymmetric
error cost on frozen cases. Test missing evidence and irrelevant speculative answers.

Use a larger model or human escalation path for hard cases. A composite weighted
score is a policy feature, not automatically a calibrated probability of an event.

Only investigate Rust if measurements identify a benefit:

- CPU/tokenization or request-management overhead may justify a Rust component.
- GPU-bound work requires better model/kernels/batching, not merely a language change.
- A full second runtime must support the chosen architecture, tokenizer, precision,
  adapters or merged weights, caches, and primitive outputs with parity tests.

Retain MLX if it meets the operating envelope. Do not maintain two backends merely
to satisfy the original runtime idea.
