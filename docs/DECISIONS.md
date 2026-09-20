# Decision log

Original experiment decisions, corrected and annotated on **2026-09-19**. Historical
measurements below were recorded during development, not rerun in this review.
Current behavior lives in [Architecture](ARCHITECTURE.md); future work lives in
[Roadmap](ROADMAP.md).

## 1. Start from an existing small backbone; change it where the task benefits

The initial choice was LoRA plus token readout before custom architecture work.
That remains the baseline, not a ban on architectural changes. The goal is a
fine-tuned small SmolLM/Qwen-style model with decision-specific improvements.
Black-box probes do not confirm Jev's exact head, attention mask, backbone size,
or loss. Earlier claims that fine-tuning accounts for "90% of the value" were not
supported by an experiment and have been removed.

## 2. Decoder baseline, not a new pretrained encoder

A decoder provides pretrained language understanding and a straightforward causal
shared-state path. This is a practical starting point, not proof that encoders
cannot do the task. Compare alternatives on relevant workloads rather than inferring
a hard capacity ceiling from another project's benchmark.

## 3. Initial SmolLM baselines; later controlled SmolLM/Qwen comparison

The initial experiments used SmolLM3-3B for research and SmolLM2-135M for smoke tests.
Subsequent structured-v1 training compared SmolLM2-135M and Qwen3-0.6B; selected Qwen
at LR 1e-5 is now the preferred tested configuration (decision 21). The 3B runs remain
historical. Base checkpoints are not automatically calibrated, and preference tuning
is not categorically disqualifying. Compare instruction-tuned variants rather than
assuming their probability quality.

## 4. MLX is the primary training and serving backend

The development machine is a 36 GB Apple Silicon Mac. MLX supports the current
training and inference loop. The user clarified that efficient MLX serving is
preferred over requiring a Rust rewrite. Rust remains an evidence-gated option;
see decision 19 and the roadmap. Current HTTP serving is not production-ready.

## 5. Route Noul through lettered Choice — v0 workaround, to be revised

Early bare yes/no prompts reportedly showed a 94% yes-bias on SST-2. The recorded
bare-template result was 0.535 accuracy / 0.297 ECE; the lettered/corrected result
was 0.695 / 0.088. These settings are not a complete factorial ablation.

The workaround remains only in legacy rendering, where it erases primitive identity.
Jev documents that Noul and equivalent yes/no Choice probabilities need not agree.
Structured-v1 now preserves the type (decision 20); binary softmax remains the
readout and does not require a separate head by itself.

## 6. Letter labels, including Score levels — v0 limitation

An early numeric-label experiment found ` 0` split into multiple tokens on SmolLM2,
causing collisions when only the first token was read. Letter labels avoid that
case, but must be checked for every tokenizer. Strict single-token and collision
validation now replaces the original warning/first-token fallback.

Letter readout remains capped at 26 options and presents Score levels jointly.
Candidate readout now supports up to 255 options and independent Score descriptions;
its quality limitations are recorded in decision 21. Neither formulation establishes
Jev's private implementation.

## 7. Contextual correction is an optional bias-correction experiment

Divide raw probabilities by the mean content-free distribution. Reported untuned
ag_news accuracy improved 0.75 → 0.80 and Brier 0.40 → 0.31. This is evidence for
that setting, not a universal calibration guarantee. Keep `--calibrate` for the
existing CLI, but distinguish it from fitted temperature scaling and evaluate raw
probabilities too.

## 8. Train the raw readout with cross-entropy

The custom loop trains label-only CE on one-hot or soft targets, unlike a generic
text-completion objective. This matches the raw inference readout, **not** the
optional contextual transformation. Proper scoring rules encourage honest
probabilities in expectation; finite data, imperfect targets, and distribution
shift still cause miscalibration. This is not a disclosed RLCD implementation.

## 9. Use 1e-5 as the initial 3B LoRA learning rate

One recorded 3B run at 1e-4 worsened validation loss from 0.856 to 1.59; a run at
1e-5 reached 0.349. The 135M tolerated 1e-4 in its smoke experiment. Rank 16,
scale 20, all-layer q/k/v/o attention projections remain the defaults. These
observations are not a hyperparameter optimum or a universal failure threshold.

## 10. Hold out datasets, then add true rubric/task-family holdouts

The historical recast training tasks are ag_news, dbpedia, imdb, and yelp_stars.
SST-2 and tweet_emotion
are excluded by `TRAIN_TASKS`, but SST-2's canonical question and criteria exactly
match IMDB. Earlier "unseen questions" claims for SST-2 were incorrect.

Keep these transfer diagnostics, add genuinely unfamiliar rubrics and Score
families, and do not describe public test sets as uncontaminated by pretraining.
Kev's local manifest holds out examples/groups, not entire sources.

## 11. Distill from Jev as an experiment, not a substitute for outcome evaluation

Soft targets retain more of a teacher distribution than argmax labels. Whether
that improves this student's calibration requires a matched comparison. The current
2,000-row artifact has 86.3% target-argmax agreement with gold and 52.15% of rows
with `max(target) < 1`; these recomputed statistics replace the older 88.4% / 45%
notes. See [Data](DATA.md) for exact definitions, overlap, and rounded target sums.

Teacher mistakes can transfer too. Check TypeSafe's terms before distributing
weights. Pin teacher versions, validate distributions, and resume by IDs in the
next collector; current `--resume` only skips per-task row counts.

## 12. Add option-order augmentation

The current gold builder shuffles Choice/Noul options with target remapping and
drops Choice descriptions on about 20% of rows. Score order is preserved. Fixed
orders can teach positional shortcuts. The historical adapters are recorded as
pre-shuffling runs; neither the teacher collector nor Kev converter currently
applies equivalent augmentation. Compare matched augmentation before attributing
differences to target type.

## 13. Share state computation using token-level common prefixes

The original implementation scored suffixes sequentially and rewound a cache.
That description is historical: the current opt-in shared path tiles prefix KV
across microbatch rows and scores suffixes together, recomputing the prefix for each
microbatch. Independent execution is now the default after the drift investigation.
Token-level matching avoids string/BPE boundary assumptions. Question contexts remain
isolated, but cache copying and mixed-template prefix mismatches limit reuse.

## 14. Single-threaded HTTP/1.1 development server (historical)

This configuration followed an observed MLX stream error with threaded handlers
and a client integration issue with HTTP/1.0. It is not a general prohibition on
threaded serving or a universal statement about Node fetch. A model-owning worker,
bounded queue, and concurrency tests now replace that configuration (decision 21).

## 15. Dataset substitutions

The initial loader replaced a legacy-script TREC source with `fancyzhx/dbpedia_14`
and used the namespaced `fancyzhx/ag_news` ID. These were environment-specific
loader decisions. TREC is now available in the imported Kev files; it is not
implemented as a built-in recast task. Preserve source revisions for future runs.

## 16. Batch suffixes and compute only the necessary inference head positions

The optimized path avoids projecting every suffix position to the vocabulary.
Historical 135M timings were 0.29 s for 50 questions and 0.30 s for 100 under load.
They do not establish a 3B latency target or numerical identity across batching.
Packed attention remains an option, but does not inherently require retraining if
it preserves the same computation, nor does a dense mask guarantee a speedup.

## 17. Replace speculative architecture tiers with measured milestones

Reserved option tokens, slot/pointer heads, per-level Score readouts, and packed
attention are candidates, not confirmed Jev internals or mandatory upgrades.
The 255-option cap is an API constraint, not proof of a fixed token/head layout.
Prioritize data integrity, structured inputs, primitive identity, and a controlled
model comparison. Contrastive data remains open; per-level readout and temperature
scaling were subsequently implemented and evaluated (decision 21).

## 18. Separate completed code, recorded evidence, and unimplemented work

At the initial documentation review, the repository had a prototype and one SDK
smoke test, not full compatibility certification. Confidence, serialization, error
handling, model identity, and limits needed changes. Distilled adapters and recorded
three-way results already existed, so the earlier "in flight" status was stale.

Decisions 20–21 record the subsequent implementation and experiments: v1 now pins
public confidence formulas, preserves structured JSON, and bounds serving work.
Local calibration/development partitions were derived from verified Kev training
files. The missing upstream partitions and invalid Nimble downloads remain distinct
issues. Full Jev compatibility is still not established.

## 19. Optimize MLX before committing to a Rust runtime

The desired deliverable is the small fine-tuned decision model plus efficient
inference, not a language rewrite for its own sake. Keep MLX as the main server
option. Measure GPU time, tokenization, scheduling, and KV memory separately.
Consider Rust only when it addresses a measured bottleneck or deployment need,
and require model/tokenizer/rendering parity for any second backend.

## 20. Implement data and rendering foundations before changing model heads

Added a provenance-preserving Kev importer, deterministic connected-group splits,
checksum/target/leakage validation, and canonical external evaluation. The prepared
Kev-only corpus keeps test reserved and derives development/calibration from
training; it does not make the old combined corpora safe or repair Nimble downloads.

`structured-v1` preserves JSON boundaries and Noul identity with a common state
prefix. Adapter metadata pins rendering; old adapters select `legacy-v0`. Training
now rejects invalid label tokens and overlapping validation data, seeds MLX, and
records hashes/arguments. The server preserves v1 structured state, returns 422 for
validation failures, and reports the actual loaded backbone instead of a Jev alias.

Regression tests cover data, rendering, HTTP behavior, cache retry safety, and a
two-step 135M adapter round trip. Native BF16 single/batch inference differed by
~0.03 on a fixture; FP32 reference parity passed. That precision issue remains
open, rather than being hidden behind a loose tolerance. No full v1 quality run,
specialized primitive head, fitted calibration, or bounded service was claimed at
that milestone.

## 21. Follow evidence across the five requested experiment tracks

Diagnosed shape-dependent low-precision drift and made independent execution the
default. Shared execution remains explicit, with FP32 reference tests rather than
relaxed native tolerances. Both backbones were actually trained on identical admitted
examples. Qwen regressed at 5e-5 but reached 81.3% development accuracy at the
predeclared 1e-5 control; select on development, not held-out rubric performance.

Keep candidate readout experimental: description isolation and 255-option execution
work, but a matched-source pilot is not matched compute, and 77-way quality was
near chance. Fit temperatures only on calibration, retain outcome metrics separately
from teacher fidelity, and distinguish pinned public confidence formulas from live
Jev equivalence. Cached teacher experiments cannot invent the missing teacher version.

Serving now loads and executes MLX on one worker behind bounded HTTP/queue admission.
Limits cover bytes, tokens, views, microbatches, and caches. Deadlines discard expired
queued work but cannot force-preempt a running Metal kernel. Keep loopback defaults,
report actual model identity and measured performance, and do not equate these bounds
with production security or demonstrated workflow safety.

See [Experiments](EXPERIMENTS.md) for measured results and remaining evidence gates.

## 22. Select the synth+RPS Qwen configuration (user-directed promotion)

`qwen3-0.6b-structured-v1-synth-rps` (kev train + 2,003 jev-1.13.0-scored synthetic
questions, ordinal RPS loss at weight 1.0, LR 1e-5, seed 42) replaces the kev-only
LR-1e-5 run as the selected development configuration. Evidence: it retains the
synthetic out-of-family gains (+16.8 accuracy points on the held-out synthetic
evaluation, 95% grouped interval [+10.8, +22.5]; NLL -0.414), ties the best rubric
accuracy (83.3%), reaches the family's best development NLL (0.431), and shows no
significant in-family regression. The RPS effect replicated on SmolLM2-135M with
significant across-the-board improvements (accuracy +3.0 points [+0.9, +5.2],
NLL -0.059, Brier -0.029), though there the score-MAE change itself was within
noise -- the loss generalizes; its expression differs by backbone.

Open gates, explicitly not yet met: seed replication of the combined run; the
residual rubric-NLL elevation from synthetic data (filter teacher-overconfident
missing-evidence rows and retrain); the reserved Kev test remains untouched.
Temperatures were refitted for this adapter on the calibration partition
(data/runs/synth-rps-v1/temperature.json) against a combined-corpus manifest
whose development/calibration/test partitions are byte-identical to kev-v1.

## 23. Review corrections to the synthetic/RPS arc (2026-09-20)

An internal review of decisions 20-22 raised five issues; dispositions:

1. **Non-ordinal synthetic Score rubrics (confirmed).** Some generated Score
   questions present categorical outcomes as levels (verified example:
   resolution outcomes as a 6-level "scale"). The loader marks every
   letter-readout Score row ordinal, so RPS trained arbitrary distances on
   those rows in the combined arm. Scope: 579/2,003 synthetic rows are Score
   with unknown contamination fraction; the RPS validation itself is
   unaffected (both replications trained kev-only, whose Score sources are
   genuinely ordinal). The generator now requires and records a declared
   `scale_dimension` and forbids categorical levels; existing synthetic Score
   rows need an ordinality audit before the next data cycle.
2. **Synthetic evaluation measures teacher agreement (acknowledged; fidelity
   added).** Soft-distribution fidelity now sits beside argmax agreement:
   the selected adapter improved mean JS divergence to the teacher's full
   distributions (0.187 -> 0.117 nats) and Noul MAE (0.289 -> 0.222), so the
   agreement gain did not come from discarding uncertainty. Outcome
   correctness still requires the independently adjudicated benchmark.
3. **Overconfidence filter refined (confirmed).** A teacher confidently
   selecting an explicit cannot-determine option is correct uncertainty
   handling. The filter now exempts such rows (29 of the original 232;
   203 dropped); the queued retrain uses the corrected corpus.
4. **Promotion evidence is a tradeoff, not a clean win (acknowledged).** The
   development NLL interval crosses zero, and the rubric diagnostic pairs
   +3 correct answers with worse NLL/Brier. Decision 22 stands as a
   user-directed selection with these limits; seed evaluations and the
   filtered retrain are in flight.
5. **Reproducibility gaps fixed.** audit_synthetic now exits nonzero on FAIL;
   the teacher collector validates resumed rows against the requested teacher
   and current scenario content, and aborts if the API reports a different
   version mid-collection.

