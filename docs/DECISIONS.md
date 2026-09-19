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

## 3. SmolLM3-3B research baseline; SmolLM2-135M smoke tests

These are the implemented backbones used in the recorded experiments. Qwen remains
a candidate comparison. Base checkpoints were chosen to avoid preference-trained
output biases, but base models are not automatically calibrated. The former
"base checkpoints only" rule is superseded: compare instruction-tuned variants
when their rubric understanding may help.

## 4. MLX is the primary training and serving backend

The development machine is a 36 GB Apple Silicon Mac. MLX supports the current
training and inference loop. The user clarified that efficient MLX serving is
preferred over requiring a Rust rewrite. Rust remains an evidence-gated option;
see decision 19 and the roadmap. Current HTTP serving is not production-ready.

## 5. Route Noul through lettered Choice — v0 workaround, to be revised

Early bare yes/no prompts reportedly showed a 94% yes-bias on SST-2. The recorded
bare-template result was 0.535 accuracy / 0.297 ECE; the lettered/corrected result
was 0.695 / 0.088. These settings are not a complete factorial ablation.

The workaround remains in code, but it erases primitive identity. Jev documents
that Noul and equivalent yes/no Choice probabilities need not agree. Preserve the
type in the next rendering/training version; a binary softmax is still a valid
candidate readout and does not require a separate head by itself.

## 6. Letter labels, including Score levels — v0 limitation

An early numeric-label experiment found ` 0` split into multiple tokens on SmolLM2,
causing collisions when only the first token was read. Letter labels avoid that
case, but must be checked for every tokenizer. The code's warning/first-token
fallback is not sufficient validation.

Choice is capped at 26 options. Score currently presents all levels together;
Jev's docs describe separate-level evaluation. Compare those formulations rather
than assuming Choice and Score share the same internals.

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

Training tasks are ag_news, dbpedia, imdb, and yelp_stars. SST-2 and tweet_emotion
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
That description is historical: current code tiles prefix KV tensors across batch
rows and scores suffixes together. Token-level matching avoids string/BPE boundary
assumptions. Question contexts remain isolated, but cache copying and mixed-template
prefix mismatches limit reuse.

## 14. Single-threaded HTTP/1.1 development server

This configuration followed an observed MLX stream error with threaded handlers
and a client integration issue with HTTP/1.0. It is not a general prohibition on
threaded serving or a universal statement about Node fetch. Production work needs
a controlled inference worker, bounded queue, and concurrency tests.

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
model comparison. Contrastive examples and temperature scaling remain planned.

## 18. Separate completed code, recorded evidence, and unimplemented work

The repository has a prototype engine and scripts plus one SDK smoke test, not
full compatibility certification. Confidence differs from the published adapter;
structured serialization, error handling, model identity, and limits need work.
The distilled adapter and three-way results now exist, so earlier "in flight"
status was stale. The local Nimble files are 404 bodies; Kev checksums match its
manifest, but calibration/development files are missing.

This documentation pass changes no model code, weights, or data. New requirements
and tests are listed in the roadmap, not described as finished features.

## 19. Optimize MLX before committing to a Rust runtime

The desired deliverable is the small fine-tuned decision model plus efficient
inference, not a language rewrite for its own sake. Keep MLX as the main server
option. Measure GPU time, tokenization, scheduling, and KV memory separately.
Consider Rust only when it addresses a measured bottleneck or deployment need,
and require model/tokenizer/rendering parity for any second backend.
