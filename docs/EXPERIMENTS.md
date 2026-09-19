# Implementation experiments

Local Apple Silicon measurements, 2026-09-19. These are development results, not
Jev compatibility certification or a claim of unseen-task generalization.

## 1. Numerical stability

Protocol: structured-v1 mixed Noul/Choice/Score fixtures, 3 and 12 questions,
1 and 16 repeated state messages, five warm timing repetitions. Compare shared
prefix execution with independent execution at the **same** precision. Both paths
now project only the last hidden position, avoiding full-sequence vocabulary output.

| Backbone | Native BF16 max probability drift | FP16 drift | FP32 drift |
|---|---:|---:|---:|
| SmolLM2-135M | 0.03113 | 0.00374 | 0.00000331 |
| Qwen3-0.6B | 0.05965 | 0.00745 | 0.00000960 |

Native Qwen changed the argmax on one of the tested question occurrences. This
is not just display rounding. The strong precision dependence, zero observed
reordering drift, and FP32 reference agreement support shape-dependent numerical
rounding rather than cross-question attention as the cause. This is a diagnosis
of these fixtures, not proof about every kernel/backbone/context.

**Mitigation:** default to `execution_mode=independent`, which gives each prompt
the same computation regardless of siblings. `--execution-mode shared` is explicit;
use `--precision float32` when exploring the shared-prefix optimization. FP16 did
not meet a 0.002 probability-drift gate. Native shared mode remains experimental.

Warm 12-question, longer-state fixture (not HTTP latency):

| Backbone | Native independent p50 | FP32 shared p50 | FP32 shared peak MLX active memory |
|---|---:|---:|---:|
| SmolLM2-135M | 151.7 ms | 49.8 ms | 1.34 GB |
| Qwen3-0.6B | 403.0 ms | 113.4 ms | 4.06 GB |

Only five repetitions were used; this is not a reliable production p95 estimate.
Memory is MLX peak active allocation, not process RSS or complete unified memory.
The two backbones differ in size and tokenizer; these are practical comparisons,
not an isolated architecture effect.

Reproduce with `scripts/benchmark_engine.py --model MODEL --out FRESH.json`.
Full local reports: `data/runs/stability-smollm2.json` and
`data/runs/stability-qwen3.json` (gitignored). No quality conclusions follow from
this synthetic benchmark. The original full-prefill/projection path can also differ
numerically from last-position projection; new experiments must use matched code.

## 2. Candidate readout implementation

`--readout candidate-v1` is an explicit alternative to `letters-v1`, recorded in
adapter metadata. A candidate adapter cannot silently use letter readout, or vice
versa. This changes prompt factorization/readout and supervision, not the backbone
parameter layout or a purported private Jev head.

- **Score:** each level is evaluated without sibling descriptions or level indices.
  Its yes probability is normalized with the other level probabilities afterward.
- **Choice:** each named candidate sees the complete alternatives, including relational
  options. Binary readouts support all 255 schema options without alphabet truncation.
- **Noul:** retains its primitive-aware binary prompt.
- **Training:** canonical soft/one-hot targets become candidate Bernoulli targets.
  Each view has weight `1/K`, so a question does not get K times the total weight.
  The optimizer still steps over view batches: compute/exposure differs from the
  letter model and must be reported, not mistaken for a matched-compute ablation.
- **Memory:** shared execution microbatches at four views by default; candidate
  evaluation costs multiple forwards. A 255-option test is not a constant-cost claim.

Verified: descriptor isolation/reordering fixtures, target mass/weights, adapter
compatibility, actual 255-option 135M inference, and a candidate LoRA save/reload.
A quality comparison follows separately; passing these tests does not establish
that normalized binary judgments outperform joint scoring.

## 3. Calibration, confidence, and frozen diagnostic data

The temperature fitter accepts only complete predictions whose IDs, labels, and
hash match the declared calibration partition. It checks split disjointness and
adapter training/development hashes, and refuses already scaled predictions.
Artifacts bind weights/tokenizer hashes, renderer/readout, precision, and execution
policy. `--temperature FILE` fails closed on a configuration mismatch. Per-primitive
scalar temperatures minimize outcome NLL; argmax is unchanged and out-of-domain
calibration is not guaranteed.

Structured-v1 confidence now defaults to the public TypeSafe adapter formulas pinned
to commit `fb52b1030b7fc1f4f1cf39910afa5da54f9835e3`: scaled peak probability for Choice,
modal-distance concentration for Score. `--confidence entropy-v0` retains the old
statistic (also the legacy renderer default). These are not probabilities of
correctness and have not been verified against a live Jev version.

Frozen **before evaluation**:

- `tests/fixtures/heldout_rubrics.json`: 42 synthetic cases (14 conservative groups) across explicit cancellation,
  speech-act routing, reproduction evidence, and operational impact. Includes new
  Score rubrics. These hand-authored gold labels need independent review; this is a
  small diagnostic, not a representative generalization benchmark. Never train,
  calibrate, or tune on these cases.
- `scripts/prepare_teacher_experiment.py`: exact legacy-prompt round trips recover
  the cached teacher's original truncated input and rubric. The four regimes share
  records, grouped splits, and targets aligned to the same keys: gold, teacher hard,
  teacher soft, and 50/50 gold/soft. All start from base weights. Unknown teacher
  versions and original source IDs remain explicitly unknown. No paid API calls.

Prepared data are under `data/experiments/heldout-rubrics/` and
`data/experiments/teacher-matched/`. The latter is a controlled experiment with a
historical unversioned teacher, not fidelity evidence about current Jev. A pinned
live-teacher collection still requires a chosen model version and approved budget.

## 4. First matched backbone training results

Frozen code `520038f`; all 8,769 training and 1,128 development questions passed
both tokenizers' 768-token admission check (no exclusions). Each backbone trained
one epoch, batch 8, learning rate 5e-5, seed 42, rank-16 attention LoRA. Exactly
8,769 example presentations and 1,097 optimizer steps each. Native independent
inference, no contextual correction or fitted temperature.

| Model | Development accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| SmolLM2-135M untuned | 29.9% | 1.402 | 0.735 | 0.169 |
| SmolLM2-135M LoRA | 67.0% | 0.747 | 0.425 | 0.040 |
| Qwen3-0.6B untuned | 44.2% | 1.229 | 0.647 | 0.094 |
| Qwen3-0.6B LoRA | 39.6% | 1.231 | 0.651 | 0.058 |

Qwen regressed: lower ECE does not mean a better decision model. This protocol
compares particular backbones/settings, not their best achievable performance.
The separately predeclared 1e-5 control used identical Qwen examples, epoch, batch
size, and seed and recovered substantially better performance (below). Rubric
results were not used to choose hyperparameters.

Artifacts: `data/runs/baselines-v1/`, adapters
`adapters/smollm2-135m-structured-v1/` and `adapters/qwen3-0.6b-structured-v1/`.
These are in-family development point estimates on 800 connected groups, not a
final test result. The reserved Kev test remains unevaluated. Training wall times
include incidental development activity and are not isolated throughput benchmarks.

`scripts/compare_runs.py` verifies paired IDs/labels and reports group-bootstrap
intervals for outcome differences. Optional teacher targets add JS divergence,
argmax agreement, and Noul MAE, separate from outcome correctness.

## 5. Learning-rate control, calibration, and unseen rubrics

Follow-up code was frozen at `b7d5932`. Qwen at **1e-5**, with the same 8,769
presentations / 1,097 steps, reached **81.3% development accuracy**, NLL **0.460**,
Brier **0.256**, and ECE **0.033**. It was selected by minimum development NLL among
untuned, 5e-5, and 1e-5 variants, before rubric evaluation. The result does not show
that Qwen was intrinsically worse in the first comparison: this fine-tune was very
learning-rate-sensitive. SmolLM and selected Qwen now have different learning rates;
this is a practical selected-model comparison, not an isolated architecture effect.

Temperatures were fitted on **1,103 calibration questions**, never on development
or rubric cases. On the 1,128 development questions:

| Model | Raw NLL → scaled | Raw Brier → scaled | Raw ECE → scaled | Accuracy |
|---|---:|---:|---:|---:|
| SmolLM2-135M | .747 → .743 | .425 → .424 | .040 → .040 | 67.0% |
| Qwen3-0.6B, LR 1e-5 | .460 → .448 | .256 → .252 | .033 → .022 | 81.3% |

Qwen's paired NLL change was −0.0121, with group-bootstrap 95% interval
[−0.0212, −0.0038] (2,000 draws of whole groups). Argmax does not change.
Qwen Score MAE was .587 raw / .592 scaled: optimizing NLL does not optimize every
metric. These development intervals are descriptive after model selection, not a
substitute for a frozen final test.

On the 42-case, four-family synthetic rubric holdout:

| Model | Accuracy | Raw NLL | Scaled NLL | Raw ECE |
|---|---:|---:|---:|---:|
| SmolLM2-135M | 35.7% | 1.000 | .996 | .110 |
| Qwen3-0.6B, LR 1e-5 | 76.2% | .538 | .554 | .216 |

Qwen's primitive accuracies were 10/12 Choice, 11/12 Noul, and 11/18 Score.
Its in-family temperature **worsened** rubric NLL by .0153 (95% grouped interval
[−.0046, .0332]); distribution-shift calibration remains unsolved. There are only
14 groups, labels need independent review, and repeated inspection now makes this
a regression diagnostic rather than an indefinitely fresh benchmark. Do not tune
against it and continue calling it unseen.

Artifacts: `data/runs/followup-v1/`, including selection policy, predictions,
temperature files, manifests, logs, and paired comparisons. Selected adapter:
`adapters/qwen3-0.6b-structured-v1-lr1e-5/`. The main Kev test remains untouched.

## 6. Readout pilot and wide Choice

Both SmolLM pilots started from the same base and saw the same **450 source
questions** (150 per primitive), one epoch, batch 8, LR 5e-5, seed 42. Letters used
450 views / 57 steps; candidates used **1,951 views / 244 steps**. Candidate losses
were weighted 1/K. This is matched-source supervision, **not matched compute or
optimizer exposure**, and neither pilot replaces the full-data baseline.

| Development metric | Joint letters | Candidate binary |
|---|---:|---:|
| Accuracy | 36.5% | 38.9% |
| NLL | 1.294 | 1.223 |
| Brier | .685 | .648 |
| Score MAE | 1.374 | 1.194 |

Paired candidate-minus-letter NLL: −.0712 [−.0907, −.0517]. Accuracy difference:
+2.39 points [−.63, +5.49]. Candidate rubric accuracy fell to **28.6%**, versus
38.1% for the letter pilot; this does not establish better generalization.

A separate **30-case Banking77 holdout**, excluded from model training and checked
against train/dev/calibration overlap, exercised 77-way Choice with the candidate
adapter, FP32/shared execution. Accuracy was **0/30**, NLL **4.341** (uniform:
log(77) ≈ 4.344), ECE .0157. The low ECE is not competence: predictions were nearly
uniform. Actual 255-option inference also passes a contract test, not a quality
benchmark. Keep candidate readout experimental; wider API capacity is not evidence
of useful wide-option decisions.

## 7. Matched cached-teacher experiment

Four SmolLM models each started from base weights and trained on the same **1,280
questions**, one epoch / 160 steps, LR 5e-5, batch 8, seed 42. Development had 160
examples; calibration 160; the matched outcome test had **400 groups/examples**.
No Kev-trained adapter initialized these runs. Inputs preserve historical truncation
and reconstruct the exact cached teacher rubric; no new teacher requests were made.
Original dataset gold labels are retained even where historical truncation could
remove decisive evidence, which limits the interpretation of outcome correctness.

| Training target | Test accuracy | NLL | Brier | ECE | Teacher JS (nats) | Teacher argmax agreement |
|---|---:|---:|---:|---:|---:|---:|
| Gold | 36.0% | 1.484 | .711 | .082 | .312 | 34.5% |
| Teacher hard | 32.0% | 1.508 | .719 | .091 | .316 | 32.8% |
| Teacher soft | 37.8% | 1.405 | .676 | .108 | .287 | 38.8% |
| 50/50 gold-soft | 37.8% | 1.472 | .702 | .081 | .308 | 37.5% |

Soft versus gold NLL: −.0789, grouped 95% interval [−.1178, −.0372]. Its accuracy
change interval [−2.0, +6.0 percentage points] includes zero, and ECE worsened.
This is modest evidence for these cached soft targets under this small protocol,
not proof of universal calibration benefit or strong teacher imitation. All 400
teacher test distributions are **unversioned**. A live, pinned teacher comparison
still needs an approved provider/version/budget.

## Remaining evidence gates

- Independently review and expand rubric labels; establish a new frozen workflow
  holdout with error-cost/risk–coverage criteria before claiming safe automation.
- Repeat promising comparisons across seeds and training budgets. Test larger
  candidate pilots without treating extra view updates as matched compute.
- Freeze a final model/configuration before touching the reserved Kev test.
- Keep shared execution opt-in; validate precision, shapes, adapters, and calibration
  together. No current result establishes private Jev architecture or full parity.
