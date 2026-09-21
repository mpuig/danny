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
[−0.0212, −0.0038] (2,000 draws of whole groups). This is an **NLL interval**, not
an interval for ECE; the comparison report does not bootstrap ECE. Argmax does not change.
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

## 8. Bounded HTTP serving measurements

Apple **M4 Max, 36 GiB unified memory**, without concurrent model training. Full
sweep code: `2fcffe0`; lifecycle/configuration follow-up: `64313c7`. These runs use
the trained adapters (unmerged LoRA), not the bare backbones from section 1.
Each backbone/policy ran 16 cases: 3/12 mixed questions, 1/16 repeated state
messages, 3/26 Choice options, and concurrency 1/4, with 20 warm requests per case.
Requests open new connections. No fitted temperature or contextual correction was
used in the main sweep. The synthetic fixtures are not a quality benchmark.

Twelve questions, longer state, **three options**, concurrency one:

| Adapter / execution | HTTP p50 | HTTP p95 | Successful requests/s | Cumulative peak MLX active memory, full sweep |
|---|---:|---:|---:|---:|
| SmolLM native independent | 258 ms | 261 ms | 3.87 | .75 GB |
| SmolLM FP32 shared | 121 ms | 123 ms | 8.27 | 1.35 GB |
| Qwen native independent | 723 ms | 749 ms | 1.38 | 2.11 GB |
| Qwen FP32 shared | 425 ms | 454 ms | 2.33 | 4.54 GB |

These are small-sample quantiles on one working machine, not production p95
certification. Peak allocation includes startup and preceding cases; it is not the
incremental cost of this row. Reported process peak RSS was .44–.45 GB for SmolLM
and about 1.55 GB for Qwen. RSS and MLX counters measure different/overlapping
resources, not complete system memory; do not add them. GB here means 10^9 bytes.

Increasing Choice width to 26 largely removed the shared-mode latency advantage:
SmolLM p50 was 305 ms native / 300 ms FP32 shared; Qwen 1,166 / 1,139 ms. We have
not isolated padding, precision, and LoRA overhead as causes. Shared mode currently
recomputes a common prefix per microbatch and physically copies its KV. It is not
constant-cost parallelism or paged attention.

Concurrency four introduced queue delay rather than automatic cross-request
batching: in the three-option Qwen fixture, p50 rose to 3,222 ms native / 1,725 ms
FP32 shared. A dedicated saturation test, queue capacity one and eight clients,
returned **2 successful responses and 46 HTTP 503s** across 48 submissions, with
no inference failure. A fresh post-fix rerun reproduced those admission counts.
Oversized-body, question-count, and token-budget probes returned 413, 422, and 422.

For a separate SmolLM contextual-correction case (12 questions, longer state),
the first request took 580 ms versus warm p50 262 ms; input scoring tokens dropped
from 10,184 to 6,074 after caching priors. New case-specific rubrics forced cold
prior misses. Python cache sizes remained within their configured entry ceilings.
The MLX free-buffer threshold is reclaimed on the next allocation, not an exact
instantaneous cache-memory ceiling.

HTTP single-versus-multi drift for the first Choice in each case was **zero** in
native independent mode; maxima were **3.94e-6 SmolLM / 6.17e-6 Qwen** in FP32 shared
mode, below the predeclared 2e-5 gate. This compares each policy to itself, not native
to FP32. Calibration artifacts must match precision/execution/microbatch settings;
the native temperature files cannot be used to claim calibrated FP32 performance.

The first benchmark driver exposed an operational defect: a background shell passed
ignored SIGINT to the server, so cleanup waited 60 seconds and then killed it. That
delay was outside measured HTTP samples. Explicit SIGINT/SIGTERM handlers now fix
it; benchmark success also requires a clean exit. Fresh runs across both models and
policies, plus saturation, passed with **no forced shutdown**. A real-model lifecycle
test reproduces inherited SIGINT masking to guard against regression.

The official TypeScript SDK smoke test also passed against selected Qwen with its
fitted temperature. Across 24 development questions (eight per primitive), current
HTTP probabilities matched the frozen calibrated evaluation **exactly**. These are
contract/numerical checks, not additional independent model-quality evidence.

Reports: `data/runs/serving-v1/` and `data/runs/serving-validation-v1/`. Reproduce:

```bash
uv run python scripts/benchmark_service.py --model Qwen/Qwen3-0.6B \
  --adapter adapters/qwen3-0.6b-structured-v1-lr1e-5 \
  --out-dir data/runs/qwen-http-new
# Repeat into another fresh directory with --precision float32 --execution-mode shared.
# Use --calibrate for cold/warm prior measurements; --queue-capacity 1 --concurrency 8
# for an explicit saturation run. See --help for all axes and sample counts.
```

See [Serving](SERVING.md) for deadlines, operating limits, and security caveats.

## Remaining evidence gates

- Independently review and expand rubric labels; establish a new frozen workflow
  holdout with error-cost/risk–coverage criteria before claiming safe automation.
- Repeat promising comparisons across seeds and training budgets. Test larger
  candidate pilots without treating extra view updates as matched compute.
- Freeze a final model/configuration before touching the reserved Kev test.
- Keep shared execution opt-in; validate precision, shapes, adapters, and calibration
  together. No current result establishes private Jev architecture or full parity.

## 9. Reserved test, spent once (2026-09-21)

The frozen configuration (decision 25) was evaluated on the reserved 1,048-question
Kev test partition, raw and with its pinned per-primitive temperatures — one model,
one look; the partition is now spent for unbiased evaluation.

| Variant | Accuracy | NLL | Brier | ECE | Score MAE |
|---|---:|---:|---:|---:|---:|
| Raw | 77.6% | .523 | .301 | .056 | .614 |
| Temperature-scaled | 77.6% | .516 | .298 | .049 | .614 |

Per primitive (scaled): Choice 80.6% / ECE .051 (n=464), Noul 84.7% / ECE .045
(n=424), Score 50.0% / ECE .130 (n=160). Relative to development (82.0%, ECE .020)
this is a ~4-point accuracy generalization gap and roughly doubled calibration
error — a realistic in-family transfer estimate, not evidence about unfamiliar
rubrics or workflows. The fitted temperatures helped on this shift (ECE .056 -> .049),
unlike on the rubric diagnostic. Score remains the weakest primitive. Reports:
data/runs/final-test/. Future models cannot be selected on these numbers without
a fresh reserved partition.

## 10. Risk-coverage on the spent test (analysis only, no new model looks)

scripts/risk_coverage.py over the final-test scaled predictions (top-1
probability as the automation signal):

| Threshold | Coverage | Risk among automated |
|---:|---:|---:|
| 0.80 | 64.8% | 8.4% |
| 0.90 | 51.3% | 4.8% |
| 0.95 | 40.3% | 2.8% |
| 0.99 | 12.3% | 0.8% |

Per primitive at 0.95: Choice 39% coverage / 2.7% risk, Noul 56% / 2.9%,
Score barely automates (12% coverage at 0.80, 15% risk). Thresholds are
in-distribution numbers and do not transfer to shifted workloads; confidence
is top-1 probability, not a probability of correctness.

## 11. Scale-up: MiniCPM5-2B-Base under the frozen recipe (2026-09-21)

Identical corpus, objective, and seed to the selected 0.6B configuration; batch 4
(hardware-necessitated, as with earlier >2B runs). Final val loss 0.3773 vs the
0.6B's 0.4990. Development comparison vs the frozen adapter (grouped bootstrap):
accuracy +4.8 points [+2.7, +7.0], NLL -0.102 [-0.137, -0.068], Brier significant;
raw dev ECE 0.018 - no calibration tax from the benchmark-tuned base. Held-out
synthetic: NLL -0.163 [-0.240, -0.090]; accuracy +4.7 points with an interval
touching zero [-0.004, +0.098].

The score slice moved for the first time: development score accuracy 0.562 ->
0.673, MAE 0.564 -> 0.435. Confident errors halved (2.2% at t>=0.9 vs 4.8%).
The rubric canary reads 0.905 acc / NLL 0.252; accuracy there is seed noise by
decision 24, but the NLL sits far outside the 0.54-0.66 band every 0.6B variant
occupied. Backbone selection is a pending decision, not automatic: serving
latency at 2.5B is unmeasured, and the reserved-test equivalent for this model
does not exist (the old partition is spent). Adapter:
adapters/minicpm5-2b-structured-v1-synthfiltered-rps; reports data/runs/minicpm-v1/.

