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

## Next experiments

- Run both backbones on the same jointly token-admitted training/development rows,
  one epoch with the same optimizer settings and exact example exposures.
- Compare a candidate-level binary readout with the joint letter baseline; retain
  the original Score model as a control and test wide Choice separately.
- Fit temperature only on calibration, evaluate held-out rubrics separately, and
  distinguish cached unversioned teacher experiments from pinned live distillation.
- Measure the bounded service under concurrency, queue saturation, length limits,
  and cold/warm calibration caches. Keep native shared mode opt-in.
