# Evaluation

Evaluate three separate properties: **decision quality against labels/outcomes**,
**fidelity to Jev**, and **runtime/API correctness**. None establishes the other two.
The product goal is useful automation at an acceptable error cost, not ECE alone.

## Implemented metrics

`src/jev/metrics.py` reports:

- **NLL:** negative log probability of the true label; penalizes confident errors.
- **Brier:** mean sum of squared errors against one-hot labels, not divided by
  class count. For binary tasks this is twice the scalar binary Brier convention.
- **Accuracy** and **mean top-1 probability**.
- **ECE:** 15 fixed-width bins comparing top-1 probability with empirical accuracy.
  This uses `max(probabilities)`, **not** the API's entropy-derived `confidence`.
- **Score MAE** in the evaluator: error between the weighted-mean level index and
  the gold level. It is not a complete measure of ordinal distribution quality.

NLL/Brier are proper scoring rules and should lead the probabilistic comparison.
Retain accuracy and ECE as complementary diagnostics. ECE at n=200 is sensitive
to binning and sampling; small differences need uncertainty intervals. Low ECE
can coexist with a model too uninformative to automate useful work.

## What the current splits establish

| Evaluation | What it measures |
|---|---|
| Test splits of ag_news, dbpedia, imdb, yelp_stars | New examples within training task families |
| SST-2 validation split | Cross-dataset movie-sentiment transfer: canonical question and criteria match training IMDB |
| Tweet-emotion test split | A held-out question relative to the four-task recast mix, still sentiment-adjacent |
| Kev test, after training on Kev train | In-family performance, including grouped robustness variants |
| Unfamiliar rubric/task-family suite | **Not implemented**; needed for the central generalization claim |

There is no held-out Score family in the recast suite. Public dataset names and
splits do not rule out pretraining exposure. Historical tuning against SST-2 also
means it should not be treated as a pristine final test for new research choices.

Check [Data](DATA.md) before training: the current validation set overlaps teacher
and Kev training. Different seeds do not ensure separation. For new experiments,
reserve test groups first and keep calibration separate from model-selection data.
Kev test questions/variants share documents; confidence intervals must respect
those groups rather than treating all 1,048 converted questions as independent.

## Commands that exist today

The recast evaluator defaults to **n=100**, canonical phrasing, seed 42, and
contextual correction **off**. Historical tables below use the stated n and flag.
Run matched raw and corrected evaluations rather than enabling correction silently.

```bash
# Raw baseline on a built-in recast task.
uv run python scripts/eval_baseline.py --model mlx-community/SmolLM3-3B-Base-bf16 \
    --task sst2 --n 200

# Trained variant; requires an existing matching adapter.
uv run python scripts/eval_baseline.py --model mlx-community/SmolLM3-3B-Base-bf16 \
    --task sst2 --n 200 --adapter adapters/smollm3-3b --calibrate

# Choice option-order sensitivity: one random nonidentity permutation per example.
uv run python scripts/permutation_test.py --model mlx-community/SmolLM3-3B-Base-bf16 \
    --task ag_news --n 100 --adapter adapters/smollm3-3b --calibrate

# Teacher fidelity. Calls Jev only if the local task/n cache is absent.
uv run python scripts/agreement_vs_jev.py --model mlx-community/SmolLM3-3B-Base-bf16 \
    --task tweet_emotion --n 100 --adapter adapters/smollm3-3b-distill --calibrate
```

The agreement script reports epsilon-smoothed `KL(ours || Jev)`, argmax agreement,
and both models' accuracy. It does **not** currently report Noul MAE despite its
module description. Rounded teacher zeros make KL sensitive to the `1e-9` epsilon;
normalize targets and add total variation or Jensen–Shannon distance before using
KL as the only fidelity metric.

Teacher caches are named only by task and n. An existing incomplete cache is
accepted without checking count, rubric, dataset revision, or teacher version.
Audit the cache before comparing runs; pin and record versions in the next harness.

There is **no external-JSONL evaluation command yet**. `eval_baseline.py` accepts
only the six registered recast tasks. Passing converted Kev data as the trainer's
`--val` gives a loss, not the full evaluation battery, and using the test file there
would contaminate model selection.

### SDK smoke test

```bash
# Terminal 1:
uv run python scripts/serve.py --model HuggingFaceTB/SmolLM2-135M --port 8399
# Terminal 2, with a Node version supporting TypeScript type stripping:
cd tests/ts && npm install && node test.ts
# Optional live comparison, requires TYPESAFE_API_KEY:
node test.ts --with-jev
```

This checks one basic request with three primitives. It does not certify all SDKs,
invalid inputs, structured criteria, confidence formulas, context limits, question
independence, or numerical parity. See [Architecture](ARCHITECTURE.md).

## Historical results — retained, not rerun in this review

These tables were recorded before the documentation audit. Local adapter files
exist, but complete run manifests and per-example student predictions are absent.
The tables are exploratory results, not independently reproduced measurements.

### Recast comparisons (SmolLM3-3B-Base, n=200)

All columns use contextual correction. Gold corpus: 8,000 rows; teacher corpus:
2,000 rows. Corpus size is not optimizer exposure count. Data composition, target
source, and training budget were not controlled to isolate soft-target effects.
The existing rendered gold data was subsequently augmented; historical runs are
recorded as predating shuffled-data retraining.

| Task / metric | Untuned | Gold-label LoRA | Jev-distilled LoRA |
|---|---:|---:|---:|
| SST-2 accuracy | 0.695 | 0.925 | 0.920 |
| SST-2 ECE | 0.088 | 0.068 | 0.061 |
| SST-2 NLL | 0.59 | 0.228 | 0.224 |
| Tweet-emotion accuracy | 0.820 | 0.820 | 0.825 |
| Tweet-emotion ECE | 0.205 | 0.134 | 0.065 |
| Tweet-emotion Brier | 0.324 | 0.292 | 0.256 |

The distilled adapter's lower tweet-emotion ECE/Brier is promising, but does not
prove that it learned transferable calibration from soft targets. Gold-label CE
can also learn calibrated distributions. Use matched examples, augmentation,
training budgets, multiple seeds, and paired intervals for that claim.

An earlier bare yes/no Noul template reported SST-2 accuracy 0.535 / ECE 0.297.
It differs from the standard lettered template, so it is not a clean raw-versus-
corrected ablation of the current engine.

### Agreement with Jev (n=100, contextual correction)

| Task | Metric | Untuned | Gold-label | Distilled | Jev |
|---|---|---:|---:|---:|---:|
| SST-2 | KL(ours ∥ Jev) | 0.303 | 0.056 | 0.072 | — |
| SST-2 | Argmax agreement | 0.73 | 0.91 | 0.91 | — |
| SST-2 | Accuracy | 0.69 | 0.91 | 0.91 | 0.94 |
| Tweet emotion | KL(ours ∥ Jev) | 4.23 | 3.04 | 1.69 | — |
| Tweet emotion | Argmax agreement | 0.88 | 0.89 | 0.88 | — |
| Tweet emotion | Accuracy | 0.82 | 0.79 | 0.84 | 0.86 |

The teacher accuracies can be recomputed from the local 100-row caches; the student
metrics require inference reruns. Large KL values reflect teacher near-zero tails
as well as disagreement. These small samples do not establish general parity with
Jev or equivalence of downstream decision policies.

### Choice permutation sensitivity (ag_news, n=100, contextual correction)

| Metric | Untuned | Gold-label | Distilled |
|---|---:|---:|---:|
| Argmax flip rate | 0.11 | 0.09 | 0.06 |
| Mean total variation | 0.147 | 0.088 | 0.070 |

Recorded as **pre-shuffling** adapter baselines. Re-run after retraining, using the
same states/permutations. External projects' flip rates use different protocols
and are not direct baselines. Jev itself has documented order sensitivity; zero
sensitivity is a robustness goal, not a promise of behavioral replication.

### Other historical observations

- Untuned ag_news: raw 0.75 accuracy / 0.117 ECE / 0.40 Brier;
  corrected 0.80 / 0.107 / 0.31.
- 135M SST-2: untuned 0.58 accuracy at n=50; LoRA plus correction 0.515 accuracy /
  0.103 ECE at n=200. Different n prevents a direct comparison.
- An external Jev probe reports MMLU ECE around 0.031 with a different dataset,
  binning, and model. It is context, not this project's acceptance threshold.
- Current teacher-target integrity and gold agreement are in [Data](DATA.md).

## Required next evaluation work (not implemented)

1. **Data/provenance:** frozen group-disjoint train/dev/calibration/test manifests;
   external JSONL support; per-example predictions and complete run configuration.
2. **Probability quality:** paired bootstrap intervals, reliability plots,
   classwise/primitive-level analysis, and raw/corrected/temperature-scaled variants.
3. **Rubric transfer:** same state under different questions and changed criteria;
   unfamiliar task families; missing evidence; structured paths; contrastive pairs.
4. **Primitive behavior:** preserve Noul identity; compare joint versus per-level
   Score models; test applicability gates versus relative Choice selection.
5. **Workflow usefulness:** error cost, risk–coverage, threshold-local reliability,
   and final actions. Weighted composite scores are not automatically calibrated
   event probabilities, and marginal probabilities need not be independent.
6. **Engine correctness:** single versus batched results, mixed primitives, padding,
   question addition/removal/reordering, ID renaming, finite distributions, tokenizer
   labels, cache fallback, and adapter reload parity. Use declared tolerances.
7. **Serving:** target-hardware latency percentiles, throughput, and peak memory
   across state length, question count, option count, concurrency, and cold/warm
   correction caches. Compare serial and concurrent baselines fairly.

Select thresholds and model variants on development/calibration data, then freeze
them before final test evaluation. Arithmetic and broad reasoning tests diagnose
scope limits; they are not the main acceptance test for a System One model.
