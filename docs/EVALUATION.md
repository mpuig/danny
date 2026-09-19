# How to Evaluate

## Metrics (calibration first)

Defined in `src/jev/metrics.py`:

- **ECE** (expected calibration error, 15 bins): |top-1 confidence − empirical accuracy|,
  bin-weighted. The headline metric.
- **Brier**: mean squared error between distribution and one-hot label. Proper scoring rule.
- **NLL**: −log p(true label). Proper scoring rule, punishes confident errors hardest.
- **accuracy**, **mean_top1_prob** (their gap is a quick overconfidence read).
- **score_mae** for score tasks: |weighted-mean score − true level|.

Accuracy alone is *not* success — a model can be accurate and dangerously overconfident.

## Protocol

- **Held-out tasks** (never in training): `sst2` (noul), `tweet_emotion` (choice).
  These measure generalization to unseen questions — the product claim.
- In-domain test splits of training tasks measure fit.
- Default n=200, canonical question phrasing (index 0), `--calibrate` on unless the
  point is to measure raw behavior.
- Compare like with like: same n, same split, same calibration flag.

## Commands

```bash
# calibration eval on any task, any checkpoint:
uv run python scripts/eval_baseline.py --model mlx-community/SmolLM3-3B-Base-bf16 \
    --task sst2 --n 200 --calibrate [--adapter adapters/smollm3-3b]

# option-order robustness (choice tasks):
uv run python scripts/permutation_test.py --model ... --task ag_news --n 100 \
    [--adapter ...] [--calibrate]
# reports argmax_flip_rate and mean_tv_distance; Jev-quality target: low single digits %

# one Jev-shaped request, all three primitives:
uv run python scripts/demo_request.py --model ...

# drop-in API compatibility via the official TypeScript SDK:
uv run python scripts/serve.py --model ... --adapter ... --calibrate --port 8399 &
cd tests/ts && npm install && node test.ts            # against our server
cd tests/ts && node test.ts --with-jev                # also against the real Jev API
```

## Results so far (SmolLM3-3B-Base unless noted; n=200)

### Held-out tasks (generalization) — three-way comparison

All columns + contextual calibration, n=200. One-hot LoRA: 8,000 gold examples.
Distill LoRA: 2,000 Jev soft targets (¼ the data).

| Task | untuned | one-hot LoRA | **Jev-distilled LoRA** |
|---|---|---|---|
| sst2 acc | 0.695¹ | 0.925 | 0.920 |
| sst2 ECE | 0.088¹ | 0.068 | **0.061** |
| sst2 NLL | 0.59¹ | 0.228 | **0.224** |
| tweet_emotion acc | 0.82 | 0.82 | **0.825** |
| tweet_emotion ECE | 0.205 | 0.134 | **0.065** |
| tweet_emotion Brier | 0.324 | 0.292 | **0.256** |

¹ with the lettered noul template; the original bare yes/no template scored 0.535 acc /
0.297 ECE. Key result: the distilled model **halves ECE vs one-hot on the harder held-out
task from a quarter of the training data** — Jev's soft targets carry calibration that
one-hot labels cannot.

### Agreement with the real Jev (held-out states, n=100, + calib)

| Task | metric | untuned | one-hot | distill | Jev itself |
|---|---|---|---|---|---|
| sst2 | KL(ours ‖ Jev) | 0.303 | **0.056** | 0.072 | — |
| sst2 | argmax agreement | 0.73 | 0.91 | 0.91 | — |
| sst2 | accuracy | 0.69 | 0.91 | 0.91 | 0.94 |
| tweet_emotion | KL(ours ‖ Jev) | 4.23 | 3.04 | **1.69** | — |
| tweet_emotion | argmax agreement | 0.88 | 0.89 | 0.88 | — |
| tweet_emotion | accuracy | 0.82 | 0.79 | **0.84** | 0.86 |

(The large tweet_emotion KLs are dominated by Jev's near-zero tail probabilities; the
relative ordering is the signal.) Within 2–3 accuracy points of Jev on both held-out
tasks with a local 3B model.

### Permutation sensitivity (ag_news, n=100, + calib; adapters trained BEFORE
shuffling augmentation — these are the "before" baselines)

| | untuned | one-hot | distill |
|---|---|---|---|
| argmax flip rate | 0.11 | 0.09 | **0.06** |
| mean TV distance | 0.147 | 0.088 | **0.070** |

Reference: kev reports 7.4% flips *with* augmentation. Our next training cycle uses the
shuffled data; rerun this test after it.

### In-domain (ag_news, untuned)

raw: 0.75 acc / 0.117 ECE / 0.40 Brier → +calibration: 0.80 / 0.107 / 0.31.

### Reference points

- 135M sst2: 0.58 acc untuned (n=50); 0.515 / 0.103 ECE with LoRA+calib (n=200) — tiny
  backbone generalizes weakly; used for pipeline smoke tests only.
- Real Jev (from its API on our demo ticket): refund noul 0.99, choice confidence 1.00.
  Jev on MMLU (external probe): ECE 0.031. Those are the targets.
- Distillation set analysis: Jev argmax = gold on 88.4% of 2000 pulled examples; 45% of
  its targets genuinely soft.

## Pending evals

1. Rerun permutation test after the shuffled-data retrain (expect flip rate to drop).
2. Post-temperature-scaling ECE per primitive (use kev's decision-v1/v2 calibration split).
3. External benchmark: `data/kev_test.jsonl` (kev's frozen test split, 1,048 questions).
4. Combined-mix training (ours 8k shuffled + kev 11k + Jev 2k soft) → full battery.
