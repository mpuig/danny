---
license: mit
base_model: Qwen/Qwen3-0.6B
tags: [decision-model, calibration, system-one, mlx, lora]
---

# danny-qwen3-0.6b — volume-tier System One adapter

LoRA adapter for **Qwen3-0.6B** (base, Apache-2.0) turning it into a
**System One decision model**: typed questions (choice / score / noul) over a
JSON state, answered with calibrated probability distributions read directly
from restricted logits in one forward pass — no text generation. Part of
[danny](https://github.com/mpuig/danny), an open learning project rebuilding the
behavior of Typesafe's Jev. **Not affiliated with or endorsed by TypeSafe.**

## Files

- `adapters.safetensors` + `adapter_config.json` — rank-16 attention LoRA (MLX format)
- `temperature.json` — per-primitive temperatures (choice 1.019 / noul 1.156 /
  score 1.056), fitted on a dedicated calibration split and **bound to these
  exact weights**: the serving engine refuses them against anything else.

## Training

Frozen recipe (decision 25 in the repo's audited decision log): 8.7k questions
recast from public classification datasets plus ~1.6k synthetic scenarios whose
probability targets were produced by **pinned jev-1.13.0** (teacher
distillation; disclosed deliberately) and filtered for teacher overconfidence.
Readout-matched cross-entropy on target distributions + Ranked Probability
Score on ordinal questions; LR 1e-5, batch 8, one epoch, seed 42, Apple MLX.

## Evaluation (full protocol and CIs in the repo)

| Split | Accuracy | ECE | Notes |
|---|---:|---:|---|
| Development (n=1,128) | 82.0% | 0.020 | used for steering; optimistic |
| **Reserved in-family test (n=1,048, spent once)** | **77.6%** | **0.049** | unbiased; choice 80.6 / noul 84.7 / score 50.0 |
| Fresh out-of-family test (n=1,048, spent once) | 66.5% | 0.076 | new task families; see shift warning |

**Shift warning (measured):** confidence thresholds are valid in-family only.
On out-of-family workloads confident-error rates reached ~19% at t>=0.9 versus
~2% in-family. The serving runtime supports per-workload temperature fitting
from ~100 labeled examples (`POST /v1/calibrations`), which repaired scalar
miscalibration to 3-4% confident errors in the repo's resampled experiment.

## Use

```bash
git clone https://github.com/mpuig/danny && cd danny && uv sync
uv run python scripts/serve.py --model Qwen/Qwen3-0.6B \
  --adapter <this-repo> --temperature <this-repo>/temperature.json
```

Apple Silicon required (MLX). Score is this tier's weak primitive (50% on the
unbiased test); use the quality tier for score-heavy workloads. This is a
research prototype, not a certified production service.
