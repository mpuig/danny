---
license: mit
base_model: openbmb/MiniCPM5-2B-Base
tags: [decision-model, calibration, system-one, mlx, quantized]
---

# system-one-minicpm5-2b-q8 — quality-tier System One model (8-bit)

**MiniCPM5-2B-Base** (Apache-2.0) fine-tuned with the system-one frozen recipe,
LoRA-fused and quantized to 8 bits with MLX (8.501 bits/weight, 2.5 GB). A
**System One decision model**: typed questions (choice / score / noul) over a
JSON state, answered with calibrated probabilities from one forward pass — no
generation. Part of [system-one](https://github.com/mpuig/system-one), an open learning
project rebuilding the behavior of Typesafe's Jev. **Not affiliated with or
endorsed by TypeSafe.**

## Files

- fused, 8-bit quantized model (MLX format; load with `--model <dir>`, no adapter)
- `temperature.json` — per-primitive temperatures (choice 0.961 / noul 1.073 /
  score 1.387), fitted under this exact quantized identity and **bound to these
  weights** — the serving engine fails closed on any mismatch.
- the unquantized LoRA adapter is published alongside for retraining/re-fusing.

## Training and quantization

Same frozen recipe as the volume tier (public recasts + synthetic scenarios
teacher-scored by **pinned jev-1.13.0** — disclosed deliberately — with
overconfidence filtering; readout CE + RPS loss, LR 1e-5, seed 42). Quantization
was gate-checked: zero argmax flips across the drift battery, dev quality within
noise of BF16, median 1.61x serving speedup. 4-bit was measured and rejected
(quality loss, no speed gain over 8-bit on Apple Silicon).

## Evaluation (full protocol and CIs in the repo)

| Split | Accuracy | ECE | Notes |
|---|---:|---:|---|
| Development (n=1,128, calibrated) | 86.6% | 0.019 | used for steering; optimistic |
| **Fresh out-of-family test (n=1,048, spent once)** | **73.3%** | **0.080** | **+6.8 pts over the 0.6B tier, CI [+4.5, +9.2]**; NLL/Brier also significantly better |

Per primitive (dev): choice 94.2 / noul 90.0 / score 65.9. Confident errors at
t>=0.95: 1.2% in-family.

**Shift warning (measured):** in-family thresholds do not transfer — out-of-family
confident errors reached ~14% at t>=0.9. Use the runtime's per-workload
temperature fitting (`POST /v1/calibrations`, ~100 labeled examples) before
trusting thresholds on a new workload; it repaired scalar miscalibration to
3-4% confident errors in the repo's resampled experiment.

## Use

```bash
git clone https://github.com/mpuig/system-one && cd system-one && uv sync
hf download mpuig/system-one-minicpm5-2b-q8 --local-dir system-one-minicpm5-2b-q8
uv run python scripts/serve.py --model system-one-minicpm5-2b-q8 \
  --temperature system-one-minicpm5-2b-q8/temperature.json
```

Light 3-question request: ~195 ms p50 on an M4 Max. Apple Silicon required
(MLX). Research prototype, not a certified production service; the honest
failure record lives in the repo's decision log and experiments file.
