# Overview

## What this is

An open reimplementation of the ideas behind [Typesafe's Jev](https://docs.typesafe.ai/introduction):
a **System One model** that answers typed questions about a text `state` and returns
calibrated probability distributions instead of generated text. Software consumes the
answers directly — no parsing, no hallucinated formats.

Three primitives, mirroring Jev's API exactly:

| Primitive | Question | Returns |
|---|---|---|
| `choice` | which of these options? | `choice`, `probabilities`, `confidence` |
| `score` | which level on an ordered spectrum? | `score` (probability-weighted mean), `probabilities`, `legend`, `confidence` |
| `noul` | is this true? | `noul` = P(yes) ∈ [0, 1] |

A request is `{state, questions}` (questions keyed by ID); the response answers **every**
question — guaranteed by construction, since nothing is generated or parsed.

## Goals

1. **Calibration first.** Jev's product is honest uncertainty ("if an intelligent system
   cannot express honest uncertainty, the system cannot be trusted"). Our primary metrics
   are ECE, Brier, and NLL — accuracy is secondary.
2. **Generalization to unseen questions.** One model, arbitrary rubrics defined at request
   time. We evaluate on tasks that were never in training.
3. **Drop-in API compatibility.** The official Typesafe SDKs work against our server via a
   base-URL override (verified with `@typesafe-ai/sdk` for TypeScript).
4. **Local and cheap.** Runs on Apple Silicon via MLX; small backbones (135M–3B).
5. **Open recipe.** Everything reproducible: data recasting, training loop, eval harness.

Non-goals: matching Jev's frontier-scale reasoning (its ~83% MMLU-Pro implies a ~10B-active
MoE backbone), text generation, multimodality.

## Why not just prompt an LLM?

LLM token probabilities exist, but (a) chat/RLHF models are miscalibrated by training,
(b) zero-shot templates carry severe biases (we measured a 94% yes-bias on bare yes/no
prompts), and (c) generation + parsing is slow and can fail. This project keeps the
pretrained knowledge, replaces generation with a one-forward-pass logit readout, and
trains specifically for calibrated decisions.

## Competitive landscape (reviewed 2026-09)

| Project | Backbone | Approach | Gap |
|---|---|---|---|
| [jeff](https://github.com/logan-markewich/jeff) | GLiFormer 400M encoder | zero-shot heads, API server | weak on irony/comprehension; breaks question independence for throughput |
| [kev](https://github.com/jaredpalmer/kev) | Qwen2.5-0.5B frozen + scoring head | block-causal packed questions, augmentation | in-distribution only; 0.5B ceiling |
| [Bespoke Nimble](https://github.com/bespokelabsai/nimble) | Qwen3.5-9B logit readout | contrastive minimal-pair curation (90.1% vs Jev 93.2%) | calibration untuned |
| openjev / openjev-sglang | various | logit readout replicas | accuracy-focused |
| **this project** | SmolLM3-3B (+ 135M for smoke tests) | logit readout + **Jev-distilled soft targets**, calibration-first, held-out generalization | see roadmap |

Best external evidence on Jev's internals: [archerhume.com probing article](https://archerhume.com/posts/jevs-architecture-unmasked/) —
shared-state encoding with isolated per-question branches, trained classifier head,
proper-scoring-rule training (ECE 0.031 on MMLU), probable sparse MoE. This matches our
Phase-3 roadmap (see ARCHITECTURE.md).

## Status

- Engine, Jev-compatible server, TS-SDK drop-in verification, training loop, distillation
  pipeline, eval harness: **done and tested**.
- Best held-out result so far (3B + one-hot LoRA + contextual calibration, n=200):
  sst2 **0.925 acc / 0.068 ECE**; tweet_emotion ECE 0.205 → **0.134** at equal accuracy.
- In flight: Jev-distilled (soft-target) adapter; then three-way comparison and
  KL-agreement-vs-Jev eval.
