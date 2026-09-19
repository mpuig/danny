# Decision Log

Each entry: the decision, why, and the evidence. Reverse-order additions welcome.

## 1. Fine-tune a small decoder; no novel architecture
Jev is almost certainly a fine-tuned transformer with a custom readout, not a new network
(confirmed later by black-box probing). New behavior lives in: (a) inference readout,
(b) fine-tuning — 90% of the value, (c) small structural additions only when a measured
wall appears. Building the clever architecture first would delay learning whether data +
loss produce calibration, which is the actual open question.

## 2. Decoder backbone, not encoder (BERT/GLiFormer) or FunctionGemma
The label space is defined at request time in natural language — that needs instruction
understanding and world knowledge. Encoders cap out (jeff's own benchmarks: "substantially
behind on irony and reading comprehension"). FunctionGemma's specialization (generating
tool-call JSON) is bypassed entirely by logit readout, and ~100M of non-embedding params
is too small. Gemma 3 270M remains a candidate distillation target for fixed high-volume
questions.

## 3. Backbones: SmolLM3-3B-Base daily driver, SmolLM2-135M smoke tests
SmolLM3 has a fully open training recipe (debuggable calibration behavior), Apache 2.0,
base checkpoint. Qwen3-4B-Base remains the planned "checkpoint of record" comparison.
**Base checkpoints only** — RLHF/DPO wrecks calibration (mode collapse, overconfidence).

## 4. Apple MLX (`mlx-lm`), not PyTorch
User decision; dev machine is a 36 GB Apple Silicon Mac. mlx-lm loads HF checkpoints,
exposes logits, and adapters interop with `mlx_lm.load(adapter_path=...)`. Schema, metrics,
and data modules stay framework-agnostic for a later port.

## 5. Noul renders through the choice template
Bare yes/no completion on a base model shows severe acquiescence bias: SmolLM3-3B answered
"yes" 94% of the time on SST-2 and the residual signal was slightly *inverted* (calibration
alone pushed accuracy below chance, 0.41). Lettered A/no B/yes readout: 0.535 → **0.695
acc**, ECE 0.297 → **0.088**. API unchanged (still returns one `noul` value).

## 6. Letter labels; score levels lettered too
` 0` is two tokens in SmolLM2's vocab (space + digit) — all score levels collided on the
space token. ` A`…` Z` are single tokens across the BPE vocabs we use. Score responses map
letters back to level numbers. Consequence: v0 caps choice at 26 options.

## 7. Contextual calibration (Zhao et al. 2021) as an engine feature
Divide out the label prior measured on content-free states. Zero-training gains on choice
(ag_news +5 acc, −22% Brier). Diagnostic value: on a broken template it exposes absent
signal rather than masking it (SST-2 below chance → led to decision #5).

## 8. Training loss = the inference readout
Cross-entropy over the restricted label-token softmax at the last prompt position, against
a target *distribution* (one-hot or soft). CE is a proper scoring rule → calibration in
expectation with honest targets. `mlx_lm.lora`'s text-completion loss doesn't fit; we run a
custom loop (`scripts/train_lora.py`).

## 9. LoRA lr 1e-5 for the 3B (1e-4 is destructive)
Measured: lr 1e-4 took 3B val loss 0.856 → **1.59** (worse than untuned). lr 1e-5:
0.856 → **0.349**. The 135M tolerated 1e-4. Rank 16, scale 20, attention projections,
all layers.

## 10. Held-out-task evaluation protocol
Train tasks: ag_news, dbpedia, imdb, yelp_stars. **Never trained**: sst2, tweet_emotion.
Generalization to unseen questions is the product claim; in-distribution accuracy is not
evidence (kev's key limitation).

## 11. Distill from the real Jev API (user's idea)
Jev's distributions are exactly the soft targets the loss wants, and store its judgment on
ambiguity (45% of pulled targets are genuinely soft; mean normalized entropy 0.162; Jev
argmax = gold label 88.4% — the 11.6% gap is label noise + ambiguity + Jev errors, all of
which distillation transfers). Also enables a KL-agreement benchmark against Jev. **Check
Typesafe's ToS before using distilled weights beyond experimentation.** Ops notes:
retries must cover HTTP 529 and socket `TimeoutError`; puller is resumable (`--resume`).

## 12. Option-shuffling augmentation (from kev)
Fixed option order teaches letter positions, not option text; kev still flips 7.4% of
argmaxes under reordering *with* augmentation. We shuffle choice/noul options per training
example (target remapped) and drop descriptions on ~20% of choice examples.
`scripts/permutation_test.py` measures flip rate and TV distance.

## 13. Shared-prefix KV caching (from Nimble's ParallelScorer)
Token-level common prefix across a request's prompts is encoded once; suffixes scored and
rewound (`trim_prompt_cache`). Token-level matching (not string-level) avoids BPE boundary
drift; rewinding preserves question independence.

## 14. Server: single-threaded, HTTP/1.1
MLX ops crash off the owning thread; undici rejects HTTP/1.0. See ARCHITECTURE.md.

## 15. Dataset substitutions
`trec` (legacy HF script, unsupported by current `datasets`) → `fancyzhx/dbpedia_14`
(14 options, label order verified against dataset metadata). `ag_news` must load as
`fancyzhx/ag_news` (namespaced IDs required by current `huggingface_hub`).

## 16. Flat multi-question latency via batched suffixes + readout-position head
Goal: 100 questions ≈ 1 question (Jev's probed behavior). Implemented without retraining:
prefix KV cache tiled across the batch, all suffixes in one forward, LM head computed only
at each row's last real token (the full `(n, len, vocab)` logits tensor was the bottleneck
— ~300 MB for 100 questions). Result: 50 q 0.29 s → 100 q 0.30 s on the 135M under load;
answers bit-identical to sequential. The true packed design (one sequence, block-diagonal
mask, per-block position reset) is deferred to v2 because it needs model surgery *and*
fine-tuning — same milestone as the scoring head, per kev/archerhume.

## 17. Planned, not yet built
Contrastive minimal pairs **scored by Jev** (Nimble's evidence-sensitivity + our calibrated
targets — the combination nobody has); per-primitive temperature scaling; one-hot retrain
on shuffled data; v1 reserved option tokens; v2 scoring head + packed questions.
