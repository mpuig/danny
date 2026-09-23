# Examples

Request templates for the decision patterns in [Jev's use-case catalog](https://docs.typesafe.ai/),
adapted to system-one's three primitives. Each page shows the full request, a one-line
curl, the **actual output of both tiers** (captured at v0.1.0 — reproducible:
same weights, same temperatures, deterministic readout), an honest reading of
the answers *including the misses*, and how code should act on them.

| Example | Decision shapes | Request file |
|---|---|---|
| [Support triage](./support-triage.md) | classification · detection · scoring · routing | `support-triage.json` |
| [LLM guardrails](./llm-guardrail.md) | detection · verification · scoring | `llm-guardrail.json` |
| [Model routing](./model-routing.md) | classification · scoring · detection | `model-routing.json` |
| [RAG reranking](./rag-rerank.md) | scoring · ranking · retrieval | `rag-rerank.json` |
| [Citation checking](./citation-check.md) | verification · detection · scoring | `citation-check.json` |
| [Content moderation](./moderation.md) | classification · scoring · detection | `moderation.json` |
| [Insurance claims intake](./claims-intake.md) | classification · detection · scoring | `claims-intake.json` |
| [Lead scoring](./lead-scoring.md) | scoring · detection · routing | `lead-scoring.json` |


## Correctness across five systems

Scored with [`scripts/score_examples.py`](../scripts/score_examples.py) against
[authored intended answers](./expected.json) (noul thresholded at 0.5; choice and
score by top answer within the acceptable set). Committed raw outputs:
[`outputs/`](./outputs/). **24 hand-labeled questions — an illustration, not a
benchmark.**

| Example | 2B base (no FT) | 0.6B FT | 2B FT | 2B FT q8 (served) | Jev 1.13.0 |
|---|---|---|---|---|---|
| [citation-check](./citation-check.md) | 0/2 | 0/2 | 2/2 | 2/2 | 1/2 |
| [claims-intake](./claims-intake.md) | 3/4 | 3/4 | 4/4 | 4/4 | 4/4 |
| [lead-scoring](./lead-scoring.md) | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| [llm-guardrail](./llm-guardrail.md) | 2/3 | 2/3 | 2/3 | 2/3 | 3/3 |
| [model-routing](./model-routing.md) | 2/3 | 3/3 | 2/3 | 2/3 | 3/3 |
| [moderation](./moderation.md) | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| [rag-rerank](./rag-rerank.md) | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| [support-triage](./support-triage.md) | 3/4 | 3/4 | 4/4 | 4/4 | 4/4 |
| **Total** | **17/24** | **17/24** | **22/24** | **22/24** | **23/24** |

The ladder reads left to right: fine-tuning is worth +5 questions on the same
2B backbone, 8-bit quantization costs nothing, and pinned Jev leads by one —
it is the only system that catches the prompt injection, while the fine-tuned
tiers are the only ones that catch the citation overreach Jev waves through.

## Run any of them

```bash
# start a tier (see the main README quickstart), then:
curl -s -X POST http://127.0.0.1:8399/v1/systemone \
  -H "Content-Type: application/json" --data @examples/support-triage.json
```

## Read this before trusting a threshold

All eight workloads are **out-of-family** relative to system-one's training data.
The project measured what that means: confident-error rates of **14–19% at
t≥0.9** out-of-family versus ~2% in-family ([EXPERIMENTS §14](../docs/EXPERIMENTS.md)).
Two of the pages show live consequences — both tiers miss a blatant prompt
injection, and single-choice moderation splits probability across co-occurring
violations. The measured fix: fit per-workload temperatures from ~100 labeled
decisions of your own traffic (`POST /v1/calibrations`,
[SERVING.md](../docs/SERVING.md)), which cut out-of-family confident errors to
3–4% where miscalibration is scalar — and tells you, via its verdict, when it
is not.

The pattern everywhere: **the model answers atomic questions; code owns the
policy.** Thresholds, escalation bands, and side effects live in your workflow,
where they can be reviewed and changed without retraining.
