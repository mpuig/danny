# Examples

Request templates for the decision patterns in [Jev's use-case catalog](https://docs.typesafe.ai/),
adapted to danny's three primitives. Each page shows the full request, a one-line
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

## Run any of them

```bash
# start a tier (see the main README quickstart), then:
curl -s -X POST http://127.0.0.1:8399/v1/systemone \
  -H "Content-Type: application/json" --data @examples/support-triage.json
```

## Read this before trusting a threshold

All eight workloads are **out-of-family** relative to danny's training data.
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
