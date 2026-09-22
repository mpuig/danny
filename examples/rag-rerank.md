# RAG reranking

**Decision shapes:** scoring · ranking · retrieval — after the "Search and retrieval" use case in Jev's docs.

Score one query-candidate pair on an ordered relevance rubric. Run it per candidate and sort by score — a cross-encoder-style reranker with calibrated probabilities instead of opaque logits.

## Request

```json
{
  "state": {
    "query": "How do I rotate API keys without downtime?",
    "candidate_passage": "Key rotation is supported through overlapping validity windows: create a second key, deploy it to all services, verify traffic has moved by watching the per-key request metrics, then revoke the old key. Both keys remain valid during the overlap, so no request is ever rejected during the rotation."
  },
  "questions": {
    "relevance": {
      "type": "score",
      "instructions": "How well does candidate_passage answer the query?",
      "criteria": [
        "Irrelevant; different topic entirely.",
        "Tangential; shares keywords but does not answer.",
        "Partial; answers some of the question or lacks steps.",
        "Strong; directly answers with actionable detail."
      ]
    },
    "self_contained": {
      "type": "noul",
      "instructions": "Could candidate_passage answer the query on its own, without additional context?"
    }
  }
}
```

With a [local server running](../README.md#quickstart):

```bash
curl -s -X POST http://127.0.0.1:8399/v1/systemone \
  -H "Content-Type: application/json" --data @examples/rag-rerank.json
```

## Actual output (v0.1.0, temperature-scaled)

| Question | Volume tier (0.6B) | Quality tier (2.5B q8) |
|---|---|---|
| `relevance` | score **2.86**, top level 3 @ 0.89 | score **2.77**, top level 3 @ 0.84 |
| `self_contained` | `0.79` P(yes) | `0.69` P(yes) |

## Reading the answers

The textbook case: both tiers put relevance at ~2.8 of 3 with high confidence in the top level (0.89 / 0.84) — the passage directly answers the query with actionable steps — and both consider it self-contained. Score questions shine here because the *expected score* (a fractional value) gives a smooth ranking key even when the argmax level ties across candidates.

## Acting on it

One request per candidate, rank by the `score` field, and keep the probability distribution for tie-breaks and confidence-aware cutoffs (drop candidates whose top level is `Irrelevant` at any confidence).

---
*Out-of-family caveat: none of these workloads were in the training or evaluation
family, and [measured confident-error rates out-of-family are 14–19% at t≥0.9](../docs/EXPERIMENTS.md)
until you fit per-workload temperatures (~100 labeled decisions,
`POST /v1/calibrations` — see [SERVING.md](../docs/SERVING.md)). Thresholds in
"Acting on it" are patterns, not certified numbers.*
