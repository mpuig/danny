# Model routing

**Decision shapes:** classification · scoring · detection — after the "Model routing / harness engineering" use case in Jev's docs.

Decide which LLM gets the prompt: classify domain, score difficulty on an ordered rubric, and detect when a small model would be risky — before spending frontier-model money.

## Request

```json
{
  "state": {
    "prompt": "We have a Postgres table of 40M events with a composite index on (tenant_id, created_at). Query latency degraded 20x after adding a JSONB column with per-row metadata blobs of ~8KB. Explain the most likely causes, how TOAST storage interacts with our index-only scans, and propose a migration plan that avoids downtime."
  },
  "questions": {
    "domain": {
      "type": "choice",
      "instructions": "Which domain best matches the prompt?",
      "criteria": {
        "software_engineering": "Code, databases, infrastructure, debugging.",
        "writing": "Prose drafting, editing, or summarization.",
        "data_analysis": "Statistics, spreadsheets, business metrics.",
        "general_chat": "Casual conversation or broad questions."
      }
    },
    "difficulty": {
      "type": "score",
      "instructions": "How difficult is this prompt for a language model?",
      "criteria": [
        "Trivial; a one-line factual answer suffices.",
        "Easy; common knowledge, low reasoning depth.",
        "Moderate; multi-step reasoning or niche knowledge.",
        "Hard; deep domain expertise and multi-part synthesis."
      ]
    },
    "needs_frontier_model": {
      "type": "noul",
      "instructions": "Would routing this to a small general model risk a materially wrong or incomplete answer?"
    }
  }
}
```

With a [local server running](../README.md#quickstart):

```bash
curl -s -X POST http://127.0.0.1:8399/v1/systemone \
  -H "Content-Type: application/json" --data @examples/model-routing.json
```

## Actual output (v0.1.0, temperature-scaled)

| Question | Volume tier (0.6B) | Quality tier (2.5B q8) |
|---|---|---|
| `domain` | **software_engineering** @ 0.86 | **software_engineering** @ 1.00 |
| `difficulty` | score **2.49**, top level 3 @ 0.55 | score **2.39**, top level 3 @ 0.55 |
| `needs_frontier_model` | `0.55` P(yes) | `0.44` P(yes) |

## Reading the answers

Domain classification is emphatic (0.86 volume, ~1.00 quality — a rounded post-temperature probability, still not a certainty claim). Difficulty lands at ~2.4 of 3, brushing the "hard" level, while `needs_frontier_model` hedges near 0.5 on both tiers — the model is less sure about *consequences* than about *properties*, which is a common and honest pattern.

## Acting on it

Combine signals in code rather than trusting one: route up when `difficulty ≥ 2.2` **or** `needs_frontier_model ≥ 0.6`, and log the probabilities so routing mistakes are auditable.

---
*Out-of-family caveat: none of these workloads were in the training or evaluation
family, and [measured confident-error rates out-of-family are 14–19% at t≥0.9](../docs/EXPERIMENTS.md)
until you fit per-workload temperatures (~100 labeled decisions,
`POST /v1/calibrations` — see [SERVING.md](../docs/SERVING.md)). Thresholds in
"Acting on it" are patterns, not certified numbers.*
