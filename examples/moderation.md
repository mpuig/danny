# Content moderation

**Decision shapes:** classification · scoring · detection — after the "Moderation and trust & safety" use case in Jev's docs.

Categorize a post against policy, score the severity of leaving it up, and detect whether a human needs to look — the allow/warn/review/block pattern.

## Request

```json
{
  "state": {
    "community": "consumer-fitness app, public comments",
    "post": "Stop wasting money on protein. I lost 20kg taking 3x the label dose of these fat burner pills, DM me for my supplier, way cheaper than pharmacy and no prescription needed."
  },
  "questions": {
    "violation": {
      "type": "choice",
      "instructions": "Which moderation category best fits the post?",
      "criteria": {
        "none": "No policy violation.",
        "unsafe_advice": "Encourages harmful health practices or misuse.",
        "spam_or_solicitation": "Unsolicited selling, referral bait, or link farming.",
        "harassment": "Attacks or demeans a person or group.",
        "prohibited_goods": "Sells or sources regulated or illegal products."
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is the harm if this post stays up?",
      "criteria": [
        "Minimal; cosmetic or off-topic.",
        "Low; mildly misleading or annoying.",
        "Moderate; could cause real harm to some readers.",
        "High; likely to cause physical, financial, or legal harm."
      ]
    },
    "review_needed": {
      "type": "noul",
      "instructions": "Should a human moderator review this post before any automated action?"
    }
  }
}
```

With a [local server running](../README.md#quickstart):

```bash
curl -s -X POST http://127.0.0.1:8399/v1/systemone \
  -H "Content-Type: application/json" --data @examples/moderation.json
```

## Actual output (v0.1.0, temperature-scaled)

| Question | Volume tier (0.6B) | Quality tier (2.5B q8) |
|---|---|---|
| `violation` | **unsafe_advice** @ 0.54 | **unsafe_advice** @ 0.28 |
| `severity` | score **2.62**, top level 3 @ 0.71 | score **2.29**, top level 3 @ 0.55 |
| `review_needed` | `0.55` P(yes) | `0.76` P(yes) |

## Reading the answers

A deliberate stress test that teaches rubric design: the post violates *three* categories at once (unsafe dosage advice, solicitation, sourcing prescription-free pills), so the single-choice question splits its probability mass — the quality tier's top category is only 0.28 despite the post being obviously bad. The operational signals carry the load instead: severity ~2.3–2.6 of 3 and `review_needed` at 0.55/0.76. **Lesson: when violations can co-occur, ask one noul detector per policy instead of one choice across policies.**

## Acting on it

Block on severity ≥ high threshold, queue for human review on `review_needed`, and if you need per-policy decisions, restructure as parallel noul questions — the engine answers all of them in one request.

---
*Out-of-family caveat: none of these workloads were in the training or evaluation
family, and [measured confident-error rates out-of-family are 14–19% at t≥0.9](../docs/EXPERIMENTS.md)
until you fit per-workload temperatures (~100 labeled decisions,
`POST /v1/calibrations` — see [SERVING.md](../docs/SERVING.md)). Thresholds in
"Acting on it" are patterns, not certified numbers.*
