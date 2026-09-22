# Lead scoring

**Decision shapes:** scoring · detection · routing — after the "Lead generation" use case in Jev's docs.

Match an inbound message against an ideal customer profile: graded fit, buying-intent detection, and a routing decision — the SDR queue, automated.

## Request

```json
{
  "state": {
    "ideal_customer_profile": "B2B SaaS companies, 50-500 employees, with in-house support teams handling over 1,000 tickets per month.",
    "inbound_message": "Hi - I run support ops at a 120-person logistics software company. We're drowning: about 4,000 tickets/month across email and chat, and our team of 9 can't keep up with triage. Currently evaluating tools for this quarter's budget. Can you share pricing and whether you integrate with Zendesk?"
  },
  "questions": {
    "icp_fit": {
      "type": "score",
      "instructions": "How well does the sender's company match the ideal_customer_profile?",
      "criteria": [
        "No fit; wrong segment entirely.",
        "Weak; matches one attribute loosely.",
        "Good; matches most attributes.",
        "Excellent; squarely inside the profile."
      ]
    },
    "buying_intent": {
      "type": "noul",
      "instructions": "Does the message show active purchase intent (budget, timeline, evaluation), not just curiosity?"
    },
    "next_step": {
      "type": "choice",
      "instructions": "What should happen next with this lead?",
      "criteria": {
        "route_to_sales": "Qualified; a salesperson should reply promptly.",
        "nurture": "Interested but not ready; add to nurture sequence.",
        "self_serve": "Point to docs/pricing page; no sales touch needed.",
        "disqualify": "Not a fit; polite decline."
      }
    }
  }
}
```

With a [local server running](../README.md#quickstart):

```bash
curl -s -X POST http://127.0.0.1:8399/v1/systemone \
  -H "Content-Type: application/json" --data @examples/lead-scoring.json
```

## Actual output (v0.1.0, temperature-scaled)

| Question | Volume tier (0.6B) | Quality tier (2.5B q8) |
|---|---|---|
| `icp_fit` | score **1.25**, top level 0 @ 0.36 | score **2.00**, top level 3 @ 0.38 |
| `buying_intent` | `0.70` P(yes) | `0.87` P(yes) |
| `next_step` | **nurture** @ 0.38 | **route_to_sales** @ 0.43 |

## Reading the answers

A textbook fit lead (right size, right pain, active evaluation, budget timeline) — and the tiers split. The quality tier reads it correctly: ICP fit 2.0 of 3, intent 0.87, route to sales. The volume tier under-reads the fit (1.25) and would send a hot lead to a nurture sequence. Missed revenue is the price of the cheap tier on nuanced matching; this is a workload where the 118 ms difference is obviously worth it.

## Acting on it

Route on the combination: `route_to_sales` when intent is high AND fit ≥ your bar; let marketing own the thresholds and revisit them against closed-won ground truth — exactly the feature-extraction loop Jev's docs describe.

---
*Out-of-family caveat: none of these workloads were in the training or evaluation
family, and [measured confident-error rates out-of-family are 14–19% at t≥0.9](../docs/EXPERIMENTS.md)
until you fit per-workload temperatures (~100 labeled decisions,
`POST /v1/calibrations` — see [SERVING.md](../docs/SERVING.md)). Thresholds in
"Acting on it" are patterns, not certified numbers.*
