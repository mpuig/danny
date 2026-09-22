# Insurance claims intake

**Decision shapes:** classification · detection · scoring — after the "Insurance claims" use case in Jev's docs.

First-notice-of-loss triage: classify the claim, detect missing information and fraud indicators, and score complexity to choose straight-through processing versus specialist review.

## Request

```json
{
  "state": {
    "policy_type": "home contents",
    "first_notice_of_loss": "Policyholder reports a water leak discovered Saturday morning under the kitchen sink. States the dishwasher hose 'probably' failed some weeks ago. Laminate flooring is warped across kitchen and hallway. No photos provided yet. Policyholder mentions a similar claim at a previous address two years ago and asks specifically which documentation would maximize the payout."
  },
  "questions": {
    "claim_type": {
      "type": "choice",
      "instructions": "Classify the reported loss.",
      "criteria": {
        "water_damage": "Leaks, escapes of water, flooding from internal sources.",
        "fire": "Fire, smoke, or scorching damage.",
        "theft": "Burglary or theft of contents.",
        "accidental_damage": "Sudden one-off damage to contents."
      }
    },
    "missing_information": {
      "type": "noul",
      "instructions": "Is key information missing to assess this claim (evidence, dates, cause)?"
    },
    "fraud_indicator": {
      "type": "noul",
      "instructions": "Does the report contain indicators that warrant fraud screening (not an accusation)?"
    },
    "complexity": {
      "type": "score",
      "instructions": "How complex will this claim be to process?",
      "criteria": [
        "Simple; clear cause, small scope, straight-through processable.",
        "Standard; needs routine documentation and one assessment.",
        "Complex; unclear causation, gradual damage, or coverage questions.",
        "Specialist; multiple red flags or high-value investigation."
      ]
    }
  }
}
```

With a [local server running](../README.md#quickstart):

```bash
curl -s -X POST http://127.0.0.1:8399/v1/systemone \
  -H "Content-Type: application/json" --data @examples/claims-intake.json
```

## Actual output (v0.1.0, temperature-scaled)

| Question | Volume tier (0.6B) | Quality tier (2.5B q8) |
|---|---|---|
| `claim_type` | **water_damage** @ 0.78 | **water_damage** @ 0.97 |
| `missing_information` | `0.58` P(yes) | `0.78` P(yes) |
| `fraud_indicator` | `0.37` P(yes) | `0.69` P(yes) |
| `complexity` | score **1.87**, top level 2 @ 0.43 | score **2.27**, top level 3 @ 0.46 |

## Reading the answers

The clearest quality-tier win in the set. It classifies water damage at 0.97, flags missing information at 0.78 (no photos, vague dates), and raises the fraud indicator to 0.69 — plausibly reading the gradual-damage framing, the prior claim, and the payout-maximizing question as a cluster. The volume tier sees the same claim at 0.37 fraud signal. For judgment-heavy intake where a miss is expensive, the +6.8-point tier earns its latency.

## Acting on it

Straight-through only when complexity is low AND both detectors are quiet; anything with `fraud_indicator` above your calibrated threshold routes to screening — as an indicator for a workflow, never an accusation.

---
*Out-of-family caveat: none of these workloads were in the training or evaluation
family, and [measured confident-error rates out-of-family are 14–19% at t≥0.9](../docs/EXPERIMENTS.md)
until you fit per-workload temperatures (~100 labeled decisions,
`POST /v1/calibrations` — see [SERVING.md](../docs/SERVING.md)). Thresholds in
"Acting on it" are patterns, not certified numbers.*
