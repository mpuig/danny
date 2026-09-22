# Support triage

**Decision shapes:** classification · detection · scoring · routing — after the "Customer support" use case in Jev's docs.

Classify an incoming ticket, detect refund demand and churn risk, and score urgency — four decisions in one request. Code owns what happens next (queue assignment, SLA clocks, escalation); the model only answers the atomic questions.

## Request

```json
{
  "state": {
    "channel": "email",
    "customer_tier": "premium",
    "message": "This is the third time I'm writing. My September invoice shows two charges of 49.90 EUR for the same subscription. I was told last week this would be fixed in 48 hours. Nothing happened. If this isn't resolved by Friday I'm cancelling my account and disputing the charge with my bank."
  },
  "questions": {
    "issue_area": {
      "type": "choice",
      "instructions": "Which product area does this ticket belong to?",
      "criteria": {
        "billing": "Charges, invoices, refunds, payment methods.",
        "account": "Login, credentials, profile, cancellation mechanics.",
        "product_bug": "The product malfunctions or behaves unexpectedly.",
        "how_to": "The customer needs usage guidance."
      }
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Does the customer explicitly or implicitly request money back?"
    },
    "churn_risk": {
      "type": "noul",
      "instructions": "Does the customer threaten or strongly imply they will cancel?"
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this ticket?",
      "criteria": [
        "Routine; no deadline or escalation pressure.",
        "Elevated; repeated contact or explicit dissatisfaction.",
        "Urgent; deadline, threat, or imminent financial action."
      ]
    }
  }
}
```

With a [local server running](../README.md#quickstart):

```bash
curl -s -X POST http://127.0.0.1:8399/v1/systemone \
  -H "Content-Type: application/json" --data @examples/support-triage.json
```

## Actual output (v0.1.0; ✓/✗ against [authored intended answers](./expected.json))

| Question | 2B base | 0.6B FT | 2B FT | 2B FT q8 | Jev 1.13.0 |
|---|---|---|---|---|---|
| `issue_area` | billing (0.50) ✓ | billing (0.82) ✓ | billing (0.93) ✓ | billing (0.93) ✓ | billing (1.00) ✓ |
| `refund_requested` | 0.50 ✓ | 0.45 ✗ | 0.68 ✓ | 0.67 ✓ | 0.90 ✓ |
| `churn_risk` | 0.62 ✓ | 0.82 ✓ | 0.92 ✓ | 0.91 ✓ | 0.98 ✓ |
| `urgency` | L0 (0.39) ✗ | L2 (0.76) ✓ | L2 (0.90) ✓ | L2 (0.81) ✓ | L2 (1.00) ✓ |

## Reading the answers

The closest workload to the training family, and it shows. Both tiers agree confidently on `billing` and on the churn threat (0.82 / 0.91). The interesting cell is `refund_requested`: the customer never asks *us* for money back — they threaten a bank chargeback — so the volume tier's 0.45 hedge is defensible and the quality tier's 0.67 reads the implication. Urgency lands high-but-not-maximal on both, which matches a deadline threat without an outage.

## Acting on it

Route on `issue_area` when its top probability clears your threshold; open a retention play when `churn_risk` is high regardless of queue; let `urgency` order the queue rather than gate it.

---
*Out-of-family caveat: none of these workloads were in the training or evaluation
family, and [measured confident-error rates out-of-family are 14–19% at t≥0.9](../docs/EXPERIMENTS.md)
until you fit per-workload temperatures (~100 labeled decisions,
`POST /v1/calibrations` — see [SERVING.md](../docs/SERVING.md)). Thresholds in
"Acting on it" are patterns, not certified numbers.*
