# Citation checking

**Decision shapes:** verification · detection · scoring — after the "Scientific discovery / universal verification" use case in Jev's docs.

Does the cited passage actually support the claim? A binary check plus a graded faithfulness rubric — the verification pattern behind hallucination and citation audits.

## Request

```json
{
  "state": {
    "claim": "The study found that remote workers were 13% more productive than their office-based counterparts.",
    "cited_passage": "Treatment group workers completed 13.5% more calls per shift. The authors attribute roughly 9 points of the gain to quieter working conditions and the remainder to longer effective shifts, noting that promotion rates conditional on performance fell for the remote group."
  },
  "questions": {
    "supported": {
      "type": "noul",
      "instructions": "Does cited_passage support the claim as stated?"
    },
    "support_quality": {
      "type": "score",
      "instructions": "Rate how faithfully the claim represents the cited_passage.",
      "criteria": [
        "Contradicted; the passage says otherwise.",
        "Unsupported; the passage is about something else.",
        "Partially supported; directionally right but imprecise or missing caveats.",
        "Fully supported; numbers and framing match."
      ]
    }
  }
}
```

With a [local server running](../README.md#quickstart):

```bash
curl -s -X POST http://127.0.0.1:8399/v1/systemone \
  -H "Content-Type: application/json" --data @examples/citation-check.json
```

## Actual output (v0.1.0, temperature-scaled)

| Question | Volume tier (0.6B) | Quality tier (2.5B q8) |
|---|---|---|
| `supported` | `0.79` P(yes) | `0.44` P(yes) |
| `support_quality` | score **2.64**, top level 3 @ 0.69 | score **2.18**, top level 2 @ 0.62 |

## Reading the answers

The subtlest example, and the tiers *disagree productively*. The claim says "13% more productive"; the passage says 13.5% more *calls*, attributes the gain to specific conditions, and adds a caveat about promotions. The volume tier calls it supported (0.79); the quality tier hedges at 0.44 — and the hedge is the better answer, because the claim overreaches the evidence. Both score rubrics land in "partially supported" territory, catching what the binary blurred. When a binary and its graded twin disagree, trust the rubric.

## Acting on it

Gate on the score rubric, not the noul: accept at `Fully supported` with high top-probability, flag `Partially supported` for editing, and reject below.

---
*Out-of-family caveat: none of these workloads were in the training or evaluation
family, and [measured confident-error rates out-of-family are 14–19% at t≥0.9](../docs/EXPERIMENTS.md)
until you fit per-workload temperatures (~100 labeled decisions,
`POST /v1/calibrations` — see [SERVING.md](../docs/SERVING.md)). Thresholds in
"Acting on it" are patterns, not certified numbers.*
