"""Screen synthetic Score rubrics for ordinality (decision 23, finding 1).

For every distinct Score question in the given canonical example files, asks a
judge model whether the levels form increasing positions of ONE monotonic
quantity or are distinct categorical outcomes. Writes per-question verdicts and
prints a summary with affected row counts. This is a screening audit by a single
LLM judge at low temperature, not human adjudication: spot-check verdicts before
acting on individual rows.

    uv run python scripts/audit_ordinality.py \
        data/experiments/synthetic-v1/examples.jsonl \
        data/experiments/synthetic-v1/eval_examples.jsonl \
        --out data/experiments/synthetic-v1/ordinality_verdicts.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from generate_scenarios import DEFAULT_MODEL, chat, env_key, extract_json
from jev.data import read_jsonl

JUDGE_PROMPT = """You audit rating scales. Decide whether the following "score" question's levels
form a genuine ORDINAL scale: successive levels must be increasing positions of ONE
monotonic quantity (e.g. severity, satisfaction, completeness, risk, urgency).

If the levels are distinct categorical outcomes (different resolutions, different
entities, different event types) — even if numbered — the scale is NOT ordinal.
A final "unknown/cannot determine" level appended to an otherwise ordinal scale
still breaks strict ordinality but note it separately as "mixed_unknown_tail".

Question: {instructions}
Levels (low to high):
{levels}

Return strict JSON only:
{{"verdict": "ordinal" | "categorical" | "mixed_unknown_tail",
  "dimension": "<the monotonic quantity, or null>",
  "reason": "<one sentence>"}}"""


def rubric_key(question: dict) -> str:
    payload = json.dumps([question["instructions"], question["criteria"]], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--key-env", default="FIREWORKS_API_KEY")
    args = ap.parse_args()

    key = env_key(args.key_env)
    rubrics: dict[str, dict] = {}
    rows_by_rubric: dict[str, list[str]] = {}
    for path in args.files:
        for row in read_jsonl(path):
            if row["question"]["type"] != "score":
                continue
            k = rubric_key(row["question"])
            rubrics.setdefault(k, row["question"])
            rows_by_rubric.setdefault(k, []).append(f"{path}:{row['id']}")
    print(f"{sum(len(v) for v in rows_by_rubric.values())} score rows, "
          f"{len(rubrics)} distinct rubrics")

    def judge(item):
        k, question = item
        levels = "\n".join(f"{i}. {c}" for i, c in enumerate(question["criteria"]))
        prompt = JUDGE_PROMPT.format(instructions=json.dumps(question["instructions"]),
                                     levels=levels)
        raw = extract_json(chat(key, args.model, prompt, temperature=0.1))
        if raw.get("verdict") not in ("ordinal", "categorical", "mixed_unknown_tail"):
            raise ValueError(f"bad verdict: {raw}")
        return k, raw

    verdicts: dict[str, dict] = {}
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge, item) for item in rubrics.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="judging"):
            try:
                k, verdict = future.result()
                verdicts[k] = verdict
            except Exception:
                failures += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for k, verdict in verdicts.items():
            handle.write(json.dumps({
                "rubric_key": k,
                "instructions": rubrics[k]["instructions"],
                "criteria": rubrics[k]["criteria"],
                "rows": rows_by_rubric[k],
                **verdict,
            }) + "\n")

    summary: dict[str, dict] = {}
    for k, verdict in verdicts.items():
        bucket = summary.setdefault(verdict["verdict"], {"rubrics": 0, "rows": 0})
        bucket["rubrics"] += 1
        bucket["rows"] += len(rows_by_rubric[k])
    print(json.dumps({"summary": summary, "judge_failures": failures,
                      "unjudged_rubrics": len(rubrics) - len(verdicts),
                      "judge_model": args.model, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
