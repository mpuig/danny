"""Pull soft targets from the real Jev API for our recast states.

For each training-task state, asks Jev the same question we render locally and
stores Jev's probability distribution as the training target (same JSONL format
as build_data.py, so train_lora.py consumes it unchanged). Jev's raw answer is
kept under "jev" for agreement analysis.

    uv run python scripts/distill_from_jev.py --per-task 500 --out data/distill_train.jsonl

Requires TYPESAFE_API_KEY in the environment or a .env file at the repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

from jev.engine import _LETTERS, SystemOneEngine
from jev.recast import TASKS, TRAIN_TASKS, records

API_URL = "https://api.typesafe.ai/v1/systemone"


def api_key() -> str:
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("TYPESAFE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("TYPESAFE_API_KEY not found in environment or .env")


def ask_jev(key: str, state: str, question) -> dict:
    payload = {
        "state": state,
        "model": "jev-latest",
        "questions": {
            "q": {
                "type": question.type,
                "instructions": question.instructions,
                **({"criteria": question.criteria} if question.criteria else {}),
            }
        },
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())["answers"]["q"]
        except urllib.error.HTTPError as e:
            if e.code in (408, 429, 500, 502, 503, 529) and attempt < 4:
                time.sleep(2**attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < 4:
                time.sleep(2**attempt)
                continue
            raise


def target_from_answer(task, answer: dict) -> list[float]:
    """Jev's distribution, ordered to match our lettered option order."""
    if task.question_type == "noul":
        p_yes = float(answer["noul"])
        return [1.0 - p_yes, p_yes]  # our noul-as-choice order: [no, yes]
    probs = answer["probabilities"]
    if task.question_type == "score":
        return [float(probs[str(i)]) for i in range(len(task.criteria))]
    return [float(probs[name]) for name in task.criteria]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=500)
    ap.add_argument("--out", default="data/distill_train.jsonl")
    ap.add_argument("--seed", type=int, default=7)  # differs from build_data's 42
    ap.add_argument("--resume", action="store_true", help="append missing rows only")
    args = ap.parse_args()

    key = api_key()
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)

    # resume: same seed -> same record order, so skip rows already on disk
    existing: dict[str, int] = {}
    if args.resume and out_path.exists():
        for line in out_path.open():
            row = json.loads(line)
            existing[row["task"]] = existing.get(row["task"], 0) + 1
        print(f"resuming; existing rows per task: {existing}")

    written = 0
    with open(args.out, "a" if args.resume else "w") as f:
        for name in TRAIN_TASKS:
            task = TASKS[name]
            recs = list(
                records(task, split=task.train_split, n=args.per_task, seed=args.seed)
            )
            done = existing.get(name, 0)
            for i, rec in enumerate(tqdm(recs, desc=name)):
                if i < done:
                    continue
                q = task.question(phrasing=i % len(task.phrasings))
                answer = ask_jev(key, rec["state"], q)
                target = target_from_answer(task, answer)

                if q.type == "noul":
                    render_q = SystemOneEngine._as_yes_no_choice(q)
                    prompt, _ = SystemOneEngine._choice_prompt(rec["state"], render_q)
                elif q.type == "score":
                    prompt = SystemOneEngine._score_prompt(rec["state"], q)
                else:
                    prompt, _ = SystemOneEngine._choice_prompt(rec["state"], q)

                f.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "labels": [f" {_LETTERS[j]}" for j in range(len(target))],
                            "target": target,
                            "task": task.name,
                            "gold_label": rec["label"],
                            "jev": answer,
                        }
                    )
                    + "\n"
                )
                written += 1
    print(f"wrote {written} rows -> {args.out}")


if __name__ == "__main__":
    main()
