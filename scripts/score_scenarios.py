"""Score generated scenarios with a pinned Jev teacher into canonical examples.

Reads scenario rows from generate_scenarios.py, asks the pinned teacher for each
question, and writes format_version-1 canonical examples (data.Example) whose
target is the teacher's distribution. Resumable by example id. The response's
reported model id is recorded per row; a manifest summarizes hashes, counts, and
generator/teacher argmax disagreement.

    uv run python scripts/score_scenarios.py data/experiments/synthetic-v1/scenarios.jsonl \
        --teacher jev-1.13.0 --out data/experiments/synthetic-v1/examples.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from tqdm import tqdm

from jev.data import Example, read_jsonl, sha256_file
from jev.jev_api import api_key, systemone
from jev.schema import Question

SOURCE = "synthetic-fireworks-v1"


def teacher_target(question: Question, answer: dict) -> list[float]:
    if question.type == "noul":
        p_yes = float(answer["noul"])
        return [1.0 - p_yes, p_yes]
    probabilities = answer["probabilities"]
    return [float(probabilities[key]) for key in question.answer_keys] \
        if question.type == "score" else \
        [float(probabilities[name]) for name in question.criteria]


def generator_argmax(question: Question, expected) -> int:
    if question.type == "choice":
        return list(question.criteria).index(expected)
    if question.type == "score":
        return int(expected)
    return int(bool(expected))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios")
    ap.add_argument("--teacher", required=True,
                    help="pinned Jev model id, e.g. jev-1.13.0 (aliases move; avoid them)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.teacher.endswith(("-latest", "-preview")):
        ap.error("pin a versioned teacher id; aliases move between releases")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    key = api_key()
    rows = list(read_jsonl(args.scenarios))
    scenario_by_id = {row["id"]: row for row in rows}

    done: set[str] = set()
    if args.resume and out.exists():
        for existing in read_jsonl(out):
            teacher = existing.get("provenance", {}).get("teacher", {})
            if teacher.get("requested") != args.teacher:
                raise SystemExit(
                    f"{out}: existing row {existing['id']} was scored with teacher "
                    f"{teacher.get('requested')!r}, not {args.teacher!r}; use a new path")
            if teacher.get("reported") not in ("", args.teacher):
                raise SystemExit(
                    f"{out}: existing row {existing['id']} reports teacher "
                    f"{teacher.get('reported')!r}; refusing to mix versions")
            source = scenario_by_id.get(existing["id"])
            if source is None or source["state"] != existing["state"]:
                raise SystemExit(
                    f"{out}: existing row {existing['id']} does not match the current "
                    f"scenario file; inputs changed since the earlier run")
            done.add(existing["id"])
    elif out.exists():
        raise SystemExit(f"{out} exists; use --resume or a new path")
    reported_models: set[str] = set()
    scored, agree = 0, 0

    with out.open("a") as handle:
        for row in tqdm(rows, desc=f"teacher {args.teacher}"):
            if row["id"] in done:
                continue
            question = Question(**row["question"])
            response = systemone(key, row["state"], question, model=args.teacher)
            reported = str(response.get("model", ""))
            if reported and reported != args.teacher:
                raise SystemExit(
                    f"teacher reported {reported!r} for a request pinned to "
                    f"{args.teacher!r}; aborting to keep the collection single-version")
            reported_models.add(reported)
            answer = response["answers"]["q"]
            target = teacher_target(question, answer)

            example = Example(
                id=row["id"],
                group_id=row["group_id"],
                source=SOURCE,
                state=row["state"],
                question=question,
                target=target,
                target_origin=f"teacher:{reported or args.teacher}",
                provenance={
                    "cell": row["cell"],
                    "generator": row["generator"],
                    "teacher": {
                        "requested": args.teacher,
                        "reported": reported,
                        "answer": answer,
                        "usage": response.get("usage", {}),
                        "collected": dt.date.today().isoformat(),
                    },
                    "scenario_format": row["format_version"],
                },
            )
            handle.write(json.dumps(example.to_dict()) + "\n")
            handle.flush()
            scored += 1
            if max(range(len(target)), key=target.__getitem__) \
                    == generator_argmax(question, row["generator"]["expected"]):
                agree += 1

    manifest = {
        "scenarios": {"path": args.scenarios, "sha256": sha256_file(args.scenarios),
                      "rows": len(rows)},
        "output": {"path": str(out), "sha256": sha256_file(out)},
        "teacher": {"requested": args.teacher, "reported": sorted(reported_models)},
        "scored_this_run": scored,
        "generator_teacher_argmax_agreement": round(agree / scored, 4) if scored else None,
        "source": SOURCE,
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
