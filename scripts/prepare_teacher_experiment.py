"""Matched gold/hard/soft/mixed experiments from the EXISTING teacher cache.

No API calls. The cache lacks resolved teacher versions and original source IDs;
those facts are preserved as unknown, not repaired by inventing provenance.
Models in this experiment must start from the base backbone, not a Kev adapter.
"""

import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

from jev.data import (
    Example,
    content_key,
    grouped_indices,
    normalize_target,
    read_jsonl,
    sha256_file,
    split_examples,
)
from jev.recast import TASKS
from jev.rendering import LEGACY_V0, render
from jev.serialization import dumps


def recover(rows: list[dict], digest: str) -> list[Example]:
    examples = []
    for index, row in enumerate(rows):
        task = TASKS[row["task"]]
        prompt = row["prompt"]
        state = prompt.split("State:\n", 1)[1].rsplit("\n\nQuestion:", 1)[0]
        matches = [
            task.question(i)
            for i in range(len(task.phrasings))
            if render(state, task.question(i), LEGACY_V0) == (prompt, row["labels"])
        ]
        if len(matches) != 1:
            raise ValueError(
                f"row {index}: cannot exactly recover historical state/rubric"
            )
        question = matches[0]
        target = normalize_target(row["target"], len(question.answer_keys))
        answer = row["jev"]
        probabilities = (
            [1 - answer["noul"], answer["noul"]]
            if question.type == "noul"
            else [answer["probabilities"][key] for key in question.answer_keys]
        )
        if any(
            abs(a - b) > 1e-10
            for a, b in zip(target, normalize_target(probabilities, len(target)))
        ):
            raise ValueError("cached target does not match teacher answer")
        label = row["gold_label"]
        if type(label) is not int or not 0 <= label < len(target):
            raise ValueError("invalid gold label")
        identity = hashlib.sha256(f"{digest}:{index}".encode()).hexdigest()
        examples.append(
            Example(
                id=f"teacher-cache:{identity}",
                group_id=content_key(state),
                source=f"cached:{task.name}",
                state=state,
                question=question,
                target=target,
                original_target=row["target"],
                target_origin="teacher-soft",
                provenance={
                    "cache_sha256": digest,
                    "cache_row": index,
                    "source_dataset": task.hf_id,
                    "original_source_id": None,
                    "teacher_requested_model": "jev-latest",
                    "teacher_resolved_model": None,
                    "gold_label": label,
                    "teacher_answer": answer,
                    "original_teacher_target": row["target"],
                    "state_policy": "historical 1500-character truncation retained",
                    "rubric_recovery": "exact legacy prompt round trip",
                },
            )
        )
    return examples


def with_target(example: Example, regime: str) -> Example:
    gold = [
        float(i == example.provenance["gold_label"]) for i in range(len(example.target))
    ]
    hard = [
        float(i == max(range(len(example.target)), key=example.target.__getitem__))
        for i in range(len(example.target))
    ]
    target = {
        "gold": gold,
        "hard": hard,
        "soft": example.target,
        "mixed": [(a + b) / 2 for a, b in zip(gold, example.target)],
    }[regime]
    return replace(
        example,
        target=target,
        original_target=example.original_target if regime == "soft" else target,
        target_origin="gold" if regime == "gold" else f"teacher-{regime}",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/distill_train.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.out_dir.exists():
        ap.error("output exists")
    digest = sha256_file(args.cache)
    examples = recover(list(read_jsonl(args.cache)), digest)
    groups = grouped_indices(examples)
    random.Random(args.seed).shuffle(groups)
    test_groups = max(1, round(len(groups) * 0.2))
    test = [examples[i] for group in groups[:test_groups] for i in group]
    pool = [examples[i] for group in groups[test_groups:] for i in group]
    splits, removed = split_examples(pool, test, seed=args.seed)
    if removed:
        raise ValueError("unexpected overlap after connected-group assignment")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "cache_sha256": digest,
        "seed": args.seed,
        "examples": len(examples),
        "groups": len(groups),
        "teacher_version": None,
        "limitations": "Historical unversioned teacher; exact recovered truncated inputs. No claim about current Jev. Start each variant from base weights.",
        "files": {},
        "variants": {},
    }

    def write(name, rows):
        path = args.out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(dumps(e.to_dict()) + "\n" for e in rows))
        return {"sha256": sha256_file(path), "examples": len(rows)}

    for regime in ("gold", "hard", "soft", "mixed"):
        manifest["variants"][regime] = write(
            f"{regime}/train.jsonl", [with_target(e, regime) for e in splits["train"]]
        )
    for split in ("development", "calibration", "test"):
        manifest["files"][f"{split}.jsonl"] = write(
            f"{split}.jsonl", [with_target(e, "gold") for e in splits[split]]
        )
    manifest["files"]["teacher_test.jsonl"] = write(
        "teacher_test.jsonl", splits["test"]
    )
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
