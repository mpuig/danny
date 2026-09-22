"""Build the reserved-v2 gold test partition from never-used public sources.

Purpose: the original reserved test (kev-v1/test.jsonl) was spent once on the
selected 0.6B (EXPERIMENTS §9). No unbiased test exists for the MiniCPM family.
This builder cuts a fresh, test-only partition from four public gold sources
that appear in no training, evaluation, or selection decision of this project:

  choice  yahoo_answers_topics  10-way topic       464 questions
  noul    glue/cola             acceptability      212
  noul    sms_spam              spam detection     212
  score   app_reviews           1-5 star rating    160

Composition mirrors the spent test's primitive shape (464/424/160 = 1,048).
Rotten Tomatoes / SST-family and tweet_eval sources were rejected: their text
overlaps diagnostics this project consulted. Every candidate row is checked by
content key against all four synthfiltered-corpus partitions (the spent test is
byte-identical to its test.jsonl) and deduplicated within the batch.

This partition is a reserved test ONLY: one pre-registered look (decision 29),
spent afterward for model selection. Do not train, tune, or iterate on it.

    uv run python scripts/build_reserved_v2.py --out-dir data/experiments/reserved-v2
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from datasets import load_dataset

from jev.data import FORMAT_VERSION, Example, load_examples, partition_summary, sha256_file
from jev.schema import Question
from jev.serialization import dumps

MAX_STATE_CHARS = 1500
MIN_STATE_CHARS = 10

YAHOO_TOPICS = {
    "society_culture": "Society, culture, customs, religion, and etiquette.",
    "science_math": "Science, mathematics, physics, chemistry, and biology.",
    "health": "Health, medicine, diseases, fitness, and diet.",
    "education_reference": "Education, schools, homework, and reference questions.",
    "computers_internet": "Computers, software, hardware, and the internet.",
    "sports": "Sports, athletes, teams, and competitions.",
    "business_finance": "Business, finance, money, jobs, and investing.",
    "entertainment_music": "Entertainment, music, movies, television, and celebrities.",
    "family_relationships": "Family, relationships, dating, and marriage.",
    "politics_government": "Politics, government, law, elections, and military.",
}
YAHOO_FEATURE_ORDER = [
    "Society & Culture", "Science & Mathematics", "Health",
    "Education & Reference", "Computers & Internet", "Sports",
    "Business & Finance", "Entertainment & Music",
    "Family & Relationships", "Politics & Government",
]

SOURCES = {
    "yahoo_topics": {
        "hf_id": "community-datasets/yahoo_answers_topics",
        "split": "test",
        "count": 464,
        "question": Question(
            type="choice",
            instructions="Which topic does the question in the state belong to?",
            criteria=YAHOO_TOPICS,
        ),
    },
    "cola": {
        "hf_id": "nyu-mll/glue",
        "hf_config": "cola",
        "split": "validation",
        "count": 212,
        "question": Question(
            type="noul",
            instructions="Is the sentence in the state grammatically acceptable English?",
            criteria={
                "true": "The sentence is well-formed, grammatical English.",
                "false": "The sentence is ungrammatical or ill-formed.",
            },
        ),
    },
    "sms_spam": {
        "hf_id": "ucirvine/sms_spam",
        "split": "train",
        "count": 212,
        "question": Question(
            type="noul",
            instructions="Is the message in the state unsolicited spam?",
            criteria={
                "true": "Unsolicited promotional, scam, or bulk content.",
                "false": "An ordinary personal or expected message.",
            },
        ),
    },
    "app_reviews": {
        "hf_id": "sealuzh/app_reviews",
        "split": "train",
        "count": 160,
        "question": Question(
            type="score",
            instructions="How satisfied is the user in the state's app review?",
            criteria=[
                "Very dissatisfied; serious complaints or the app is unusable.",
                "Dissatisfied; significant problems outweigh the positives.",
                "Mixed; works but with notable issues or missing features.",
                "Satisfied; works well with minor complaints.",
                "Very satisfied; enthusiastic praise, no real complaints.",
            ],
        ),
    },
}


def one_hot(index: int, width: int) -> list[float]:
    if not 0 <= index < width:
        raise ValueError(f"label {index} outside {width} answers")
    return [1.0 if i == index else 0.0 for i in range(width)]


def rows_for(name: str, spec: dict, seed: int):
    """Yield (state, label_index, provenance) for one source, seeded order."""
    if spec.get("hf_config"):
        ds = load_dataset(spec["hf_id"], spec["hf_config"], split=spec["split"])
    else:
        ds = load_dataset(spec["hf_id"], split=spec["split"])
    if name == "yahoo_topics" and ds.features["topic"].names != YAHOO_FEATURE_ORDER:
        raise ValueError("yahoo_answers_topics label order changed upstream")
    if name == "cola" and ds.features["label"].names != ["unacceptable", "acceptable"]:
        raise ValueError("cola label order changed upstream")
    if name == "sms_spam" and ds.features["label"].names != ["ham", "spam"]:
        raise ValueError("sms_spam label order changed upstream")
    ds = ds.shuffle(seed=seed)
    for position, row in enumerate(ds):
        if name == "yahoo_topics":
            text = (row["question_title"].strip() + "\n" + row["question_content"].strip()).strip()
            label = row["topic"]
            extra = {"upstream_id": row["id"]}
        elif name == "cola":
            text, label, extra = row["sentence"].strip(), row["label"], {"upstream_idx": row["idx"]}
        elif name == "sms_spam":
            text, label, extra = row["sms"].strip(), row["label"], {}
        elif name == "app_reviews":
            text, label = row["review"].strip(), row["star"] - 1
            extra = {"package_name": row["package_name"]}
        else:
            raise ValueError(name)
        if len(text) < MIN_STATE_CHARS:
            continue
        yield text[:MAX_STATE_CHARS], label, {"shuffled_position": position, **extra}


def build(args) -> dict:
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError(f"output already exists: {out}; choose a new directory")
    reference_keys: set[str] = set()
    reference_files = {}
    for split in ("train", "development", "calibration", "test"):
        path = Path(args.reference) / f"{split}.jsonl"
        reference_files[str(path)] = sha256_file(path)
        for example in load_examples(path):
            reference_keys |= example.leakage_keys
    examples, dropped = [], {"leaked": 0, "duplicate": 0}
    seen_content: set[str] = set()
    for name, spec in SOURCES.items():
        kept = 0
        question = spec["question"]
        width = len(question.answer_keys)
        for text, label, extra in rows_for(name, spec, args.seed):
            example = Example(
                id=f"rv2:{name}:{extra.get('shuffled_position')}",
                group_id=f"rv2:{name}:{extra.get('shuffled_position')}",
                source=f"reserved-v2:{name}",
                state=text,
                question=question,
                target=one_hot(label, width),
                target_origin="gold",
                provenance={
                    "builder": "build_reserved_v2",
                    "hf_id": spec["hf_id"],
                    "hf_config": spec.get("hf_config"),
                    "hf_split": spec["split"],
                    "seed": args.seed,
                    "raw_label": label,
                    **extra,
                },
            )
            keys = example.leakage_keys
            if keys & reference_keys:
                dropped["leaked"] += 1
                continue
            content = next(k for k in keys if k.startswith("content:"))
            if content in seen_content:
                dropped["duplicate"] += 1
                continue
            seen_content.add(content)
            examples.append(example)
            kept += 1
            if kept == spec["count"]:
                break
        if kept < spec["count"]:
            raise ValueError(f"{name}: only {kept}/{spec['count']} rows survived filters")
    report = {
        "format_version": FORMAT_VERSION,
        "partition": "reserved test only; one pre-registered look (decision 29), spent afterward",
        "seed": args.seed,
        "sources": {
            name: {k: v for k, v in spec.items() if k != "question"}
            | {"instructions": spec["question"].instructions, "type": spec["question"].type}
            for name, spec in SOURCES.items()
        },
        "leakage_reference": reference_files,
        "dropped": dropped,
        "files": {},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".jev-rv2-", dir=out.parent))
    try:
        path = staging / "test.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(dumps(example.to_dict()) + "\n")
        report["files"]["test.jsonl"] = {
            "sha256": sha256_file(path), **partition_summary(examples),
        }
        (staging / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        out.mkdir()
        try:
            for item in staging.iterdir():
                item.rename(out / item.name)
        except BaseException:
            shutil.rmtree(out)
            raise
    finally:
        shutil.rmtree(staging)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reference", default="data/experiments/synthfiltered-corpus",
                        help="partition dir the new test must stay disjoint from")
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    try:
        report = build(args)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"build_reserved_v2: {exc}\n")
    print(json.dumps({"output": args.out_dir, "dropped": report["dropped"],
                      "files": report["files"]}, indent=2))


if __name__ == "__main__":
    main()
