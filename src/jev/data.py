"""Canonical decision examples, integrity checks, and leakage-safe grouped splits.

The matching policy is conservative exact content after Unicode/whitespace/case
normalization, plus upstream group IDs. It is NOT fuzzy or pretraining decontamination.
This module deliberately has no MLX/datasets dependency.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from .rendering import LEGACY_V0, STRUCTURED_V1, LETTER_READOUT, render_views
from .schema import Question
from .serialization import State, dumps, loads, validate_json, validate_state

FORMAT_VERSION = 1
MATCHING_POLICY = "group+normalized-content-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Fail with path/line context, including for HTML and 404 download bodies."""
    count = 0
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid JSONL: {exc}") from exc
            count += 1
            yield row
    if count == 0:
        raise ValueError(f"{path}: empty JSONL dataset")


def normalize_target(target: list, size: int) -> list[float]:
    """Normalize near-unit rounded distributions, rejecting malformed supervision.

    At most 0.02 absolute sum error is accepted. This repairs two-decimal rounding,
    not arbitrary weights or missing probability mass. Retain the original vector.
    """
    if not isinstance(target, list) or len(target) != size or size < 1:
        raise ValueError(f"target must contain {size} probabilities")
    if any(type(p) not in (float, int) or not 0 <= p <= 1 or not math.isfinite(p) for p in target):
        raise ValueError("target probabilities must be finite numbers in [0, 1]")
    total = math.fsum(target)
    if total <= 0 or abs(total - 1) > 0.020000001:
        raise ValueError(f"target probabilities sum to {total}, not approximately 1")
    return [p / total for p in target]


def _normalized(value):
    if isinstance(value, str):
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    return value


def content_key(state: State) -> str:
    """Unwrap ONLY known one-document Kev wrappers for leakage matching.

    Rendering never unwraps or normalizes state. Arbitrary multi-field objects retain
    all structure; shared short leaf values must not merge unrelated documents.
    """
    validate_state(state)
    if isinstance(state, dict) and set(state) == {"document"} and isinstance(state["document"], str):
        state = state["document"]
    elif isinstance(state, dict) and set(state) == {"ticket"}:
        ticket = state["ticket"]
        if isinstance(ticket, dict) and set(ticket) == {"channel", "body"} and isinstance(ticket["body"], str):
            state = ticket["body"]
    elif isinstance(state, list) and len(state) == 1 and isinstance(state[0], dict):
        message = state[0]
        if set(message) == {"role", "content"} and isinstance(message["content"], str):
            state = message["content"]
    text = dumps(_normalized(state), sort_keys=True)
    return "content:" + hashlib.sha256(text.encode()).hexdigest()


@dataclass
class Example:
    id: str
    group_id: str
    source: str
    state: State
    question: Question
    target: list[float]
    target_origin: str
    provenance: dict
    original_target: list[float] | None = None
    format_version: int = FORMAT_VERSION

    def __post_init__(self):
        if type(self.format_version) is not int or self.format_version != FORMAT_VERSION:
            raise ValueError(f"unsupported example format: {self.format_version!r}")
        for field in ("id", "group_id", "source", "target_origin"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"{field} must be a nonempty string")
        if not isinstance(self.question, Question):
            raise ValueError("question must be a Question")
        validate_state(self.state)
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be an object")
        validate_json(self.provenance)
        original = self.target if self.original_target is None else self.original_target
        self.target = normalize_target(self.target, len(self.question.answer_keys))
        normalized_original = normalize_target(original, len(self.target))
        if any(abs(a - b) > 1e-10 for a, b in zip(self.target, normalized_original)):
            raise ValueError("original_target and target disagree after normalization")
        self.original_target = list(original)

    @property
    def leakage_keys(self) -> set[str]:
        return {"example:" + self.id, "group:" + self.group_id, content_key(self.state)}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> Example:
        row = dict(row)
        if row.get("format_version") != FORMAT_VERSION:
            raise ValueError("canonical example requires format_version=1")
        question = row.get("question")
        if not isinstance(question, dict):
            raise ValueError("question must be an object")
        row["question"] = Question(**question)
        return cls(**row)


def load_examples(path: str | Path) -> list[Example]:
    examples = []
    seen = set()
    for number, row in enumerate(read_jsonl(path), 1):
        try:
            example = Example.from_dict(row)
            if example.id in seen:
                raise ValueError(f"duplicate example id: {example.id}")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:record {number}: {exc}") from exc
        seen.add(example.id)
        examples.append(example)
    return examples


def kev_examples(rows: list[dict], input_hash: str, *, max_options: int = 26) -> tuple[list[Example], dict]:
    """Import Kev without discarding state, rubric, grouping, or source provenance."""
    if not 1 <= max_options <= 255:
        raise ValueError("max_options must be between 1 and 255")
    output = []
    skipped = Counter()
    seen = set()
    for row_number, row in enumerate(rows, 1):
        try:
            meta = row["_meta"]
            for name in ("id", "group_id", "source", "repo", "revision", "split", "variant"):
                if not isinstance(meta.get(name), str) or not meta[name]:
                    raise ValueError(f"missing Kev metadata: {name}")
            state = row["state"]
            validate_state(state)
            if not isinstance(row["questions"], dict) or not row["questions"]:
                raise ValueError("questions must be a nonempty object")
            for qid, spec in row["questions"].items():
                question = Question(
                    type=spec["type"], instructions=spec["instructions"], criteria=spec.get("criteria")
                )
                source = spec["src"]
                if not isinstance(source, str) or not source:
                    raise ValueError("question src must be a nonempty string")
                label = spec["label"]
                if question.type == "choice":
                    if not isinstance(label, str) or label not in question.criteria:
                        raise ValueError(f"unknown choice label: {label!r}")
                    index = question.answer_keys.index(label)
                elif question.type == "noul":
                    if type(label) is not bool:
                        raise ValueError("Kev Noul label must be boolean")
                    index = int(label)
                else:
                    if type(label) is not int or not 0 <= label < len(question.criteria):
                        raise ValueError(f"invalid Score label: {label!r}")
                    index = label
                if source == "sst5" or meta["source"] == "sst5":
                    skipped["sst5"] += 1
                    continue
                if question.type == "choice" and len(question.criteria) > max_options:
                    skipped["wide_choice"] += 1
                    continue
                identity = dumps([meta["repo"], meta["id"], meta["variant"], qid])
                eid = "kev:" + hashlib.sha256(identity.encode()).hexdigest()
                if eid in seen:
                    raise ValueError(f"duplicate Kev question identity: {identity}")
                seen.add(eid)
                target = [float(i == index) for i in range(len(question.answer_keys))]
                output.append(Example(
                    id=eid, group_id=f"kev:{meta['repo']}:{meta['group_id']}",
                    source=f"kev:{source}", state=state, question=question, target=target,
                    target_origin="gold",
                    provenance={"upstream": meta, "question_id": qid, "input_sha256": input_hash},
                ))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Kev record {row_number}: {exc}") from exc
    return output, dict(skipped)


def verify_kev_file(path: str | Path, manifest: dict, split: str) -> tuple[list[dict], str]:
    entry = manifest["files"][f"{split}.jsonl"]
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise ValueError(f"{path}: SHA-256 does not match manifest for {split}")
    rows = list(read_jsonl(path))
    if sha256_file(path) != actual:
        raise ValueError(f"{path}: input changed while loading")
    if len(rows) != entry["records"]:
        raise ValueError(f"{path}: record count does not match manifest")
    if sum(len(row.get("questions", {})) for row in rows) != entry["questions"]:
        raise ValueError(f"{path}: question count does not match manifest")
    return rows, actual


def grouped_indices(examples: list[Example]) -> list[list[int]]:
    """Connected components: group IDs and content overlaps are transitive."""
    parents = list(range(len(examples)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    keys = {}
    for i, example in enumerate(examples):
        for key in sorted(example.leakage_keys):
            if key in keys:
                parents[find(i)] = find(keys[key])
            else:
                keys[key] = i
    groups = defaultdict(list)
    for i in range(len(examples)):
        groups[find(i)].append(i)
    return list(groups.values())


def assert_disjoint(partitions: dict[str, list[Example]]) -> None:
    seen = {}
    for split, examples in partitions.items():
        for example in examples:
            for key in example.leakage_keys:
                if key in seen and seen[key] != split:
                    raise ValueError(f"leakage between {seen[key]} and {split}: {key}")
                seen[key] = split


def split_examples(
    train: list[Example], test: list[Example], *, seed: int = 42,
    development_fraction: float = 0.1, calibration_fraction: float = 0.1,
) -> tuple[dict[str, list[Example]], list[str]]:
    """Reserve supplied test groups first; split remaining training components.

    Fractions apply to connected groups, NOT rows/questions. Test overlaps remove
    entire training components. No test case moves into training or calibration.
    """
    if not (0 < development_fraction < 1 and 0 < calibration_fraction < 1
            and development_fraction + calibration_fraction < 1):
        raise ValueError("development/calibration fractions must be positive and sum to less than 1")
    combined = train + test
    groups, removed = [], []
    for group in grouped_indices(combined):
        training = [combined[i] for i in group if i < len(train)]
        if any(i >= len(train) for i in group):
            removed.extend(example.id for example in training)
        elif training:
            groups.append(sorted(training, key=lambda example: example.id))
    groups.sort(key=lambda group: tuple(example.id for example in group))
    random.Random(seed).shuffle(groups)
    n_dev = max(1, round(len(groups) * development_fraction))
    n_cal = max(1, round(len(groups) * calibration_fraction))
    if n_dev + n_cal >= len(groups):
        raise ValueError("not enough independent training groups for nonempty train/dev/calibration splits")
    partitions = {
        "development": [e for group in groups[:n_dev] for e in group],
        "calibration": [e for group in groups[n_dev:n_dev+n_cal] for e in group],
        "train": [e for group in groups[n_dev+n_cal:] for e in group],
        "test": list(test),
    }
    if not partitions["test"]:
        raise ValueError("test partition is empty")
    for examples in partitions.values():
        examples.sort(key=lambda example: example.id)
    assert_disjoint(partitions)
    return partitions, sorted(removed)


def partition_summary(examples: list[Example]) -> dict:
    return {
        "examples": len(examples),
        "groups": len(grouped_indices(examples)),
        "sources": dict(sorted(Counter(e.source for e in examples).items())),
        "primitives": dict(sorted(Counter(e.question.type for e in examples).items())),
    }


def load_training_rows(path: str | Path, renderer_version: str | None = None,
                       readout_version: str = LETTER_READOUT) -> list[dict]:
    """Render canonical records or validate historical pre-rendered v0 files.

    Do not silently mix formats. Legacy content extraction is best-effort provenance,
    not recovery of the original structured state or group identifiers.
    """
    rows = list(read_jsonl(path))
    canonical = ["format_version" in row for row in rows]
    if any(canonical) and not all(canonical):
        raise ValueError(f"{path}: mixed canonical and legacy examples")
    output = []
    if all(canonical):
        version = renderer_version or STRUCTURED_V1
        for example in load_examples(path):
            views = render_views(example.state, example.question, version, readout_version)
            for index, (prompt, labels) in enumerate(views):
                target = (example.target if readout_version == LETTER_READOUT or example.question.type == "noul"
                          else [1 - example.target[index], example.target[index]])
                output.append({
                    "prompt": prompt, "labels": labels, "target": target,
                    "task": example.source, "id": f"{example.id}/view/{index}", "example_id": example.id,
                    "renderer_version": version, "readout_version": readout_version,
                    "weight": 1 / len(views), "leakage_keys": example.leakage_keys,
                    # joint distribution over ordered levels; eligible for ordinal losses
                    "ordinal": example.question.type == "score" and readout_version == LETTER_READOUT,
                })
    else:
        if readout_version != LETTER_READOUT:
            raise ValueError("candidate readout requires canonical data, not legacy prompts")
        if renderer_version not in (None, LEGACY_V0):
            raise ValueError(f"{path}: pre-rendered legacy data requires {LEGACY_V0}")
        for number, row in enumerate(rows, 1):
            try:
                prompt, labels = row["prompt"], row["labels"]
                if not isinstance(prompt, str) or not isinstance(labels, list) or not labels:
                    raise ValueError("expected prompt string and nonempty labels list")
                if any(not isinstance(label, str) for label in labels) or len(set(labels)) != len(labels):
                    raise ValueError("labels must be distinct strings")
                # Only this repository's historical prompt shape is supported.
                state = prompt.split("State:\n", 1)[1].rsplit("\n\nQuestion:", 1)[0]
                if "\n\nQuestion:" not in prompt:
                    raise ValueError("cannot identify state in legacy prompt")
                output.append({
                    **row, "target": normalize_target(row["target"], len(labels)),
                    "id": f"{path}:{number}", "renderer_version": LEGACY_V0,
                    "readout_version": LETTER_READOUT, "weight": 1.0, "ordinal": False,
                    "leakage_keys": {content_key(state)},
                })
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:record {number}: invalid legacy training row: {exc}") from exc
    return output


def assert_training_disjoint(train: list[dict], validation: list[dict]) -> None:
    keys = set().union(*(row["leakage_keys"] for row in train))
    overlapping = [row["id"] for row in validation if keys & row["leakage_keys"]]
    if overlapping:
        raise ValueError(
            f"training/validation leakage: {len(overlapping)} validation rows overlap; "
            f"examples: {overlapping[:3]}"
        )
