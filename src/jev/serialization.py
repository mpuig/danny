"""JSON boundaries shared by data ingestion, rendering, and HTTP serving.

No MLX or model imports: data tooling and contract tests run without an accelerator.
"""

from __future__ import annotations

import json
import math
from typing import TypeAlias, Union

JSONValue: TypeAlias = Union[
    str, int, float, bool, None, list["JSONValue"], dict[str, "JSONValue"]
]
Entry: TypeAlias = Union[str, list[JSONValue], dict[str, JSONValue], None]
State: TypeAlias = Union[str, list[JSONValue], dict[str, JSONValue]]


def validate_json(value: object, *, path: str = "value", depth: int = 0) -> None:
    """Reject non-JSON values, nonfinite numbers, cycles, and excessive nesting."""
    if depth > 64:
        raise ValueError(f"{path}: JSON nesting exceeds 64 levels (or contains a cycle)")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path}: numbers must be finite")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            validate_json(item, path=f"{path}[{i}]", depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: object keys must be strings")
            validate_json(item, path=f"{path}.{key}", depth=depth + 1)
        return
    raise ValueError(f"{path}: unsupported JSON value {type(value).__name__}")


def validate_entry(value: object, *, path: str) -> None:
    if value is not None and not isinstance(value, (str, dict, list)):
        raise ValueError(f"{path}: expected string, object, array, or null")
    validate_json(value, path=path)


def validate_state(value: object) -> None:
    if not isinstance(value, (str, dict, list)):
        raise ValueError("state must be a string, object, or array")
    validate_json(value, path="state")


def dumps(value: object, *, sort_keys: bool = False) -> str:
    """Lossless JSON, including string boundaries; preserves option order by default."""
    validate_json(value)
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        sort_keys=sort_keys,
    )


def loads(text: str | bytes) -> JSONValue:
    """Unlike default json.loads, reject duplicate keys and NaN/Infinity."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
        validate_json(value)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds supported depth") from exc
    return value
