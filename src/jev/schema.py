"""Validated JSON question/answer types for the System One API subset.

State, instructions, and descriptions can be structured JSON. Readout-specific
limits (currently 26 Choice options) are enforced by the renderer, not this schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .serialization import Entry, validate_entry


@dataclass
class Question:
    """One typed question evaluated against a state.

    type: "choice" | "score" | "noul"
    criteria:
      - choice: dict of option name -> description (description may be None)
      - score:  ordered list of level descriptions, low to high (2-10 levels)
      - noul:   optional dict with "true"/"false" descriptions
    """

    type: str
    instructions: Entry
    criteria: dict[str, Entry] | list[Entry] | None = None

    def __post_init__(self) -> None:
        if self.type not in ("choice", "score", "noul"):
            raise ValueError(f"unknown question type: {self.type!r}")
        validate_entry(self.instructions, path="instructions")
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not 1 <= len(self.criteria) <= 255:
                raise ValueError("choice requires a criteria object with 1-255 options")
            if any(not isinstance(name, str) or not name for name in self.criteria):
                raise ValueError("choice option names must be nonempty strings")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
                raise ValueError("score requires criteria: ordered list of 2-10 levels")
        elif self.criteria is not None:
            if not isinstance(self.criteria, dict) or set(self.criteria) - {"true", "false"}:
                raise ValueError("noul criteria must contain only true/false descriptions")
        if self.criteria is not None:
            entries = self.criteria.items() if isinstance(self.criteria, dict) else enumerate(self.criteria)
            for key, value in entries:
                validate_entry(value, path=f"criteria.{key}")

    @property
    def answer_keys(self) -> list[str]:
        """Target/probability order; independent of the model's readout tokens."""
        if self.type == "noul":
            return ["no", "yes"]
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria))]
        return list(self.criteria)


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    type: str = "choice"


@dataclass
class ScoreAnswer:
    score: float
    probabilities: dict[str, float]  # keyed by level number as string
    legend: dict[str, Entry] = field(default_factory=dict)
    confidence: float = 0.0
    type: str = "score"


@dataclass
class NoulAnswer:
    noul: float  # P(yes), 0..1
    type: str = "noul"


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer
