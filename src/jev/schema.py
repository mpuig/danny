"""Request/response types mirroring the Typesafe System One API.

A request is a `state` (text) plus a dict of questions keyed by ID.
Answers carry typed values and probability distributions, never text.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    instructions: str
    criteria: dict | list | None = None

    def __post_init__(self) -> None:
        if self.type not in ("choice", "score", "noul"):
            raise ValueError(f"unknown question type: {self.type!r}")
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not self.criteria:
                raise ValueError("choice requires criteria: dict of option -> description")
            if len(self.criteria) > 26:
                raise ValueError("v0 readout supports at most 26 options")
        if self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
                raise ValueError("score requires criteria: ordered list of 2-10 levels")


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
    legend: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    type: str = "score"


@dataclass
class NoulAnswer:
    noul: float  # P(yes), 0..1
    type: str = "noul"


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer
