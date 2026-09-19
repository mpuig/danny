"""Explicit admission and cache budgets. All lengths are token counts, not bytes."""

from dataclasses import dataclass, fields


class RequestValidationError(ValueError):
    pass


class RequestLimitError(RequestValidationError):
    pass


class InferenceDeadlineExceeded(TimeoutError):
    pass


@dataclass(frozen=True)
class EngineLimits:
    max_state_tokens: int = 2048
    max_prompt_tokens: int = 4096
    max_request_tokens: int = 1_000_000
    max_questions: int = 32
    max_options: int = 255
    max_views: int = 512
    max_prior_cache_entries: int = 128
    max_token_cache_entries: int = 64

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            minimum = 0 if "cache_entries" in field.name else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{field.name} must be an integer >= {minimum}")
        if self.max_options > 255:
            raise ValueError("max_options cannot exceed schema limit 255")
