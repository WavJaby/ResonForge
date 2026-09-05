"""Typed positions for locating generated-token validation evidence."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationPosition:
    """One generated-token position and its latest emitted musical shift."""

    token_index: int
    shift_value: int | None

    def __post_init__(self) -> None:
        if self.token_index < 0:
            raise ValueError("token_index must be non-negative")
        if self.shift_value is not None and self.shift_value < 0:
            raise ValueError("shift_value must be non-negative")
