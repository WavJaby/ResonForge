"""What the decode loop needs from a row's guard, and nothing it does not.

The guard runs *inside* the executor: an interrupt has to land at token
granularity, not when the row finishes, so the policy cannot run it from
the outside. This is the surface the executor drives it through -- feed
tokens, get an action, collect what it found -- with no detector, mode or
role named, so `runtime/` imports one protocol and the policy behind it
stays free to change.

Actions, findings and the summary live here too: they ride back to the
policy on `GenerationResult`, which makes them part of the contract rather
than of the detectors that produce them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from resonforge.transcribers.muscriptor.quality.generation_position import (
    GenerationPosition,
)

GuardStatus = Literal["clear", "warning", "critical"]
GuardAction = Literal[
    "continue",
    "pause_for_verify",
    "interrupt_to_recovery",
    "reject_candidate",
    "reject_terminal",
    "accept_candidate",
    "request_recovery",
]


@dataclass(frozen=True)
class GuardFinding:
    detector: str
    status: GuardStatus
    reason: str
    metrics: dict[str, float | int | bool | str | None] = field(
        default_factory=dict
    )
    observed_at: GenerationPosition | None = None
    suspected_start: GenerationPosition | None = None
    last_safe_frontier: GenerationPosition | None = None


@dataclass(frozen=True)
class AnomalyMetrics:
    generated_tokens: int
    window_tokens: int
    repeated_4gram_rate: float
    repeated_8gram_rate: float
    repeated_16gram_rate: float
    unique_token_rate: float
    gzip_ratio: float
    tokens_since_shift: int
    score: float
    provisional_gzip_thresholds: bool = True
    late_missing_eos_available: bool = False
    same_pitch_retrigger_available: bool = False


@dataclass(frozen=True)
class MonitorSummary:
    generated_tokens: int
    emitted_eos: bool
    trigger_reason: str | None
    last_metrics: AnomalyMetrics | None
    poll_metrics: tuple[AnomalyMetrics, ...]
    periodic_window_streak: int = 0


class RowGuard(Protocol):
    """One row's guard, driven by the decode loop.

    Stateful across the row's life: a row that is paused and resumed keeps
    the same guard, which is why a request carries a factory for one rather
    than an instance.
    """

    findings: Sequence[GuardFinding]

    def observe(self, tokens: Sequence[int]) -> GuardAction:
        """Account for generated tokens (never prompt, never EOS)."""
        ...

    def complete(self, *, emitted_eos: bool, reached_max_length: bool) -> GuardAction:
        ...

    def summary(self, emitted_eos: bool) -> MonitorSummary | None:
        ...
