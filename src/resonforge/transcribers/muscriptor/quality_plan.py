"""The chunk schedule and token stream ResonForge hands MuScriptor.

`TranscriptionModel.transcribe` owns audio, chunking and note decoding. What
runs between them -- overlap replay, verification, recovery candidates, the
guards -- is a *plan*, supplied by the caller. This module builds ours.

Why the direction is this way round: every one of those is a policy about
output quality, and the scheduler that executes it lives here, not in the
model. Passing them as twenty-odd keyword arguments meant `transcribe`
carried the whole vocabulary of a subsystem it does not run; one object
carries it instead, and a caller with no plan gets the model's own behaviour.

The preconditions below moved out of `transcribe` with the parameters they
constrain. Two of its old checks are gone rather than moved -- "recovery
requires prelude_forcing and batch_size 1" -- because a plan cannot be
scheduled any other way: `transcribe` refuses a plan beside a wider batch.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

_SEGMENT_DURATION = 5.0
#: `resolve_overlap_window` widens verification backwards while it hunts for
#: evidence and stops here, so the ceiling -- not the requested minimum -- is
#: what has to fit inside one segment.
_MAXIMUM_VERIFY_SECONDS = 0.8


@dataclass(frozen=True)
class MuscriptorQualityPlan:
    """Overlap replay, recovery and guard policy for one region's chunks."""

    anomaly_detection: bool
    overlap_detection: bool
    recovery: bool
    recovery_shift_seconds: float
    recovery_seed: int
    replay_seconds: float
    min_verify_seconds: float
    checkpoint_generation: bool
    fresh_reanchor: bool
    first_chunk_model: str | None
    fresh_reanchor_model: str | None
    chunk_quality_history: Any
    hard_pitch_envelope: Any
    trace_collector: Any
    trace_context_prefix: tuple[object, ...]
    overlap_probe: bool

    def __post_init__(self) -> None:
        if self.recovery_seed < 0:
            raise ValueError("recovery_seed must be non-negative")
        if not 0 <= self.recovery_shift_seconds < _SEGMENT_DURATION:
            raise ValueError("recovery shift must be in [0, 5)")
        if self.replay_seconds <= 0 or self.min_verify_seconds <= 0:
            raise ValueError("a plan requires positive replay and verification")
        ceiling = max(self.min_verify_seconds, _MAXIMUM_VERIFY_SECONDS)
        if self.replay_seconds + ceiling >= _SEGMENT_DURATION:
            raise ValueError("replay plus maximum verification must be under 5s")
        if self.recovery_shift_seconds + self.overlap_seconds > _SEGMENT_DURATION:
            raise ValueError(
                "recovery shift plus overlap must fit the five-second condition"
            )

    @property
    def overlap_seconds(self) -> float:
        return self.replay_seconds + self.min_verify_seconds

    @property
    def stride_seconds(self) -> float:
        """Chunks advance by less than a segment; the rest is replayed."""
        return _SEGMENT_DURATION - self.overlap_seconds

    def stream(self, model: object, request: Any) -> Iterator[object]:
        """Yield this region's tokens, boundaries and diagnostics."""
        from resonforge.transcribers.muscriptor.quality.generation_anomaly import (
            DEFAULT_ANOMALY_CONFIG,
        )
        from resonforge.transcribers.muscriptor.quality.overlap import OverlapWindow
        from resonforge.transcribers.muscriptor.quality.recovery_runtime import (
            forcing_stream,
        )

        return forcing_stream(
            model,
            None,
            request.seek_times,
            request.wav,
            request.instrument_group,
            request.max_gen_len,
            request.use_sampling,
            request.temperature,
            request.cfg_coef,
            request.no_eos_is_ok,
            1,
            request.forbidden_tokens,
            OverlapWindow(
                replay_seconds=self.replay_seconds,
                verification_seconds=self.min_verify_seconds,
            ),
            self.anomaly_detection,
            self.overlap_detection,
            self.recovery,
            self.recovery_shift_seconds,
            "checkpoint",
            False,
            DEFAULT_ANOMALY_CONFIG,
            self.recovery_seed,
            request.stdout_logger,
            request.stderr_logger,
            # The scheduler owns every forward, so generation is always
            # deferred here -- a row is never run inline by the producer.
            True,
            self.checkpoint_generation,
            self.trace_collector,
            self.trace_context_prefix,
            self.first_chunk_model,
            self.fresh_reanchor_model,
            self.chunk_quality_history,
            self.hard_pitch_envelope,
            self.overlap_probe,
            fresh_reanchor=self.fresh_reanchor,
        )
