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
    #: Which second attempt a triggered chunk gets beside the shifted replay.
    #: `secondary_model` escalates to a larger model; `new_seed` resamples the
    #: same one. Every other candidate escalates size, so this is the only
    #: dial that asks whether a bad chunk was unlucky rather than under-served
    #: -- and the cheap answer, since it loads no second model.
    secondary_candidate: str = "secondary_model"

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
        from resonforge.transcribers.muscriptor.quality.policy.generation_anomaly import (
            DEFAULT_ANOMALY_CONFIG,
        )
        from resonforge.transcribers.muscriptor.quality.policy.overlap import (
            OverlapWindow,
        )
        from resonforge.transcribers.muscriptor.quality.policy.recovery_runtime import (
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
            beam_size=1,
            forbidden_tokens=request.forbidden_tokens,
            overlap_window=OverlapWindow(
                replay_seconds=self.replay_seconds,
                verification_seconds=self.min_verify_seconds,
            ),
            anomaly_detection=self.anomaly_detection,
            overlap_detection=self.overlap_detection,
            recovery=self.recovery,
            recovery_shift_seconds=self.recovery_shift_seconds,
            recovery_selection="checkpoint",
            secondary_recovery_shifted=False,
            anomaly_config=DEFAULT_ANOMALY_CONFIG,
            recovery_seed=self.recovery_seed,
            stdout_logger=request.stdout_logger,
            stderr_logger=request.stderr_logger,
            checkpoint_generation=self.checkpoint_generation,
            trace_collector=self.trace_collector,
            trace_context_prefix=self.trace_context_prefix,
            first_chunk_model=self.first_chunk_model,
            fresh_reanchor_model=self.fresh_reanchor_model,
            chunk_quality_history=self.chunk_quality_history,
            hard_pitch_envelope=self.hard_pitch_envelope,
            overlap_probe_enabled=self.overlap_probe,
            fresh_reanchor=self.fresh_reanchor,
            secondary_candidate=self.secondary_candidate,
        )
