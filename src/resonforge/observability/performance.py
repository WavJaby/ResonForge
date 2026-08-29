"""Workload-normalized pipeline throughput metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PerformanceMetrics:
    audio_seconds: float
    wall_seconds: float
    output_notes: int
    generated_tokens: int
    notes_per_second: float
    generated_tokens_per_second: float
    audio_seconds_per_wall_second: float
    realtime_factor: float
    processing_seconds_per_audio_minute: float
    # Time inside the transcription queue, and the realtime figure derived from
    # it. Separated from the end-to-end numbers because those move with
    # separation-cache state, which is not a transcription signal — the refactor
    # gates are stated on `transcription_realtime` for exactly that reason.
    # `None` means unknown, never zero: runs recorded before this field existed
    # simply do not have it.
    transcription_seconds: float | None = None
    transcription_realtime: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def calculate_performance(
    *,
    audio_seconds: float,
    wall_seconds: float,
    output_notes: int,
    generated_tokens: int,
    transcription_seconds: float | None = None,
) -> PerformanceMetrics | None:
    """Normalize completed work by wall time and source-audio duration."""
    if audio_seconds <= 0 or wall_seconds <= 0:
        return None
    return PerformanceMetrics(
        audio_seconds=round(audio_seconds, 6),
        wall_seconds=round(wall_seconds, 6),
        output_notes=output_notes,
        generated_tokens=generated_tokens,
        notes_per_second=round(output_notes / wall_seconds, 6),
        generated_tokens_per_second=round(generated_tokens / wall_seconds, 6),
        audio_seconds_per_wall_second=round(audio_seconds / wall_seconds, 6),
        realtime_factor=round(wall_seconds / audio_seconds, 6),
        processing_seconds_per_audio_minute=round(
            wall_seconds * 60 / audio_seconds,
            6,
        ),
        transcription_seconds=(
            round(transcription_seconds, 6)
            if transcription_seconds and transcription_seconds > 0
            else None
        ),
        transcription_realtime=(
            round(audio_seconds / transcription_seconds, 6)
            if transcription_seconds and transcription_seconds > 0
            else None
        ),
    )


def aggregate_performance(
    metrics: Iterable[PerformanceMetrics | None],
    *,
    wall_seconds: float,
) -> PerformanceMetrics | None:
    """Measure batch throughput using shared wall time, not summed job time."""
    completed = tuple(metric for metric in metrics if metric is not None)
    known = tuple(
        metric.transcription_seconds
        for metric in completed
        if metric.transcription_seconds is not None
    )
    return calculate_performance(
        audio_seconds=sum(metric.audio_seconds for metric in completed),
        wall_seconds=wall_seconds,
        output_notes=sum(metric.output_notes for metric in completed),
        generated_tokens=sum(metric.generated_tokens for metric in completed),
        # Summed, not shared: songs in a batch queue their transcription
        # sequentially against one scheduler, so the batch's queue time is the
        # total, unlike wall time. Partial knowledge is no knowledge here — one
        # song missing the field would understate the divisor and overstate the
        # rate.
        transcription_seconds=sum(known) if len(known) == len(completed) else None,
    )
