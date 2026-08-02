"""JSON-compatible tempo analysis contracts."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class MuscriptorTempoReport(TypedDict):
    stem: str
    method: str
    backend: str
    model: str
    status: str
    reason: NotRequired[str]
    bpm: NotRequired[float]
    candidate_bpm: NotRequired[float]
    beats_per_bar: NotRequired[int]
    first_downbeat: NotRequired[float]
    bar_offset: NotRequired[float]
    bar_seconds: NotRequired[float | None]

    residual_seconds: NotRequired[float]
    beat_count: NotRequired[int]
    downbeat_count: NotRequired[int]
    coverage_ratio: NotRequired[float]
    meter_agreement: NotRequired[float | None]
    outlier_fraction: NotRequired[float]


class TempoResult(TypedDict):
    method: str
    selected_bpm: float
    status: NotRequired[str]
    selected_source: NotRequired[str]
    sources_used: NotRequired[list[str]]
    all_reports: NotRequired[list[MuscriptorTempoReport]]
    beat_times_seconds: NotRequired[list[float]]
    beats_per_bar: NotRequired[int]
    first_downbeat: NotRequired[float]
    bar_seconds: NotRequired[float]
    bar_phase_seconds: NotRequired[float]
    bar_offset: NotRequired[float]
    tempo_source: NotRequired[str]
