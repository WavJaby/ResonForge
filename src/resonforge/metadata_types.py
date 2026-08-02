"""Run metadata schema shared by pipeline phases."""

from __future__ import annotations

from typing import Any, TypedDict

from .tempo_types import TempoResult


class AudioFileMetadata(TypedDict, total=False):
    path: str
    size_bytes: int
    extension: str
    sha256: str
    duration_seconds: float
    sample_rate: int
    channels: int
    frames: int
    format: str
    subtype: str


class StemMetadata(TypedDict, total=False):
    audio: str
    instruments: str
    selected_for_transcription: bool
    selection_reason: str
    adaptive_gate: dict[str, object]
    hard_zero: dict[str, object]
    presence: dict[str, object]
    presence_metrics: dict[str, object]
    transcription: dict[str, Any]
    local_cleanup: dict[str, Any]
    global_cleanup: dict[str, Any]


class OutputMetadata(TypedDict, total=False):
    directory: str
    metadata: str
    logs: dict[str, str]
    final_midi: str
    final_midi_bpm: float
    final_mp3: str


class RunMetadata(TypedDict, total=False):
    schema_version: int
    run_id: str
    status: str
    started_at: str
    finished_at: str | None
    elapsed_seconds: float | None
    command: dict[str, Any]
    input: AudioFileMetadata
    software: dict[str, Any]
    models: dict[str, Any]
    separation: dict[str, Any]
    stems: dict[str, StemMetadata]
    transcription_timing: dict[str, Any]
    tempo: TempoResult | None
    outputs: OutputMetadata
    error: dict[str, str] | None
