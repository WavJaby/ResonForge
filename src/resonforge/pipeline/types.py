"""Shared configuration and state types for the audio pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..metadata_types import RunMetadata
from ..tempo_types import TempoResult

if TYPE_CHECKING:
    from .process_runner import ProcessRunner

SKIPPABLE_STEMS = frozenset(
    {"bass", "drums", "guitar", "piano", "other", "vocals", "band"}
)


@dataclass
class StemTask:
    """One separated or combined audio stem awaiting transcription."""

    name: str
    audio: Path
    instruments: str


@dataclass(frozen=True)
class PipelineConfig:
    """Validated CLI options consumed by the pipeline."""

    input_file: Path
    default_transcriber: str = "mt3.yptf_moe_multi"
    transcribers: tuple[tuple[str, str], ...] = ()
    bs_sample_rate: int = 44_100
    cfg_coef: float = 1.0
    beam_size: int = 1
    batch_size: int = 1
    other_instruments: str = ""
    adaptive_gate_presets: tuple[tuple[str, str], ...] = ()
    gate_presence_filter: bool = True
    gate_threshold_offset_db: float | None = None
    gate_threshold_dbfs: float | None = None
    hard_zero_threshold_dbfs: float = -80.0
    hard_zero_stems: frozenset[str] = frozenset()
    only_other: bool = False
    skip_stems: frozenset[str] = frozenset()
    combine_band: frozenset[str] = frozenset()
    mix_only: bool = False
    treat_original_as_other: bool = False
    mp3_out: bool = False
    midi_cleanup: bool = False
    silence_threshold_dbfs: float = -50.0
    mp3_bitrate: int = 128
    mix_jobs: int = 4
    preprocess_jobs: int = 4
    parallelism: int = 1
    gpus: tuple[int, ...] = ()

    @classmethod
    def from_namespace(cls, args: Any) -> PipelineConfig:
        """Copy supported fields from argparse or another attribute object."""
        values: dict[str, Any] = {}
        for field in fields(cls):
            if hasattr(args, field.name):
                values[field.name] = getattr(args, field.name)
        for name in (
            "hard_zero_stems",
            "skip_stems",
            "combine_band",
        ):
            if name in values:
                values[name] = frozenset(values[name])
        if "gpus" in values:
            values["gpus"] = tuple(values["gpus"])
        if "transcribers" in values:
            values["transcribers"] = tuple(sorted(values["transcribers"]))
        if "adaptive_gate_presets" in values:
            values["adaptive_gate_presets"] = tuple(
                sorted(values["adaptive_gate_presets"])
            )
        return cls(**values)

    def transcriber_for_stem(self, stem: str) -> str:
        """Return the selected backend/model spec for a stem."""
        return dict(self.transcribers).get(stem, self.default_transcriber)

    def gate_preset_for_stem(self, stem: str) -> str | None:
        """Return the adaptive gate preset selected for a stem."""
        return dict(self.adaptive_gate_presets).get(stem)


@dataclass
class PipelineContext:
    """Mutable run-scoped resources shared by pipeline phases."""

    config: PipelineConfig
    run_prefix: str
    metadata: RunMetadata
    metadata_path: Path
    runner: ProcessRunner
    output_dir: Path | None = None
    log_dir: Path | None = None
    source: Path | None = None
    base_name: str | None = None
    input_hash: str | None = None
    tempo_result: TempoResult | None = None
    staged_input_dir: Path | None = None


@dataclass(frozen=True)
class PipelineOutputs:
    """Final artifacts produced by a pipeline run."""

    midi: Path | None
    mp3: Path | None
