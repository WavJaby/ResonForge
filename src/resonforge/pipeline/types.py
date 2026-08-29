"""Shared configuration and state types for the audio pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..midi.document import MidiDocument
from ..midi.memory_cleanup import AudioCleanupFeatures
from ..observability.metadata_types import RunMetadata
from ..observability.performance import PerformanceMetrics
from ..observability.tempo_types import MuscriptorTempoReport, TempoResult

if TYPE_CHECKING:
    from ..observability.artifacts import RunArtifactRegistry
    from ..runtime.process_runner import ProcessRunner

SKIPPABLE_STEMS = frozenset(
    {"bass", "drums", "guitar", "piano", "other", "vocals", "band"}
)



@dataclass(frozen=True)
class PipelineConfig:
    """Validated CLI options consumed by the pipeline."""

    input_file: Path
    # Where this run's song directory is created. Defaults to the repository
    # `output/`, which mixes every experiment's runs and separation caches into
    # one flat namespace keyed only by song name; point it somewhere per-arm to
    # keep an experiment's output together and disposable on its own.
    output_root: Path | None = None
    default_transcriber: str = "muscriptor.small"
    transcribers: tuple[tuple[str, str], ...] = ()
    bs_sample_rate: int = 44_100
    batch_size: int = 0
    # Per-model forced widths from `--batch-size small=8,medium=2`. The
    # uniform `batch_size` above cannot express an asymmetric vector, and an
    # asymmetric vector is the only kind that fits once two models share a
    # device: on an 11 GiB card every symmetric width above 3 exceeds the
    # solve budget while `small=8 medium=2` fits inside it.
    batch_size_by_model: tuple[tuple[str, int], ...] = ()
    # Backend-specific settings the service carries but never interprets. The
    # backend declares its own fields and reads them through `options_from`;
    # the pipeline only reports this mapping into metadata and the startup log,
    # which is why a dict beats named fields — a second backend adds keys here
    # and changes no pipeline code at all.
    backend_options: Mapping[str, Any] = field(default_factory=dict)
    # Extra evidence this run should record. Every member only *records*: none
    # changes what is generated and none writes durable state, which is what
    # makes "capture everything" a safe thing to ask for.
    debug_capture: frozenset[str] = frozenset()
    # Durable activation export for training another model. Deliberately not a
    # `debug_capture` member: that set promises to write no durable state, and
    # this exists to write some. It needs no runtime downgrade -- since G1-1b
    # the reduction runs inside the CUDA Graph capture.
    export_activations: bool = False
    log_retention: str = "errors"
    gpu_telemetry_interval_ms: int = 0
    other_instruments: str = ""
    gate_presence_filter: bool = True
    only_other: bool = False
    skip_stems: frozenset[str] = frozenset()
    combine_band: frozenset[str] = frozenset()
    treat_original_as_other: bool = False
    mp3_out: bool = False
    midi_cleanup: bool = True
    silence_threshold_dbfs: float = -50.0
    mp3_bitrate: int = 128
    mix_jobs: int = 4
    preprocess_jobs: int = 4
    parallelism: int = 1
    host_memory_budget_mib: int = 4096
    gpus: tuple[int, ...] = ()
    publish_files: bool = True

    @classmethod
    def from_namespace(cls, args: Any) -> PipelineConfig:
        """Copy supported fields from argparse or another attribute object."""
        values: dict[str, Any] = {}
        for config_field in fields(cls):
            if hasattr(args, config_field.name):
                values[config_field.name] = getattr(args, config_field.name)
        for name in (
            "skip_stems",
            "combine_band",
        ):
            if name in values:
                values[name] = frozenset(values[name])
        # `--batch-size` parses to one width or to per-model widths through a
        # single option. Splitting the two forms here keeps every downstream
        # reader of `batch_size` an int, which is what all of them assume.
        batch_size = values.get("batch_size")
        if isinstance(batch_size, tuple):
            values["batch_size_by_model"] = batch_size
            values["batch_size"] = 0
        if "gpus" in values:
            values["gpus"] = tuple(values["gpus"])
        if "transcribers" in values:
            values["transcribers"] = tuple(sorted(values["transcribers"]))
        return cls(**values)

    def transcriber_for_stem(self, stem: str) -> str:
        """Return the selected backend/model spec for a stem."""
        return dict(self.transcribers).get(stem, self.default_transcriber)


@dataclass
class PipelineContext:
    """Mutable run-scoped resources shared by pipeline phases."""

    config: PipelineConfig
    run_prefix: str
    metadata: RunMetadata
    metadata_path: Path
    runner: ProcessRunner
    # How many jobs the *service* runs at once. The capacity solver divides the
    # device transient budget by it, so it has to be the host's real
    # concurrency, not something the request declares: a server accepting one
    # request at a time would otherwise send `1` on every job while N ran
    # together, and the divisor would be wrong exactly when it matters (D1-1).
    device_job_slots: int = 1
    output_dir: Path | None = None
    song_dir: Path | None = None
    log_dir: Path | None = None
    source: Path | None = None
    base_name: str | None = None
    input_hash: str | None = None
    tempo_result: TempoResult | None = None
    staged_input_dir: Path | None = None
    artifacts: RunArtifactRegistry | None = None
    transient_output_dir: bool = False
    audio_features: dict[str, AudioCleanupFeatures] = field(default_factory=dict)
    tempo_reports: dict[str, MuscriptorTempoReport] = field(default_factory=dict)


@dataclass(frozen=True)
class InitializedRun:
    """A run whose input and output locations are resolved.

    Exists so the phases cannot be handed a half-built run. `PipelineContext`
    keeps its optional fields because it is constructed before initialization
    and the failure path needs it to report a run that never got that far; every
    phase takes this instead, where the same fields are non-optional and the
    "has not been initialized" guards have nothing left to check.
    """

    context: PipelineContext
    source: Path
    song_dir: Path
    output_dir: Path
    log_dir: Path
    base_name: str
    input_hash: str
    artifacts: RunArtifactRegistry

    @property
    def config(self) -> PipelineConfig:
        return self.context.config

    @property
    def metadata(self) -> RunMetadata:
        return self.context.metadata

    @property
    def runner(self) -> ProcessRunner:
        return self.context.runner

    @property
    def run_prefix(self) -> str:
        return self.context.run_prefix

    @property
    def device_job_slots(self) -> int:
        return self.context.device_job_slots


@dataclass(frozen=True)
class PipelineOutputs:
    """Final artifacts produced by a pipeline run."""

    final_midi: MidiDocument
    midi_path: Path | None = None
    mp3_path: Path | None = None


@dataclass(frozen=True)
class PipelineRunOutcome:
    """Transport-independent result returned without reading published files."""

    session_id: str
    status: str
    exit_code: int
    output_dir: Path | None
    metadata_path: Path | None
    performance: PerformanceMetrics | None
    error: str | None = None
