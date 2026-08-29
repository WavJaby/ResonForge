"""Common contracts for transcription backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from resonforge.audio.buffer import AudioBuffer
from resonforge.midi.document import MidiDocument
from resonforge.midi.intervals import Note

if TYPE_CHECKING:
    from resonforge.runtime.process_runner import ProcessRunner
    from resonforge.scheduler.model_types import ArenaCostProbe, ModelKey


@dataclass(frozen=True)
class BackendOptions:
    """Marker base for the options one backend reads from the caller.

    The interface deliberately does not name a configuration type. A backend
    declares its own fields and builds them from any attribute source, so the
    service can hand over a CLI namespace today and a decoded API request later
    without the transcriber ever importing the pipeline.
    """


@dataclass(frozen=True)
class StemTask:
    """One separated or combined audio stem awaiting transcription.

    Lives beside the interface rather than in the pipeline: it is the
    transcription contract's input, and nothing about it is pipeline-specific.
    """

    name: str
    audio: Path
    instruments: str
    buffer: AudioBuffer | None = None
    sources: tuple[Path, ...] = ()


@dataclass(frozen=True)
class TranscriptionRequest:
    """Backend-independent inputs for transcribing one stem."""

    task: StemTask
    model: str | None
    options: BackendOptions
    out_dir: Path
    log_dir: Path
    runner: ProcessRunner
    environment_overrides: dict[str, str] | None = None
    device: str | None = None
    model_workers: object | None = None
    summary_sink: list[str] | None = None
    report_sink: dict[str, object] | None = None
    progress_callback: Callable[[str, int, int], None] | None = None
    plan: object | None = None


@dataclass(frozen=True)
class ModelLane:
    """One model a backend will hold resident, and the key it submits under."""

    # lane identity for the block pool, which knows nothing about keys -- the name this lane's row cost is measured under
    model: str
    # scheduler identity: device, dtype, execution profile. Only the backend can build it, which is why the declaration comes from here and not from the queue that consumes it.
    key: ModelKey
    # how this lane prices one logical row, asked of its own loaded model. `None` leaves the lane unpriced and the pool never bounds it.
    arena_cost: ArenaCostProbe | None = None
    # rows this lane needs resident at once to do its job. A recovery lane needs one, and the pool holds that one back from every other lane.
    minimum_width: int = 1
    # rows this lane can ever put to work; `None` = only memory bounds it.
    # What stops the lane that opens first taking the device from the lane that does the work -- policy the backend knows and the pool can't derive.
    maximum_width: int | None = None
    # The KV length the lane's floor is priced at.
    kv_capacity: int = 2048


@dataclass(frozen=True)
class ModelDescriptor:
    """What a backend will hold resident for one request (T1 / D4-1).

    What a backend genuinely knows on its own: which models it holds resident to serve one request, under what scheduler identity, and how to price one of their rows once loaded.
    The model set is quality policy -- a primary tier plus whatever recovery and bootstrap tiers this backend's design brings -- and the scheduler can't derive it.

    Row cost is a CALLABLE, not a number, for the same reason: only the loaded model knows its own state layout, and the pool asks it fresh on every device rather than reading a stored figure.
    """

    model: str
    lanes: tuple[ModelLane, ...]

    def __post_init__(self) -> None:
        if self.model not in {lane.model for lane in self.lanes}:
            raise ValueError(f"{self.model!r} must appear in its own lanes")


@dataclass(frozen=True)
class MemoryArtifact:
    """Optional serialized output retained independently of pipeline flow."""

    name: str
    kind: str
    media_type: str
    data: bytes
    metadata: dict[str, object]


@dataclass(frozen=True)
class TranscriptionResult:
    """Complete backend result carried between stages without filesystem I/O."""

    stem: str
    events: tuple[dict[str, Any], ...]
    raw_midi: MidiDocument
    artifacts: tuple[MemoryArtifact, ...]
    metrics: dict[str, object]
    summary: str


@dataclass(frozen=True)
class ProcessedTranscription:
    """One backend result after MIDI postprocessing, ready to publish."""

    stem: str
    events: tuple[dict[str, object], ...]
    raw_midi: MidiDocument
    clean_midi: MidiDocument
    selected_midi: MidiDocument
    notes: tuple[Note, ...]
    artifacts: tuple[MemoryArtifact, ...]
    reports: dict[str, object]
    metrics: dict[str, object]


class Transcriber(Protocol):
    """Interface implemented by every transcription backend adapter."""

    name: str
    models: frozenset[str]
    supported_stems: frozenset[str] | None
    supports_default: bool
    # True when the backend admits its own work through a process-global scheduler. Such a backend must NOT also pass the service's per-stem semaphore:
    # the scheduler needs several stems in flight to fill a batch and a stem-level gate would starve it.
    # A backend running one model or subprocess per stem declares False and the semaphore binds it -- the only thing between it and unbounded concurrency.
    self_scheduling: bool
    # GM programs this backend can emit, plus 128 for channel-10 percussion. The service names GM and only GM (T9/D9-1);
    # this is how a request for something the model can't reach becomes an answerable question instead of silence.
    supported_programs: frozenset[int]

    def options_from(self, source: Any) -> BackendOptions:
        """Build this backend's options from any attribute source."""

    def describe(
        self,
        model: str | None,
        options: BackendOptions,
        device: str,
    ) -> ModelDescriptor | None:
        """Declare what stays resident, or None to claim no device residency.

        `device` is an identity, not a handle -- the backend names the device its keys belong to and queries nothing about it.
        `None` is the honest default for a backend that loads one model per stem and releases it: nothing for the capacity solver to hold widths open for.
        """
        return None

    def plan(self, task: StemTask, options: BackendOptions) -> object | None:
        """Inventory this stem's work before any model runs, or None.

        A backend that decides its own chunking publishes the full inventory up front, so the scheduler sees every unit it will be asked for rather than discovering them one at a time.
        A backend that transcribes a stem in one pass has nothing to inventory and returns `None`.
        """
        return None

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Transcribe one stem and return an in-memory result."""

    def output_path(
        self,
        task: StemTask,
        out_dir: Path,
        model: str | None,
    ) -> Path:
        """Return the raw MIDI path produced by this backend."""
