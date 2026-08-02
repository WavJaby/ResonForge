"""Common contracts for transcription backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from resonforge.pipeline.process_runner import ProcessRunner
    from resonforge.pipeline.types import PipelineConfig, StemTask


@dataclass(frozen=True)
class TranscriptionRequest:
    """Backend-independent inputs for transcribing one stem."""

    task: StemTask
    model: str | None
    config: PipelineConfig
    out_dir: Path
    log_dir: Path
    runner: ProcessRunner
    environment_overrides: dict[str, str] | None = None
    summary_sink: list[str] | None = None
    report_sink: dict[str, object] | None = None
    progress_callback: Callable[[str, int, int], None] | None = None


class Transcriber(Protocol):
    """Interface implemented by every transcription backend adapter."""

    name: str
    models: frozenset[str]
    supported_stems: frozenset[str] | None
    supports_default: bool

    def transcribe(self, request: TranscriptionRequest) -> Path:
        """Transcribe one stem and return the generated MIDI path."""

    def output_path(
        self,
        task: StemTask,
        out_dir: Path,
        model: str | None,
    ) -> Path:
        """Return the raw MIDI path produced by this backend."""
