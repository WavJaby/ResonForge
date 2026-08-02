"""In-process Basic Pitch transcription with window progress events."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from resonforge.midi.jsonl import GM
from resonforge.pipeline.types import StemTask
from resonforge.transcribers.base import TranscriptionRequest

from .inference import predict_model_output

LOGGER = logging.getLogger(__name__)
SUPPORTED_STEMS = frozenset({"bass", "guitar", "piano"})
_MODEL: Any | None = None
_MODEL_LOCK = threading.Lock()


def output_path(task: StemTask, out_dir: Path) -> Path:
    return (
        out_dir / ".basic-pitch" / task.name / f"{task.audio.stem}_basic_pitch.raw.midi"
    )


def _load_model() -> Any:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import Model

            _MODEL = Model(ICASSP_2022_MODEL_PATH)
    return _MODEL


def _predict(
    audio: Path,
    progress: Callable[[int, int], None],
) -> Any:
    from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
    from basic_pitch.note_creation import model_output_to_notes

    unwrapped = predict_model_output(audio, _load_model(), progress)
    minimum_note_length = int(
        round(0.1277 * (AUDIO_SAMPLE_RATE / FFT_HOP))
    )
    midi, _ = model_output_to_notes(
        unwrapped,
        onset_thresh=0.5,
        frame_thresh=0.3,
        min_note_len=minimum_note_length,
        multiple_pitch_bends=False,
        melodia_trick=True,
        midi_tempo=120,
    )
    return midi


def transcribe(
    task: StemTask,
    *,
    out_dir: Path,
    report_sink: dict[str, object] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Transcribe one stem through Basic Pitch's ONNX window API."""
    if task.instruments not in GM:
        raise RuntimeError(
            f"no GM program for Basic Pitch stem {task.name!r}: {task.instruments!r}"
        )
    raw_midi = output_path(task, out_dir)
    raw_midi.parent.mkdir(parents=True, exist_ok=True)
    base = raw_midi.name.removesuffix(".raw.midi")
    raw_midi.with_name(base + ".clean.midi").unlink(missing_ok=True)
    raw_midi.with_name(base + ".global.clean.midi").unlink(missing_ok=True)

    def report(completed: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(task.name, completed, total)

    LOGGER.info(
        "started stem=%s backend=basic-pitch runtime=onnx",
        task.name,
        extra={"console": False},
    )
    midi = _predict(task.audio, report)
    midi.write(str(raw_midi))
    if not raw_midi.is_file():
        raise RuntimeError(f"Basic Pitch did not create: {raw_midi}")
    if report_sink is not None:
        report_sink["raw_midi"] = str(raw_midi.resolve())
    return raw_midi


class BasicPitchTranscriber:
    name = "basic-pitch"
    models = frozenset()
    supported_stems = SUPPORTED_STEMS
    supports_default = False

    def transcribe(self, request: TranscriptionRequest) -> Path:
        return transcribe(
            request.task,
            out_dir=request.out_dir,
            report_sink=request.report_sink,
            progress_callback=request.progress_callback,
        )

    def output_path(self, task: StemTask, out_dir: Path, model: str | None) -> Path:
        return output_path(task, out_dir)


BACKEND = BasicPitchTranscriber()
