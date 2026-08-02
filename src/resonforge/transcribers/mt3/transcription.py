"""In-process MT3 transcription backend."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from resonforge.midi.jsonl import DRUMS, GM
from resonforge.pipeline.paths import MODELS_ROOT
from resonforge.pipeline.types import StemTask
from resonforge.transcribers.base import TranscriptionRequest

from .runtime import transcribe_audio

LOGGER = logging.getLogger(__name__)
MODELS = {
    "mt3_pytorch": "mt3_pytorch",
    "yptf_moe_multi": "yourmt3",
}


def output_path(task: StemTask, out_dir: Path, model_name: str) -> Path:
    return out_dir / ".mt3" / task.name / f"{task.audio.stem}_mt3-{model_name}.raw.midi"


def _valid_programs(instruments: list[str]) -> tuple[int, ...] | None:
    if not instruments or not all(
        instrument in GM or instrument in DRUMS for instrument in instruments
    ):
        return None
    return tuple(
        sorted(
            {GM[instrument] for instrument in instruments if instrument in GM}
            | ({128} if any(name in DRUMS for name in instruments) else set())
        )
    )


def transcribe(
    task: StemTask,
    *,
    model_name: str,
    out_dir: Path,
    environment_overrides: dict[str, str] | None = None,
    report_sink: dict[str, object] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Transcribe one stem through MT3-Infer's Python API."""
    runtime_model = MODELS[model_name]
    raw_midi = output_path(task, out_dir, model_name)
    raw_midi.parent.mkdir(parents=True, exist_ok=True)
    base = raw_midi.name.removesuffix(".raw.midi")
    raw_midi.with_name(base + ".clean.midi").unlink(missing_ok=True)
    raw_midi.with_name(base + ".global.clean.midi").unlink(missing_ok=True)
    instruments = [
        instrument.strip()
        for instrument in task.instruments.split(",")
        if instrument.strip()
    ]
    gpu = (environment_overrides or {}).get("CUDA_VISIBLE_DEVICES")
    device = f"cuda:{gpu}" if gpu is not None else "cuda"

    def report(completed: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(task.name, completed, total)

    LOGGER.info(
        "started stem=%s backend=mt3 model=%s device=%s",
        task.name,
        runtime_model,
        device,
        extra={"console": False},
    )
    transcribe_audio(
        task.audio,
        raw_midi,
        model_name=runtime_model,
        device=device,
        checkpoint_dir=MODELS_ROOT / "mt3",
        valid_programs=_valid_programs(instruments),
        progress_callback=report,
    )
    if len(instruments) == 1:
        instrument = instruments[0]
        if instrument not in GM and instrument not in DRUMS:
            raise RuntimeError(
                f"no GM program for MT3 stem {task.name!r}: {instrument!r}"
            )
    if report_sink is not None:
        report_sink["raw_midi"] = str(raw_midi.resolve())
    return raw_midi


class MT3Transcriber:
    name = "mt3"
    models = frozenset(MODELS)
    supported_stems = None
    supports_default = True

    def transcribe(self, request: TranscriptionRequest) -> Path:
        if request.model is None:
            raise RuntimeError("MT3 transcriber requires a model name")
        return transcribe(
            request.task,
            model_name=request.model,
            out_dir=request.out_dir,
            environment_overrides=request.environment_overrides,
            report_sink=request.report_sink,
            progress_callback=request.progress_callback,
        )

    def output_path(self, task: StemTask, out_dir: Path, model: str | None) -> Path:
        if model is None:
            raise RuntimeError("MT3 transcriber requires a model name")
        return output_path(task, out_dir, model)


BACKEND = MT3Transcriber()
