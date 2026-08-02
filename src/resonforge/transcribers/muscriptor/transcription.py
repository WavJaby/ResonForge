"""MuScriptor pipeline backend."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from pathlib import Path

from resonforge.gpu import muscriptor_dtype
from resonforge.logging_config import configure_muscriptor_loggers
from resonforge.midi.jsonl import convert_jsonl_to_midi
from resonforge.pipeline.process_runner import ProcessRunner
from resonforge.pipeline.types import PipelineConfig, StemTask
from resonforge.transcribers.base import TranscriptionRequest

_MODEL_CACHE: dict[tuple[str | None, str | None], object] = {}
LOGGER = logging.getLogger(__name__)


def output_prefix(task: StemTask) -> str:
    return f"{task.audio.stem}_muscriptor"


def output_path(task: StemTask, out_dir: Path) -> Path:
    return out_dir / ".muscriptor" / task.name / f"{output_prefix(task)}.raw.midi"


def _event_to_jsonl(event: object) -> dict[str, object] | None:
    """Convert one event into the CLI-style JSONL shape."""
    from muscriptor.events import NoteEndEvent, NoteStartEvent

    if isinstance(event, NoteStartEvent):
        return {"type": "start", **dataclasses.asdict(event)}
    if isinstance(event, NoteEndEvent):
        return {
            "type": "end",
            "end_time": event.end_time,
            "start_event_index": event.start_event_index,
        }
    return None


def _load_model(model: str, dtype: str | None = None) -> object:
    """Load one model instance per (model, dtype) pair."""
    from muscriptor.transcription_model import TranscriptionModel

    key = (model, dtype)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    loaded = TranscriptionModel.load_model(
        weights_path=model,
        dtype=dtype,
    )
    _MODEL_CACHE[key] = loaded
    return loaded


def _resolve_instruments(spec: str | None) -> list[str] | None:
    if not spec or not spec.strip():
        return None
    names = [value.strip() for value in spec.split(",") if value.strip()]
    if not names:
        return None
    from muscriptor.tokenizer.mt3 import resolve_instrument_names

    try:
        return resolve_instrument_names(names)
    except ValueError as error:
        raise RuntimeError(f"Invalid Muscriptor instrument list: {error}") from error


def transcribe(
    task: StemTask,
    *,
    model: str,
    args: PipelineConfig,
    out_dir: Path,
    log_dir: Path,
    runner: ProcessRunner,
    environment_overrides: dict[str, str] | None = None,
    summary_sink: list[str] | None = None,
    report_sink: dict[str, object] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Transcribe one stem and create raw JSONL + MIDI artifacts."""
    raw_midi = output_path(task, out_dir)
    raw_midi.parent.mkdir(parents=True, exist_ok=True)
    base = raw_midi.name.removesuffix(".raw.midi")
    raw_midi.with_name(base + ".clean.midi").unlink(missing_ok=True)
    raw_midi.with_name(base + ".global.clean.midi").unlink(missing_ok=True)
    raw_json_path = raw_midi.with_name(f"{output_prefix(task)}.raw.json")
    logs = configure_muscriptor_loggers(
        log_dir,
        task.name,
    )
    LOGGER.info(
        "started stem=%s backend=muscriptor model=%s",
        task.name,
        model,
        extra={"console": False},
    )

    try:
        model_obj = _load_model(
            model=model,
            dtype=muscriptor_dtype(
                0
                if environment_overrides
                and "CUDA_VISIBLE_DEVICES" in environment_overrides
                else None
            ),
        )
    except BaseException:
        logs.close()
        raise
    instruments = _resolve_instruments(task.instruments)
    prelude_forcing = args.batch_size <= 1

    try:
        from muscriptor.events import ProgressEvent

        with raw_json_path.open("w", encoding="utf-8") as json_handle:
            events = model_obj.transcribe(
                task.audio,
                use_sampling=False,
                cfg_coef=args.cfg_coef,
                instruments=instruments,
                batch_size=args.batch_size,
                no_eos_is_ok=True,
                beam_size=args.beam_size,
                prelude_forcing=prelude_forcing,
                stdout_logger=logs.stdout_logger,
                stderr_logger=logs.stderr_logger,
            )
            for event in events:
                if isinstance(event, ProgressEvent):
                    if progress_callback is not None:
                        progress_callback(
                            task.name,
                            event.completed,
                            event.total,
                        )
                    continue
                payload = _event_to_jsonl(event)
                if payload is not None:
                    json_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as error:
        logs.stderr_logger.exception("MuScriptor transcription failed")
        raise RuntimeError(
            f"MuScriptor transcription failed for {task.name}; "
            f"see {logs.stderr_path}"
        ) from error
    finally:
        logs.close()

    if not raw_json_path.is_file():
        raise RuntimeError(f"MuScriptor JSONL was not created: {raw_json_path}")

    convert_jsonl_to_midi(
        raw_json_path,
        out=raw_midi,
        cleanup=False,
        verbose=False,
    )
    if not raw_midi.is_file():
        raise RuntimeError(f"MuScriptor raw MIDI was not created: {raw_midi}")
    if report_sink is not None:
        report_sink.update(
            {
                "raw_json": str(raw_json_path.resolve()),
                "raw_midi": str(raw_midi.resolve()),
            }
        )
    if summary_sink is not None:
        summary_sink.append("raw MIDI ready; cleanup deferred")
    return raw_midi


class MuScriptorTranscriber:
    name = "muscriptor"
    models = frozenset({"small", "medium", "large"})
    supported_stems = None
    supports_default = True

    def transcribe(self, request: TranscriptionRequest) -> Path:
        if request.model is None:
            raise RuntimeError("MuScriptor transcriber requires a model size")
        return transcribe(
            request.task,
            model=request.model,
            args=request.config,
            out_dir=request.out_dir,
            log_dir=request.log_dir,
            runner=request.runner,
            environment_overrides=request.environment_overrides,
            summary_sink=request.summary_sink,
            report_sink=request.report_sink,
            progress_callback=request.progress_callback,
        )

    def output_path(self, task: StemTask, out_dir: Path, model: str | None) -> Path:
        return output_path(task, out_dir)


BACKEND = MuScriptorTranscriber()

