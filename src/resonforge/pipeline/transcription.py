"""Transcription backends and the parallel stem transcription queue."""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from resonforge.transcribers.base import TranscriptionRequest
from resonforge.transcribers.registry import get_transcriber

from .console_progress import TranscriptionProgress
from .process_runner import Cancelled, ProcessRunner
from .types import PipelineConfig, StemTask

OTHER_INSTRUMENTS = (
    "chromatic_percussion,organ,violin,string_ensemble,trumpet,"
    "soprano_and_alto_sax,flutes,synth_lead,synth_pad"
)
MODEL_SIZE_PRIORITY = {
    "yptf_moe_multi": 3,
    "mt3_pytorch": 2,
    "large": 3,
    "medium": 2,
    "small": 1,
}
BACKEND_PRIORITY = {
    "muscriptor": 3,
    "mt3": 2,
    "basic-pitch": 1,
}
LOGGER = logging.getLogger(__name__)


def transcriber_for_stem(args: PipelineConfig, stem: str) -> str:

    return args.transcriber_for_stem(stem)


def split_transcriber_spec(spec: str) -> tuple[str, str | None]:
    """Split a validated transcriber spec into backend and optional model."""
    selection = get_transcriber(spec)
    return selection.backend.name, selection.model


def model_size_priority(model: str) -> int:
    """Return a best-effort size rank for named or path-like models."""
    normalized = str(model).lower().replace("\\", "/")
    for size, priority in MODEL_SIZE_PRIORITY.items():
        if size in normalized.rsplit("/", 1)[-1]:
            return priority
    return 0


def order_tasks_for_queue(
    tasks: list[StemTask],
    *,
    args: PipelineConfig,
    presence_metrics: dict[str, dict[str, object]] | None = None,
) -> list[StemTask]:
    """Schedule backends and models with the longest expected jobs first."""
    metrics = presence_metrics or {}

    def priority(task: StemTask) -> tuple[int, int, float]:
        backend, model = split_transcriber_spec(transcriber_for_stem(args, task.name))
        backend_priority = BACKEND_PRIORITY[backend]
        model_priority = 0 if model is None else model_size_priority(model)
        coverage = float(
            metrics.get(task.name, {}).get(
                "coverage_above_minus_40_percent",
                0.0,
            )
        )
        return (-backend_priority, -model_priority, -coverage)

    return sorted(tasks, key=priority)


def transcribe_task(
    task: StemTask,
    *,
    args: PipelineConfig,
    out_dir: Path,
    log_dir: Path,
    runner: ProcessRunner,
    gpu: int | None = None,
    summary_sink: list[str] | None = None,
    report_sink: dict[str, object] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Dispatch a stem to its configured transcription backend."""
    environment_overrides = (
        {"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else None
    )
    selection = get_transcriber(
        transcriber_for_stem(args, task.name),
        stem=task.name,
    )
    return selection.backend.transcribe(
        TranscriptionRequest(
            task=task,
            model=selection.model,
            config=args,
            out_dir=out_dir,
            log_dir=log_dir,
            runner=runner,
            environment_overrides=environment_overrides,
            summary_sink=summary_sink,
            report_sink=report_sink,
            progress_callback=progress_callback,
        )
    )


def transcribed_midi_path(
    task: StemTask,
    *,
    args: PipelineConfig,
    out_dir: Path,
) -> Path:
    selection = get_transcriber(
        transcriber_for_stem(args, task.name),
        stem=task.name,
    )
    raw_path = selection.backend.output_path(task, out_dir, selection.model)
    base = raw_path.name.removesuffix(".raw.midi")
    suffix = (
        ".global.clean.midi"
        if getattr(args, "midi_cleanup", False)
        and not (
            getattr(args, "only_other", False)
            or getattr(args, "treat_original_as_other", False)
        )
        else ".clean.midi"
    )
    return raw_path.with_name(base + suffix)


def run_queue(
    tasks: list[StemTask],
    *,
    parallelism: int,
    args: PipelineConfig,
    out_dir: Path,
    log_dir: Path,
    runner: ProcessRunner,
    timings: dict[str, dict[str, object]] | None = None,
    transcriber=None,
) -> dict[str, Path]:
    """Transcribe stems concurrently, assigning configured GPUs round-robin."""
    collect_summaries = transcriber is None
    if transcriber is None:
        transcriber = transcribe_task
    work: queue.Queue[StemTask] = queue.Queue()
    for task in tasks:
        work.put(task)
    errors: queue.Queue[BaseException] = queue.Queue()
    results: dict[str, Path] = {}
    results_lock = threading.Lock()
    progress_updates: queue.Queue[tuple[str, int, int]] = queue.Queue()
    completion_updates: queue.Queue[tuple[str, float, str, str]] = queue.Queue()
    progress_enabled = sys.stderr.isatty()
    ordered_names = [task.name for task in tasks]
    progress = TranscriptionProgress(ordered_names, enabled=progress_enabled)

    def report_progress(stem: str, completed: int, total: int) -> None:
        progress_updates.put((stem, completed, total))

    def drain_progress() -> None:
        while True:
            try:
                stem, completed, total = progress_updates.get_nowait()
            except queue.Empty:
                break
            progress.update(stem, completed, total)
            LOGGER.info(
                "transcription progress stem=%s completed=%s total=%s percent=%.1f",
                stem,
                completed,
                total,
                completed / total * 100 if total else 0.0,
                extra={"console": False},
            )

    def drain_completions() -> None:
        while True:
            try:
                stem, elapsed, gpu_label, summary = completion_updates.get_nowait()
            except queue.Empty:
                break
            progress.complete(stem, elapsed, f"{gpu_label}; {summary}".rstrip("; "))
            progress.log_completion(
                LOGGER,
                stem,
                elapsed,
                gpu_label,
                summary,
            )

    gpus = getattr(args, "gpus", ())

    def worker(gpu: int | None) -> None:
        while not runner.stop_event.is_set():
            try:
                task = work.get_nowait()
            except queue.Empty:
                return
            started = time.monotonic()
            try:
                summaries: list[str] = []
                report: dict[str, object] = {}
                transcriber_kwargs = {
                    "args": args,
                    "out_dir": out_dir,
                    "log_dir": log_dir,
                    "runner": runner,
                    "gpu": gpu,
                }
                if collect_summaries:
                    transcriber_kwargs["summary_sink"] = summaries
                    transcriber_kwargs["report_sink"] = report
                    transcriber_kwargs["progress_callback"] = report_progress
                midi_path = transcriber(task, **transcriber_kwargs)
                elapsed = time.monotonic() - started
                with results_lock:
                    results[task.name] = midi_path
                    if timings is not None:
                        timings[task.name] = {
                            "elapsed_seconds": round(elapsed, 3),
                            "gpu": gpu,
                            "artifacts": report,
                        }
                gpu_label = f"GPU {gpu}" if gpu is not None else "default GPU"
                completion_updates.put(
                    (
                        task.name,
                        elapsed,
                        gpu_label,
                        "; ".join(summaries),
                    )
                )
            except Cancelled:
                return
            except Exception as error:  # noqa: BLE001 - forward worker failures
                errors.put(error)
                runner.request_stop()
                return
            finally:
                work.task_done()

    threads = [
        threading.Thread(
            target=worker,
            args=(gpus[index % len(gpus)] if gpus else None,),
            name=f"stem-worker-{index + 1}",
            daemon=True,
        )
        for index in range(min(parallelism, len(tasks)))
    ]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        drain_progress()
        drain_completions()
        for thread in threads:
            thread.join(timeout=0.1)
    drain_progress()
    drain_completions()
    progress.close()
    if not errors.empty():
        raise errors.get()
    if runner.stop_event.is_set():
        raise Cancelled()
    return results

