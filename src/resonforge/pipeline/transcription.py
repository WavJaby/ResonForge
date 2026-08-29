"""Transcription backends and the parallel stem transcription queue."""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext, suppress
from dataclasses import asdict
from pathlib import Path

from resonforge.midi.memory_cleanup import (
    AudioCleanupFeatures,
    extract_audio_cleanup_features,
)
from resonforge.observability.tempo_types import MuscriptorTempoReport
from resonforge.transcribers.base import (
    ModelLane,
    StemTask,
    TranscriptionRequest,
    TranscriptionResult,
)
from resonforge.transcribers.registry import get_transcriber

from ..runtime.process_runner import Cancelled, ProcessRunner
from .audio_admission import prepare_admitted_audio
from .console_progress import TranscriptionProgress
from .types import PipelineConfig

# General MIDI programs (T9/D9-1): chromatic percussion, organ, violin, string
# ensemble, trumpet, alto sax, flute, synth lead, synth pad.
OTHER_INSTRUMENTS = "12,19,40,48,56,65,73,80,88"
MODEL_SIZE_PRIORITY = {
    "large": 3,
    "medium": 2,
    "small": 1,
}
# Longest-expected-job-first ordering. A self-scheduling backend goes to the
# front because it needs several stems in flight before it can fill a batch;
# one that runs a model per stem gains nothing from starting early. Keyed on
# the capability rather than on a name, so a second backend orders correctly
# instead of raising KeyError against a table nobody thought to extend.
_SELF_SCHEDULING_PRIORITY = 3
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
        selection = get_transcriber(transcriber_for_stem(args, task.name))
        backend_priority = (
            _SELF_SCHEDULING_PRIORITY if selection.backend.self_scheduling else 0
        )
        model_priority = (
            0 if selection.model is None else model_size_priority(selection.model)
        )
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
    model_workers: object | None = None,
    summary_sink: list[str] | None = None,
    report_sink: dict[str, object] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    plan: object | None = None,
) -> TranscriptionResult:
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
            # The backend declares what it reads; the service never hands over
            # its whole configuration object.
            options=selection.backend.options_from(args),
            out_dir=out_dir,
            log_dir=log_dir,
            runner=runner,
            environment_overrides=environment_overrides,
            device=f"cuda:{gpu}" if gpu is not None else None,
            model_workers=model_workers,
            summary_sink=summary_sink,
            report_sink=report_sink,
            progress_callback=progress_callback,
            plan=plan,
        )
    )


_TRANSCRIBE_TASK_IMPL = transcribe_task


def songs_per_device(song_jobs: int, eligible_devices: int) -> int:
    """Songs that may run at once on one device.

    Session affinity admits to the least-loaded eligible device, so per-device
    occupancy never exceeds the ceiling of the mean. This scales the transient
    term only: concurrent songs overlap prefill, condition encode, and decode,
    but they queue into the same arena rather than building one each.

    Deliberately a static worst case, not live occupancy: it is part of the
    residency declaration, and the declaration is the key the width solve is
    cached under. A figure that moved with occupancy would hand two songs in
    one persistent run two different width vectors.
    """
    devices = max(1, eligible_devices)
    return max(1, -(-song_jobs // devices))


def run_queue(
    tasks: list[StemTask],
    *,
    parallelism: int,
    args: PipelineConfig,
    device_job_slots: int = 1,
    out_dir: Path,
    log_dir: Path,
    runner: ProcessRunner,
    timings: dict[str, dict[str, object]] | None = None,
    scheduler_timings: list[dict[str, object]] | None = None,
    session_id: str | None = None,
    audio_features: dict[str, AudioCleanupFeatures] | None = None,
    tempo_reports: dict[str, MuscriptorTempoReport] | None = None,
    preprocessing_metrics: dict[str, dict[str, object]] | None = None,
    device_assignment_sink: dict[str, object] | None = None,
    transcriber=None,
) -> dict[str, TranscriptionResult]:
    """Transcribe stems concurrently, assigning configured GPUs round-robin."""
    collect_summaries = transcriber is None
    if transcriber is None:
        transcriber = transcribe_task
    from resonforge.scheduler.global_service import acquire_global_scheduler
    from resonforge.scheduler.model_types import (
        ArenaDeclaration,
        ArenaLaneDeclaration,
    )
    from resonforge.transcribers.muscriptor.transcription import (
        load_worker_model,
        worker_device_for_gpu,
    )

    resolved_session_id = session_id or f"pipeline-{id(runner):x}"
    from resonforge.observability.global_waterfall import global_waterfall_store

    model_workers = acquire_global_scheduler(
        load_worker_model,
        session_id=resolved_session_id,
        # The pipeline owns both packages, so it is where the scheduler's spans
        # are wired to the store rather than the scheduler reaching for it.
        durable_sink=lambda span: global_waterfall_store().record(span),
    )
    from resonforge.scheduler.preparation_service import acquire_preparation_service

    preparation = acquire_preparation_service(session_id=resolved_session_id)
    from resonforge.scheduler.device.allocator import acquire_session_device

    device_assignment = acquire_session_device(
        resolved_session_id,
        args.gpus,
    )
    if device_assignment_sink is not None:
        device_assignment_sink.update(
            {
                "policy": "session_affinity",
                "gpu": device_assignment.gpu,
                "eligible_gpus": list(device_assignment.snapshot.eligible_gpus),
                "active_sessions_on_gpu_at_admission": (
                    device_assignment.snapshot.active_sessions_on_gpu
                ),
            }
        )
    manage_audio = collect_summaries and transcriber is _TRANSCRIBE_TASK_IMPL
    if manage_audio:
        resolved_device = worker_device_for_gpu(device_assignment.gpu)
        # A backend enrols in the joint width solve by returning a descriptor
        # (D4-1). One that loads a model per stem and releases it returns None
        # and is simply absent from the declaration, with no name test deciding
        # that on its behalf.
        descriptors = []
        for declared_task in tasks:
            declared = get_transcriber(
                transcriber_for_stem(args, declared_task.name),
                stem=declared_task.name,
            )
            # The declaration reaches into the backend, so it needs the
            # backend's options — not the service configuration. Before 3.4
            # these happened to share field names and `PipelineConfig` worked
            # by accident; it is the annotation that was always right.
            descriptor = declared.backend.describe(
                declared.model,
                declared.backend.options_from(args),
                resolved_device,
            )
            if descriptor is not None:
                descriptors.append(descriptor)
        if descriptors and resolved_device.startswith("cuda"):
            lanes: dict[str, ModelLane] = {}
            for descriptor in descriptors:
                for lane in descriptor.lanes:
                    lanes.setdefault(lane.model, lane)
            concurrent_songs = songs_per_device(
                device_job_slots,
                len(device_assignment.snapshot.eligible_gpus),
            )
            if device_assignment_sink is not None:
                # The transient budget is divided by this, so a run that cannot
                # be explained from its own record is a run that has to be
                # reproduced instead of read. It was `song_jobs` in
                # `effective_flags` until 3.5 moved the divisor to the service.
                device_assignment_sink.update(
                    {
                        "device_job_slots": device_job_slots,
                        "concurrent_songs": concurrent_songs,
                    }
                )
            # Residency is declared, not solved. Every lane that will share
            # this device is named before the first submission so the pool
            # knows which floors it still owes -- the `medium` recovery lane
            # must be able to open its arena after the primary has sized
            # itself, and a floor it does not know about is one the primary
            # will take.
            #
            # A forced width belongs on the submission's ceiling, not here:
            # nothing in the declaration derives a width any more, so there is
            # no second number for a Graph capture to disagree with.
            model_workers.declare_arenas(
                ArenaDeclaration(
                    device=resolved_device,
                    lanes=tuple(
                        ArenaLaneDeclaration(
                            key=lane.key,
                            kv_capacity=lane.kv_capacity,
                            minimum_width=lane.minimum_width,
                            maximum_width=lane.maximum_width,
                            arena_cost=lane.arena_cost,
                        )
                        for lane in lanes.values()
                    ),
                )
            )
            if device_assignment_sink is not None:
                from resonforge.scheduler.global_service import (
                    global_device_ownership_snapshot,
                )

                device_assignment_sink["core_device_ownership"] = list(
                    global_device_ownership_snapshot()
                )
    work: queue.Queue[StemTask] = queue.Queue()
    for task in tasks:
        work.put(task)
    errors: queue.Queue[BaseException] = queue.Queue()
    results: dict[str, TranscriptionResult] = {}
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

    assigned_gpu = device_assignment.gpu

    def is_self_scheduling(task: StemTask) -> bool:
        """Ask the backend, rather than testing its name (D3-2)."""
        return bool(
            get_transcriber(
                transcriber_for_stem(args, task.name), stem=task.name
            ).backend.self_scheduling
        )

    # A self-scheduling backend bypasses the --parallelism semaphore so its own
    # scheduler can batch across stems; that only pays when a batch can hold
    # more than one row and when this queue owns the run's reporting.
    batching_is_useful = args.batch_size != 1 and collect_summaries
    dynamic_stems = (
        sum(is_self_scheduling(task) for task in tasks) if batching_is_useful else 0
    )
    worker_count = min(len(tasks), max(parallelism, dynamic_stems))
    ordinary_slots = threading.Semaphore(parallelism)
    tempo_stems = (
        frozenset({"drums"})
        if any(task.name == "drums" for task in tasks)
        else frozenset(task.name for task in tasks)
    )
    if manage_audio:
        from resonforge.scheduler.host_memory import global_host_memory_budget

        host_budget = global_host_memory_budget(args.host_memory_budget_mib)
    else:
        host_budget = None

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
                    "model_workers": model_workers,
                }
                if collect_summaries:
                    transcriber_kwargs["summary_sink"] = summaries
                    transcriber_kwargs["report_sink"] = report
                    transcriber_kwargs["progress_callback"] = report_progress
                bypass_semaphore = batching_is_useful and is_self_scheduling(task)
                execution_slot = (
                    nullcontext() if bypass_semaphore else ordinary_slots
                )
                if host_budget is None:
                    with execution_slot:
                        transcription_result = transcriber(task, **transcriber_kwargs)
                else:
                    from resonforge.pipeline.muscriptor_tempo import (
                        analyze_muscriptor_tempo,
                    )

                    # Admission runs on this thread, never on a preparation
                    # worker. It blocks until the host-memory budget has room,
                    # and that room is only released by stems whose region
                    # producer runs on the *same* critical lane -- so holding a
                    # worker here closed a cycle that wedged the whole process
                    # at `--song-jobs 4` (R7). Submitting it bought no
                    # concurrency anyway: the caller awaited it immediately.
                    # Backpressure stays with `HostMemoryBudget`, which bounds
                    # decoded audio in bytes rather than in worker count.
                    def admit_audio(selected_task=task):
                        started_at = time.perf_counter()
                        try:
                            prepared = prepare_admitted_audio(
                                selected_task,
                                config=args,
                                budget=host_budget,
                                stop_event=runner.stop_event,
                            )
                            try:
                                return prepared, prepared.__enter__()
                            except BaseException:
                                prepared.__exit__(None, None, None)
                                raise
                        finally:
                            model_workers.record_waterfall_span(
                                "audio_prepare",
                                started_at,
                                time.perf_counter(),
                                resource="cpu",
                                lane=f"stem {selected_task.name}",
                                stem=selected_task.name,
                            )

                    prepared_lease, admitted = admit_audio()
                    feature_future = None
                    tempo_future = None
                    try:
                        prepared_task = admitted.task
                        if prepared_task.buffer is None:
                            raise RuntimeError(f"missing admitted audio for {task.name}")
                        selection = get_transcriber(
                            transcriber_for_stem(args, task.name),
                            stem=task.name,
                        )
                        model = selection.model
                        def analyze_in_background(
                            action, function, queued_at, *, stem_name=task.name
                        ):
                            started_at = time.perf_counter()
                            model_workers.record_waterfall_span(
                                "preparation_queue_wait",
                                queued_at,
                                started_at,
                                resource="cpu",
                                lane=f"stem {stem_name}",
                                stem=stem_name,
                                preparation_lane="background",
                                analysis=action,
                            )
                            try:
                                return function()
                            finally:
                                model_workers.record_waterfall_span(
                                    action,
                                    started_at,
                                    time.perf_counter(),
                                    resource="cpu",
                                    lane=f"stem {stem_name}",
                                    stem=stem_name,
                                )

                        def submit_background(action, function):
                            return preparation.submit(
                                "background",
                                analyze_in_background,
                                action,
                                function,
                                time.perf_counter(),
                            )

                        feature_future = (
                            submit_background(
                                "audio_features",
                                lambda selected=prepared_task: (
                                    extract_audio_cleanup_features(selected.buffer)
                                ),
                            )
                            if args.midi_cleanup
                            else None
                        )
                        tempo_future = (
                            submit_background(
                                "tempo_analysis",
                                lambda selected=prepared_task, selected_model=model: (
                                    analyze_muscriptor_tempo(
                                        selected,
                                        model=selected_model,
                                    )
                                ),
                            )
                            if task.name in tempo_stems
                            else None
                        )
                        # A backend that decides its own chunking publishes
                        # the whole inventory before any model runs, so the
                        # scheduler sees every unit it will be asked for. One
                        # that transcribes a stem in a single pass returns
                        # None and skips this (D3-2's class, not its case).
                        plan_started = time.perf_counter()
                        work_plan = selection.backend.plan(
                            prepared_task,
                            selection.backend.options_from(args),
                        )
                        if work_plan is not None:
                            transcriber_kwargs["plan"] = work_plan
                            model_workers.record_waterfall_span(
                                "region_plan",
                                plan_started,
                                time.perf_counter(),
                                resource="cpu",
                                lane=f"stem {task.name}",
                                stem=task.name,
                            )
                        with results_lock:
                            if preprocessing_metrics is not None:
                                preprocessing_metrics[task.name] = {
                                    **admitted.preprocessing,
                                    "audio_memory": {
                                        "requested_bytes": (
                                            admitted.reservation.requested_bytes
                                        ),
                                        "admitted_bytes": (
                                            admitted.reservation.admitted_bytes
                                        ),
                                        "wait_seconds": round(
                                            admitted.reservation.wait_seconds, 6
                                        ),
                                    },
                                }
                        with execution_slot:
                            transcription_result = transcriber(
                                prepared_task,
                                **transcriber_kwargs,
                            )
                        features = (
                            feature_future.result()
                            if feature_future is not None
                            else None
                        )
                        tempo_report = (
                            tempo_future.result() if tempo_future is not None else None
                        )
                        with results_lock:
                            if audio_features is not None and features is not None:
                                audio_features[task.name] = features
                            if tempo_reports is not None and tempo_report is not None:
                                tempo_reports[task.name] = tempo_report
                    finally:
                        for background_future in (feature_future, tempo_future):
                            if background_future is None or background_future.cancel():
                                continue
                            with suppress(BaseException):
                                background_future.result()
                        prepared_lease.__exit__(None, None, None)
                elapsed = time.monotonic() - started
                with results_lock:
                    results[task.name] = transcription_result
                    if timings is not None:
                        timings[task.name] = {
                            "elapsed_seconds": round(elapsed, 3),
                            "gpu": gpu,
                            "artifacts": report,
                            "metrics": getattr(transcription_result, "metrics", {}),
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
            args=(assigned_gpu,),
            name=f"stem-worker-{index + 1}",
            daemon=True,
        )
        for index in range(worker_count)
    ]
    try:
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            drain_progress()
            drain_completions()
            if runner.stop_event.is_set():
                break
            for thread in threads:
                thread.join(timeout=0.1)
        drain_progress()
        drain_completions()
    finally:
        progress.close()
        model_workers.close()
        device_assignment.close()
        preparation.close(cancel_pending=runner.stop_event.is_set())
        if scheduler_timings is not None:
            scheduler_timings.extend(
                asdict(timing)
                for timing in model_workers.timings
                if timing.pipeline_session_id == session_id
            )
    if not errors.empty():
        raise errors.get()
    if runner.stop_event.is_set():
        raise Cancelled()
    return results
