"""Phase-oriented orchestration for the audio-to-MIDI pipeline."""

from __future__ import annotations

import logging
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from resonforge.transcribers.base import StemTask

from ..observability.artifacts import RunArtifactRegistry
from ..observability.gpu_telemetry import GpuTelemetryMonitor
from ..observability.performance import calculate_performance
from ..observability.run_metadata import (
    audio_file_info,
    create_run_metadata,
    write_run_metadata,
    write_scheduler_liveness,
    write_scheduler_telemetry,
)
from ..observability.transcriber_archive import (
    archive_memory_transcriptions,
)
from ..runtime.logging_config import (
    bind_run_logging,
    configure_model_download_logging,
    configure_run_logging,
    finalize_run_logs,
)
from ..runtime.process_runner import Cancelled, ProcessRunner
from ..transcribers.base import TranscriptionResult
from ..transcribers.muscriptor.execution_profile import (
    DEFAULT_GRAPH_BUCKET_SIZE,
    DEFAULT_GRAPH_CACHE_SIZE,
)
from .audio_staging import (
    cleanup_staged_input,
    inspect_input,
    require_file,
    stage_input,
)
from .memory_postprocessing import (
    PostprocessBatchResult,
    postprocess_in_memory,
)
from .output_production import produce_outputs
from .paths import (
    BS_CACHE_ROOT,
    BS_CONFIG_PATH,
    BS_MODEL_PATH,
    BS_OUTPUT_ROOT,
    MODELS_ROOT,
    PYTHON,
)
from .preprocessing import preprocess_stems
from .separation import (
    BS_STEMS_DIRNAME,
    global_separation_admission,
    make_tasks,
    separate_stems,
    separation_cache_id,
    separation_cache_is_valid,
    separation_cache_lock,
)
from .transcription import (
    order_tasks_for_queue,
    run_queue,
    split_transcriber_spec,
    transcriber_for_stem,
)
from .types import (
    InitializedRun,
    PipelineConfig,
    PipelineContext,
    PipelineOutputs,
    PipelineRunOutcome,
)

LOGGER = logging.getLogger(__name__)


def _scheduler_liveness_report(error: BaseException | None) -> dict[str, object] | None:
    """Find device-level evidence through pipeline/transcriber error wrappers."""
    seen: set[int] = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        report = getattr(current, "report", None)
        if isinstance(report, dict):
            return report
        current = current.__cause__ or current.__context__
    return None


def make_run_prefix(now: datetime | None = None) -> str:
    """Return the local timestamp shared by all final artifacts in one run."""
    timestamp = (now or datetime.now().astimezone()).strftime("%y%m%d-%H%M%S")
    return f"{timestamp}-{secrets.token_hex(3)}"


def make_band_task(
    tasks: list[StemTask],
    output_dir: Path,
    base_name: str,
    band_names: frozenset[str] = frozenset({"bass", "guitar", "piano"}),
) -> StemTask:
    band_sources = [task for task in tasks if task.name in band_names]
    if len(band_sources) != len(band_names):
        raise RuntimeError("could not resolve all requested band stems")
    combined_label = "-".join(task.name for task in band_sources)
    return StemTask(
        "band",
        output_dir / f"{base_name}_band-{combined_label}.wav",
        ",".join(task.instruments for task in band_sources),
        sources=tuple(task.audio for task in band_sources),
    )


def prepare_pipeline_context(
    args: PipelineConfig,
    *,
    run_prefix: str | None = None,
    stop_event: threading.Event | None = None,
    device_job_slots: int = 1,
) -> PipelineContext:
    """Create typed run configuration and run-scoped mutable state."""
    config = args
    run_prefix = run_prefix or make_run_prefix()
    metadata = create_run_metadata(run_prefix, args)
    metadata_path = (
        (config.output_root or BS_OUTPUT_ROOT).expanduser().resolve()
        / Path(config.input_file).stem
        / (f"{run_prefix}-metadata.json")
    )
    return PipelineContext(
        config=config,
        run_prefix=run_prefix,
        metadata=metadata,
        metadata_path=metadata_path,
        runner=ProcessRunner(
            stop_event or threading.Event(),
            model_cache=MODELS_ROOT / "huggingface" / "hub",
        ),
        device_job_slots=device_job_slots,
    )


def initialize_run(context: PipelineContext) -> InitializedRun:
    """Validate dependencies and resolve input/output locations."""
    config = context.config
    dependencies = [(PYTHON, "Shared Python environment")]
    if not config.treat_original_as_other:
        dependencies.extend(
            (
                (BS_MODEL_PATH, "BS-RoFormer model"),
                (BS_CONFIG_PATH, "BS-RoFormer config"),
            )
        )
    for path, description in dependencies:
        require_file(path, description)

    source, song_dir, base_name, input_hash = inspect_input(
        config.input_file, config.output_root
    )
    if config.publish_files:
        output_dir = song_dir / "runs" / context.run_prefix
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix=f"resonforge-{context.run_prefix}-"))
        context.transient_output_dir = True
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_paths = configure_run_logging(output_dir, log_dir, context.run_prefix)
    model_download_log = configure_model_download_logging(MODELS_ROOT)
    from muscriptor.utils.download import configure_huggingface_download_logging

    configure_huggingface_download_logging()
    LOGGER.info("starting run %s", context.run_prefix)
    LOGGER.info("input: %s", source)
    LOGGER.info("output directory: %s", output_dir)
    context.source = source
    context.song_dir = song_dir
    context.output_dir = output_dir
    context.log_dir = log_paths.directory
    context.base_name = base_name
    context.input_hash = input_hash
    context.metadata_path = output_dir / "metadata.json"
    context.artifacts = RunArtifactRegistry(
        output_dir, context.run_prefix, session_root=True
    )
    context.metadata["input"] = {**audio_file_info(source), "sha256": input_hash}
    context.metadata["outputs"] = {
        "directory": str(output_dir),
        "metadata": str(context.metadata_path),
        "logs": {
            "pipeline": str(log_paths.pipeline.relative_to(output_dir)),
            "model_downloads": str(model_download_log),
        },
    }
    context.metadata["models"] = {
        "bs_roformer": {
            "checkpoint": str(BS_MODEL_PATH),
            "config": str(BS_CONFIG_PATH),
            "sample_rate": config.bs_sample_rate,
        },
        "transcription": {
            "default": config.default_transcriber,
            "overrides": dict(config.transcribers),
            # Recorded wholesale: whatever the backend was configured with ends
            # up in the run record without this block naming any of it.
            **dict(config.backend_options),
        },
    }
    LOGGER.info(f"Input file:      {source}")
    LOGGER.info(f"Output directory: {output_dir}")
    return InitializedRun(
        context=context,
        source=source,
        song_dir=song_dir,
        output_dir=output_dir,
        log_dir=log_paths.directory,
        base_name=base_name,
        input_hash=input_hash,
        artifacts=context.artifacts,
    )


def prepare_separated_stems(run: InitializedRun) -> list[StemTask]:
    """Resolve direct-original mode or create BS-RoFormer stems."""
    config = run.config
    if config.treat_original_as_other:
        input_dir = stage_input(
            run.source,
            run.input_hash,
            sample_rate=config.bs_sample_rate,
            runner=run.runner,
            log_dir=run.log_dir,
        )
        run.context.staged_input_dir = input_dir
        staged_audio = input_dir / f"{run.source.stem}.wav"
        require_file(staged_audio, "Staged original input")
        task = StemTask("other", staged_audio, config.other_instruments)
        run.metadata["separation"] = {
            "mode": "original_as_other",
            "skipped": True,
            "input_sha256": run.input_hash,
            "sample_rate": config.bs_sample_rate,
            "staged_audio": str(staged_audio),
        }
        run.metadata["stems"]["other"] = {
            "separated_audio": audio_file_info(task.audio),
            "selected_for_transcription": True,
            "source": "original_input",
        }
        LOGGER.info(
            "\n== Treating original input as other ==\n"
            f"  {run.source} -> {task.audio}\n"
            "  BS-RoFormer skipped",
        )
        return [task]

    cache_id = separation_cache_id(run.input_hash, config.bs_sample_rate)
    # Under BS_CACHE_ROOT rather than the run's own song directory: the entry is
    # keyed by input hash and sample rate, so it is shared by every run of the
    # same audio whatever --output-root they were given.
    stem_dir = BS_CACHE_ROOT / run.base_name / BS_STEMS_DIRNAME / cache_id
    tasks = make_tasks(stem_dir, config.other_instruments)
    cache_hit = False
    with separation_cache_lock(stem_dir):
        if separation_cache_is_valid(
            stem_dir,
            run.input_hash,
            tasks,
            config.bs_sample_rate,
        ):
            cache_hit = True
            LOGGER.info(
                "\n== Reusing cached BS-RoFormer stems ==\n"
                f"  cache: {cache_id}",
            )
        else:
            input_dir = stage_input(
                run.source,
                run.input_hash,
                sample_rate=config.bs_sample_rate,
                runner=run.runner,
                log_dir=run.log_dir,
            )
            try:
                with global_separation_admission(run.runner.stop_event):
                    separate_stems(
                        input_dir, stem_dir, run.log_dir, run.runner
                    )
                for task in tasks:
                    generated = stem_dir / f"{run.base_name}_{task.name}.wav"
                    if generated.is_file() and generated != task.audio:
                        generated.replace(task.audio)
                generated_instrumental = (
                    stem_dir / f"{run.base_name}_instrumental.wav"
                )
                generated_instrumental.unlink(missing_ok=True)
            finally:
                cleanup_staged_input(input_dir)
            for task in tasks:
                require_file(task.audio, f"{task.name} stem")

    run.metadata["separation"] = {
        "mode": "bs_roformer",
        "stem_directory": str(stem_dir),
        "cache_id": cache_id,
        "cache_hit": cache_hit,
        "input_sha256": run.input_hash,
        "sample_rate": config.bs_sample_rate,
    }
    for task in tasks:
        run.metadata["stems"][task.name] = {
            "separated_audio": (
                audio_file_info(task.audio)
                if task.audio.is_file()
                else {"path": str(task.audio), "missing": True}
            ),
            "selected_for_transcription": True,
        }
    return tasks


def select_stems(
    separated_tasks: list[StemTask],
    *,
    run: InitializedRun,
) -> list[StemTask]:
    """Apply only-other, combine-band, and explicit skip selection."""
    config = run.config
    if config.only_other:
        tasks = [task for task in separated_tasks if task.name == "other"]
    elif config.combine_band:
        band_task = make_band_task(
            separated_tasks,
            run.output_dir,
            run.base_name,
            config.combine_band,
        )
        LOGGER.info("\n== Band stems will be combined after RAM admission ==")
        tasks = [
            band_task,
            *(task for task in separated_tasks if task.name not in config.combine_band),
        ]
    else:
        tasks = list(separated_tasks)
    tasks = [task for task in tasks if task.name not in config.skip_stems]
    selected_names = {task.name for task in tasks}
    for name, stem_metadata in run.metadata["stems"].items():
        if name not in selected_names:
            stem_metadata["selected_for_transcription"] = False
            stem_metadata["selection_reason"] = (
                "explicit_skip"
                if name in config.skip_stems
                else (
                    "aggregated_into_band"
                    if name in config.combine_band
                    else "replaced_by_combine_band"
                )
            )
    for task in tasks:
        if task.name not in run.metadata["stems"]:
            run.metadata["stems"][task.name] = {
                "source_audio": (
                    audio_file_info(task.audio)
                    if task.audio.is_file()
                    else {"path": str(task.audio), "missing": True}
                ),
                "selected_for_transcription": True,
            }
    for task in tasks:
        require_file(task.audio, f"{task.name} stem")
    return tasks


def transcribe_stems(
    tasks: list[StemTask],
    *,
    run: InitializedRun,
) -> dict[str, TranscriptionResult]:
    """Run or resolve per-stem transcription outputs."""
    config = run.config
    LOGGER.info(f"Default transcriber: {config.default_transcriber}")
    if config.transcribers:
        LOGGER.info(
            "Transcriber overrides: "
            + ",".join(f"{stem}:{spec}" for stem, spec in config.transcribers),
        )
    LOGGER.info(
        "MuScriptor dynamic batch limit: %s",
        config.batch_size or "auto",
    )
    for name, value in sorted(config.backend_options.items()):
        rendered = ("on" if value else "off") if isinstance(value, bool) else value
        LOGGER.info("Backend %s: %s", name, rendered)
    if config.backend_options.get("muscriptor_runtime") == "cuda-graphs":
        LOGGER.info(
            "MuScriptor CUDA Graphs: on (bucket %d, cache %d)",
            DEFAULT_GRAPH_BUCKET_SIZE,
            DEFAULT_GRAPH_CACHE_SIZE,
        )
    LOGGER.info(f"Parallel workers: {config.parallelism}")
    if config.gpus:
        LOGGER.info(
            "Transcription GPUs: " + ",".join(str(gpu) for gpu in config.gpus),
        )
    presence_metrics = {
        task.name: run.metadata["stems"]
        .get(task.name, {})
        .get(
            "presence_metrics",
            {},
        )
        for task in tasks
    }
    queue_tasks = order_tasks_for_queue(
        tasks,
        args=config,
        presence_metrics=presence_metrics,
    )
    queue_order = [task.name for task in queue_tasks]
    LOGGER.info(
        "\n== Starting transcription queue ==\n  order: " + " -> ".join(queue_order),
    )
    timings: dict[str, dict[str, object]] = {}
    scheduler_timings: list[dict[str, object]] = []
    preprocessing_metrics: dict[str, dict[str, object]] = {}
    device_assignment: dict[str, object] = {}
    started = time.monotonic()
    try:
        results = run_queue(
            queue_tasks,
            parallelism=config.parallelism,
            args=config,
            device_job_slots=run.device_job_slots,
            out_dir=run.output_dir,
            log_dir=run.log_dir,
            runner=run.runner,
            timings=timings,
            scheduler_timings=scheduler_timings,
            session_id=run.run_prefix,
            audio_features=run.context.audio_features,
            tempo_reports=run.context.tempo_reports,
            preprocessing_metrics=preprocessing_metrics,
            device_assignment_sink=device_assignment,
        )
    finally:
        from ..scheduler.host_memory import global_host_memory_budget

        run.metadata["host_memory"] = {
            **global_host_memory_budget(config.host_memory_budget_mib).snapshot(),
            "stems": preprocessing_metrics,
        }
        for stem, metrics in preprocessing_metrics.items():
            run.metadata["stems"].setdefault(stem, {}).update(metrics)
    elapsed = time.monotonic() - started
    telemetry_path = run.artifacts.path("scheduler-telemetry.json.gz")
    telemetry, scheduler_summary = write_scheduler_telemetry(
        scheduler_timings,
        telemetry_path,
        stem_artifacts={
            stem: timing.get("artifacts", {}) for stem, timing in timings.items()
        },
        relative_to=run.output_dir,
        registry=run.artifacts,
    )
    waterfall = None

    timing_summaries: dict[str, dict[str, object]] = {}
    for stem, timing in timings.items():
        artifacts = timing.get("artifacts", {})
        activity = artifacts.get("activity_regions", {})
        trace_artifacts = [
            artifact
            for artifact in artifacts.get("artifacts", [])
            if artifact.get("kind") == "muscriptor_model_trace"
        ]
        timing_summary = {
            "elapsed_seconds": timing.get("elapsed_seconds"),
            "gpu": timing.get("gpu"),
            "activity": {
                "enabled": activity.get("enabled"),
                "mode": activity.get("mode"),
                **activity.get("summary", {}),
            },
            "execution_profiles": artifacts.get("execution_profiles", {}),
            "model_traces": trace_artifacts,
        }
        timing_summary["memory_artifacts"] = [
            artifact.name for artifact in results[stem].artifacts
        ]
        timing_summaries[stem] = timing_summary
    run.metadata["transcription_timing"] = {
        "elapsed_seconds": round(elapsed, 3),
        "parallelism": config.parallelism,
        "gpus": list(config.gpus),
        "device_assignment": device_assignment,
        "scheduler": scheduler_summary,
        "scheduler_waterfall": waterfall,
        "queue_order": queue_order,
        "stems": timing_summaries,
    }
    run.metadata["telemetry"] = telemetry
    LOGGER.info(f"Transcription queue elapsed: {elapsed:.3f}s")
    for task in tasks:
        spec = transcriber_for_stem(config, task.name)
        engine, model = split_transcriber_spec(spec)
        run.metadata["stems"].setdefault(task.name, {})["transcription"] = {
            "spec": spec,
            "engine": engine,
            "model": model,
            "input_audio": str(task.audio),
            "event_count": len(results[task.name].events),
            "output": "memory",
        }
    return results


def package_transcriber_outputs(
    batch: PostprocessBatchResult,
    *,
    run: InitializedRun,
) -> dict[str, object] | None:
    """Publish one archive after all consumers finish reading stem outputs."""
    if not run.config.publish_files:
        return None
    descriptor = archive_memory_transcriptions(
        run.artifacts,
        results=batch.stems,
        models={
            stem: split_transcriber_spec(
                transcriber_for_stem(run.config, stem)
            )[1] or "default"
            for stem in batch.stems
        },
        provenance={
            "input_sha256": run.input_hash,
            "effective_flags": run.metadata.get("effective_flags", {}),
        },
    )
    run.metadata["transcription_timing"]["transcriber_archive"] = descriptor
    return descriptor


def run_pipeline(
    args: PipelineConfig,
    *,
    session_id: str | None = None,
    stop_event: threading.Event | None = None,
    device_job_slots: int = 1,
) -> PipelineRunOutcome:
    """Execute one session through the transport-independent job boundary."""
    resolved_session_id = session_id or make_run_prefix()
    with bind_run_logging(resolved_session_id):
        return _run_pipeline(
            args,
            session_id=resolved_session_id,
            stop_event=stop_event,
            device_job_slots=device_job_slots,
        )


def _run_pipeline(
    args: PipelineConfig,
    *,
    session_id: str,
    stop_event: threading.Event | None = None,
    device_job_slots: int = 1,
) -> PipelineRunOutcome:
    """Execute the pipeline by composing independently testable phases."""
    context = prepare_pipeline_context(
        args,
        run_prefix=session_id,
        stop_event=stop_event,
        device_job_slots=device_job_slots,
    )
    started = time.monotonic()
    gpu_monitor: GpuTelemetryMonitor | None = None
    outputs: PipelineOutputs | None = None
    failure_error: BaseException | None = None
    exit_code = 1
    from ..observability.global_waterfall import global_waterfall_store

    waterfall_store = global_waterfall_store()
    waterfall_store.begin_session(
        context.run_prefix,
        label=Path(args.input_file).stem,
    )

    @contextmanager
    def phase(action: str, *, resource: str = "cpu", **details: object):
        phase_started = time.time_ns()
        try:
            yield
        finally:
            recorded_action = action
            if action == "separation" and context.metadata.get("separation", {}).get(
                "cache_hit"
            ):
                recorded_action = "separation_cache_hit"
            waterfall_store.record_span(
                recorded_action,
                phase_started,
                time.time_ns(),
                pipeline_session_id=context.run_prefix,
                resource=resource,
                lane=f"{Path(args.input_file).stem} / {action}",
                **details,
            )

    try:
        run = initialize_run(context)
        if args.gpu_telemetry_interval_ms and context.output_dir is not None:
            assert context.log_dir is not None
            assert context.artifacts is not None
            gpu_monitor = GpuTelemetryMonitor.start(
                context.artifacts.directory,
                context.log_dir,
                context.run_prefix,
                interval_ms=args.gpu_telemetry_interval_ms,
                gpu_ids=args.gpus,
            )
        with phase("separation", resource="gpu", model="BS-RoFormer"):
            separated_tasks = prepare_separated_stems(run)
        with phase("stem_selection"):
            tasks = select_stems(separated_tasks, run=run)
        with phase("presence_analysis"):
            tasks = preprocess_stems(tasks, run=run)
        with phase("transcription", resource="gpu"):
            transcriptions = transcribe_stems(tasks, run=run)
        with phase("midi_postprocess"):
            batch = postprocess_in_memory(
                transcriptions,
                tasks,
                config=args,
                audio_features=context.audio_features,
                tempo_reports=context.tempo_reports,
            )
        context.tempo_result = batch.tempo
        if batch.tempo is not None:
            context.metadata["tempo"] = batch.tempo
        with phase("output"):
            outputs = produce_outputs(batch, tasks, run=run)
            package_transcriber_outputs(batch, run=run)
        LOGGER.info("\nDone.")
        LOGGER.info(f"MIDI directory: {context.output_dir}")
        LOGGER.info(f"Logs:           {context.log_dir}")
        if outputs.midi_path is not None:
            LOGGER.info(f"Merged MIDI:    {outputs.midi_path}")
        if outputs.mp3_path is not None:
            LOGGER.info(f"Mixdown:        {outputs.mp3_path}")
        context.metadata["status"] = "complete"
        LOGGER.info("run %s completed", context.run_prefix)
        exit_code = 0
    except Cancelled:
        context.metadata["status"] = "cancelled"
        context.metadata["error"] = {
            "type": "Cancelled",
            "message": "Cancelled",
        }
        LOGGER.warning("run %s cancelled", context.run_prefix)
        LOGGER.warning("Cancelled.")
        exit_code = 130
    except Exception as error:  # noqa: BLE001 - persist job failures at boundary
        failure_error = error
        context.metadata["status"] = "failed"
        context.metadata["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        LOGGER.exception("run %s failed", context.run_prefix)
        LOGGER.error("%s", error)
        exit_code = 1
    finally:
        waterfall_store.end_session(
            context.run_prefix,
            status=str(context.metadata.get("status", "failed")),
        )
        if gpu_monitor is not None:
            try:
                context.metadata["gpu_telemetry"] = gpu_monitor.finalize(
                    context.artifacts
                )
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                LOGGER.warning("Could not finalize GPU telemetry: %s", error)
        liveness_report = _scheduler_liveness_report(failure_error)
        if isinstance(liveness_report, dict) and context.artifacts is not None:
            try:
                context.metadata["scheduler_liveness"] = write_scheduler_liveness(
                    liveness_report,
                    context.artifacts,
                )
            except (OSError, TypeError, ValueError) as error:
                LOGGER.warning("Could not persist scheduler liveness evidence: %s", error)
        if context.artifacts is not None:
            context.metadata["artifacts"] = context.artifacts.descriptors
        context.metadata["finished_at"] = datetime.now().astimezone().isoformat()
        elapsed_seconds = time.monotonic() - started
        context.metadata["elapsed_seconds"] = round(elapsed_seconds, 6)
        input_duration = float(
            context.metadata.get("input", {}).get("duration_seconds", 0)
        )
        output_notes = 0
        if outputs is not None:
            output_notes = sum(
                1
                for track in outputs.final_midi.to_mido().tracks
                for message in track
                if message.type == "note_on" and message.velocity > 0
            )
        generated_tokens = int(
            context.metadata.get("transcription_timing", {})
            .get("scheduler", {})
            .get("generated_tokens", 0)
        )
        performance = (
            calculate_performance(
                audio_seconds=input_duration,
                wall_seconds=elapsed_seconds,
                output_notes=output_notes,
                generated_tokens=generated_tokens,
                transcription_seconds=context.metadata.get(
                    "transcription_timing", {}
                ).get("elapsed_seconds"),
            )
            if context.metadata.get("status") == "complete"
            else None
        )
        if performance is not None:
            context.metadata["performance"] = performance.to_dict()
            LOGGER.info(
                "Throughput: %.2f notes/s, %.2f tokens/s, %.2f s/audio-min, "
                "%s transcription-only realtime",
                performance.notes_per_second,
                performance.generated_tokens_per_second,
                performance.processing_seconds_per_audio_minute,
                f"{performance.transcription_realtime:.3f}x"
                if performance.transcription_realtime is not None
                else "unknown",
            )
        metadata_written = False
        try:
            written_metadata = write_run_metadata(
                context.metadata,
                context.metadata_path,
            )
            metadata_written = True
            LOGGER.info(f"Metadata:       {written_metadata}")
        except (OSError, TypeError, ValueError) as metadata_error:
            LOGGER.warning("Could not write metadata: %s", metadata_error)
        context.runner.request_stop()
        if context.staged_input_dir is not None:
            try:
                cleanup_staged_input(context.staged_input_dir)
            except (OSError, RuntimeError) as cleanup_error:
                LOGGER.warning("Could not remove staged input: %s", cleanup_error)
        if context.log_dir is not None:
            try:
                log_result = finalize_run_logs(
                    context.log_dir,
                    status=str(context.metadata["status"]),
                    retention=context.config.log_retention,
                )
                if context.output_dir is not None:
                    log_result["directory"] = context.log_dir.relative_to(
                        context.output_dir
                    ).as_posix()
                context.metadata.setdefault("outputs", {})["logs"] = log_result
                if metadata_written:
                    write_run_metadata(context.metadata, context.metadata_path)
            except (OSError, ValueError) as log_error:
                context.metadata.setdefault("outputs", {})["logs"] = {
                    "retained": True,
                    "policy": context.config.log_retention,
                    "finalization_error": str(log_error),
                }
                if metadata_written:
                    write_run_metadata(context.metadata, context.metadata_path)
        if context.transient_output_dir and context.output_dir is not None:
            shutil.rmtree(context.output_dir, ignore_errors=True)
    published = context.config.publish_files and context.output_dir is not None
    return PipelineRunOutcome(
        session_id=context.run_prefix,
        status=str(context.metadata.get("status", "failed")),
        exit_code=exit_code,
        output_dir=context.output_dir if published else None,
        metadata_path=context.metadata_path if published else None,
        performance=performance,
        error=(
            None
            if context.metadata.get("error") is None
            else str(context.metadata["error"].get("message", "pipeline failed"))
        ),
    )
