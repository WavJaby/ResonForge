"""Phase-oriented orchestration for the audio-to-MIDI pipeline."""

from __future__ import annotations

import logging
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from ..logging_config import configure_run_logging
from ..run_metadata import audio_file_info, create_run_metadata, write_run_metadata
from .audio_staging import (
    cleanup_staged_input,
    inspect_input,
    require_file,
    stage_input,
)
from .midi_postprocessing import postprocess_midis
from .output_production import produce_outputs
from .paths import (
    BASIC_PITCH,
    BS_CONFIG_PATH,
    BS_MODEL_PATH,
    BS_OUTPUT_ROOT,
    MODELS_ROOT,
    PYTHON,
)
from .preprocessing import preprocess_stems
from .process_runner import Cancelled, ProcessRunner
from .separation import (
    BS_STEMS_DIRNAME,
    make_tasks,
    separate_stems,
    separation_cache_is_valid,
    write_separation_hash,
)
from .transcription import (
    order_tasks_for_queue,
    run_queue,
    split_transcriber_spec,
    transcribed_midi_path,
    transcriber_for_stem,
)
from .types import (
    PipelineConfig,
    PipelineContext,
    StemTask,
)

LOGGER = logging.getLogger(__name__)

def make_run_prefix(now: datetime | None = None) -> str:
    """Return the local timestamp shared by all final artifacts in one run."""
    return (now or datetime.now().astimezone()).strftime("%y%m%d-%H%M%S")


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
    )


def combine_band_stems(
    tasks: list[StemTask],
    output: Path,
    band_names: frozenset[str] = frozenset({"bass", "guitar", "piano"}),
) -> None:
    """Mix the requested stems into one floating-point WAV."""
    band_sources = [task for task in tasks if task.name in band_names]
    if len(band_sources) != len(band_names):
        raise RuntimeError("could not resolve all requested band stems")
    sources = [task.audio for task in band_sources]

    sample_rate = None
    audio_tracks = []
    for source in sources:
        require_file(source, f"{source.stem} stem")
        audio, current_rate = sf.read(source, dtype="float32", always_2d=True)
        if sample_rate is None:
            sample_rate = current_rate
        elif current_rate != sample_rate:
            raise RuntimeError(
                f"band stems have mismatched sample rates: "
                f"{sample_rate} and {current_rate}"
            )
        audio_tracks.append(audio)

    channels = {audio.shape[1] for audio in audio_tracks}
    if len(channels) != 1:
        raise RuntimeError("band stems have mismatched channel counts")
    length = max(len(audio) for audio in audio_tracks)
    combined = np.zeros((length, audio_tracks[0].shape[1]), dtype=np.float32)
    for audio in audio_tracks:
        combined[: len(audio)] += audio
    peak = float(np.max(np.abs(combined)))
    adjustment_db = 0.0
    if peak > 0.99:
        adjustment = 0.99 / peak
        combined *= adjustment
        adjustment_db = 20.0 * np.log10(adjustment)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, combined, sample_rate, subtype="FLOAT")
    combined_names = ", ".join(task.name for task in band_sources)
    LOGGER.info(
        f"  combined {combined_names} -> {output} "
        f"(peak adjustment {adjustment_db:+.2f} dB)",
    )


def prepare_pipeline_context(args: PipelineConfig) -> PipelineContext:
    """Create typed run configuration and run-scoped mutable state."""
    config = args
    run_prefix = make_run_prefix()
    metadata = create_run_metadata(run_prefix, args)
    metadata_path = (
        BS_OUTPUT_ROOT / Path(config.input_file).stem / (f"{run_prefix}-metadata.json")
    )
    return PipelineContext(
        config=config,
        run_prefix=run_prefix,
        metadata=metadata,
        metadata_path=metadata_path,
        runner=ProcessRunner(
            threading.Event(),
            model_cache=MODELS_ROOT / "huggingface" / "hub",
        ),
    )


def initialize_run(context: PipelineContext) -> None:
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
    if any(spec == "basic-pitch" for _, spec in config.transcribers):
        require_file(BASIC_PITCH, "Basic Pitch executable")

    source, output_dir, base_name, input_hash = inspect_input(config.input_file)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_paths = configure_run_logging(output_dir, log_dir, context.run_prefix)
    from muscriptor.utils.download import configure_huggingface_download_logging

    configure_huggingface_download_logging()
    LOGGER.info("starting run %s", context.run_prefix)
    LOGGER.info("input: %s", source)
    LOGGER.info("output directory: %s", output_dir)
    context.source = source
    context.output_dir = output_dir
    context.log_dir = log_dir
    context.base_name = base_name
    context.input_hash = input_hash
    context.metadata_path = output_dir / f"{context.run_prefix}-metadata.json"
    context.metadata["input"] = {**audio_file_info(source), "sha256": input_hash}
    context.metadata["outputs"] = {
        "directory": str(output_dir),
        "metadata": str(context.metadata_path),
        "logs": {
            "pipeline": str(log_paths.pipeline),
            "huggingface": str(log_paths.huggingface),
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
            "cfg_coef": config.cfg_coef,
            "beam_size": config.beam_size,
        },
    }
    LOGGER.info(f"Input file:      {source}")
    LOGGER.info(f"Output directory: {output_dir}")


def prepare_separated_stems(context: PipelineContext) -> list[StemTask]:
    """Resolve direct-original mode or create BS-RoFormer stems."""
    if (
        context.output_dir is None
        or context.log_dir is None
        or context.base_name is None
        or context.input_hash is None
        or context.source is None
    ):
        raise RuntimeError("pipeline context has not been initialized")
    config = context.config
    if config.treat_original_as_other:
        input_dir = stage_input(
            context.source,
            context.input_hash,
            sample_rate=config.bs_sample_rate,
            runner=context.runner,
            log_dir=context.log_dir,
        )
        context.staged_input_dir = input_dir
        staged_audio = input_dir / f"{context.source.stem}.wav"
        require_file(staged_audio, "Staged original input")
        task = StemTask("other", staged_audio, config.other_instruments)
        context.metadata["separation"] = {
            "mode": "original_as_other",
            "skipped": True,
            "input_sha256": context.input_hash,
            "sample_rate": config.bs_sample_rate,
            "staged_audio": str(staged_audio),
        }
        context.metadata["stems"]["other"] = {
            "separated_audio": audio_file_info(task.audio),
            "selected_for_transcription": True,
            "source": "original_input",
        }
        LOGGER.info(
            "\n== Treating original input as other ==\n"
            f"  {context.source} -> {task.audio}\n"
            "  BS-RoFormer skipped",
            )
        return [task]

    stem_dir = context.output_dir / BS_STEMS_DIRNAME
    tasks = make_tasks(stem_dir, context.base_name, config.other_instruments)
    cache_hit = False
    if not config.mix_only:
        if separation_cache_is_valid(
            stem_dir,
            context.input_hash,
            tasks,
            config.bs_sample_rate,
        ):
            cache_hit = True
            LOGGER.info(
                "\n== Reusing cached BS-RoFormer stems ==\n"
                f"  input SHA-256: {context.input_hash}",
                    )
        else:
            input_dir = stage_input(
                context.source,
                context.input_hash,
                sample_rate=config.bs_sample_rate,
                runner=context.runner,
                log_dir=context.log_dir,
            )
            try:
                separate_stems(input_dir, stem_dir, context.log_dir, context.runner)
            finally:
                cleanup_staged_input(input_dir)
            for task in tasks:
                require_file(task.audio, f"{task.name} stem")
            hash_file = write_separation_hash(
                stem_dir,
                context.input_hash,
                config.bs_sample_rate,
            )
            LOGGER.info(f"  separation cache hash -> {hash_file}")

    context.metadata["separation"] = {
        "mode": "bs_roformer",
        "stem_directory": str(stem_dir),
        "cache_hit": cache_hit,
        "skipped": bool(config.mix_only),
        "input_sha256": context.input_hash,
        "sample_rate": config.bs_sample_rate,
    }
    for task in tasks:
        context.metadata["stems"][task.name] = {
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
    context: PipelineContext,
) -> list[StemTask]:
    """Apply only-other, combine-band, and explicit skip selection."""
    if context.output_dir is None or context.base_name is None:
        raise RuntimeError("pipeline context has not been initialized")
    config = context.config
    if config.only_other:
        tasks = [task for task in separated_tasks if task.name == "other"]
    elif config.combine_band:
        band_task = make_band_task(
            separated_tasks,
            context.output_dir,
            context.base_name,
            config.combine_band,
        )
        if not config.mix_only:
            LOGGER.info("\n== Combining band stems ==")
            combine_band_stems(
                separated_tasks,
                band_task.audio,
                config.combine_band,
            )
        tasks = [
            band_task,
            *(task for task in separated_tasks if task.name not in config.combine_band),
        ]
    else:
        tasks = list(separated_tasks)
    tasks = [task for task in tasks if task.name not in config.skip_stems]
    selected_names = {task.name for task in tasks}
    for name, stem_metadata in context.metadata["stems"].items():
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
        if task.name not in context.metadata["stems"]:
            context.metadata["stems"][task.name] = {
                "source_audio": (
                    audio_file_info(task.audio)
                    if task.audio.is_file()
                    else {"path": str(task.audio), "missing": True}
                ),
                "selected_for_transcription": True,
            }
    if not config.mix_only:
        for task in tasks:
            require_file(task.audio, f"{task.name} stem")
    return tasks


def transcribe_stems(
    tasks: list[StemTask],
    *,
    context: PipelineContext,
) -> dict[str, Path]:
    """Run or resolve per-stem transcription outputs."""
    if context.output_dir is None or context.log_dir is None:
        raise RuntimeError("pipeline context has not been initialized")
    config = context.config
    if config.mix_only:
        return {
            task.name: transcribed_midi_path(
                task,
                args=config,
                out_dir=context.output_dir,
            )
            for task in tasks
        }

    LOGGER.info(f"Default transcriber: {config.default_transcriber}")
    if config.transcribers:
        LOGGER.info(
            "Transcriber overrides: "
            + ",".join(f"{stem}:{spec}" for stem, spec in config.transcribers),
            )
    LOGGER.info(f"Beam size: {config.beam_size}")
    LOGGER.info(f"MuScriptor batch size: {config.batch_size}")
    LOGGER.info(f"Parallel workers: {config.parallelism}")
    if config.gpus:
        LOGGER.info(
            "Transcription GPUs: " + ",".join(str(gpu) for gpu in config.gpus),
            )
    presence_metrics = {
        task.name: context.metadata["stems"]
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
    started = time.monotonic()
    midi_paths = run_queue(
        queue_tasks,
        parallelism=config.parallelism,
        args=config,
        out_dir=context.output_dir,
        log_dir=context.log_dir,
        runner=context.runner,
        timings=timings,
    )
    elapsed = time.monotonic() - started
    context.metadata["transcription_timing"] = {
        "elapsed_seconds": round(elapsed, 3),
        "parallelism": config.parallelism,
        "gpus": list(config.gpus),
        "queue_order": queue_order,
        "stems": timings,
    }
    LOGGER.info(f"Transcription queue elapsed: {elapsed:.3f}s")
    for task in tasks:
        spec = transcriber_for_stem(config, task.name)
        engine, model = split_transcriber_spec(spec)
        context.metadata["stems"].setdefault(task.name, {})["transcription"] = {
            "spec": spec,
            "engine": engine,
            "model": model,
            "input_audio": str(task.audio),
            "output_midi": str(midi_paths[task.name]),
            "artifacts": timings.get(task.name, {}).get("artifacts", {}),
        }
    return midi_paths


def run_pipeline(args: PipelineConfig) -> int:
    """Execute the pipeline by composing independently testable phases."""
    context = prepare_pipeline_context(args)
    started = time.monotonic()

    def handle_signal(_signum: int, _frame: object) -> None:
        LOGGER.warning("Stopping workers and child processes...")
        context.runner.stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    try:
        initialize_run(context)
        separated_tasks = prepare_separated_stems(context)
        tasks = select_stems(separated_tasks, context=context)
        tasks = preprocess_stems(tasks, context=context)
        midi_paths = transcribe_stems(tasks, context=context)
        midi_paths = postprocess_midis(
            midi_paths,
            tasks,
            separated_tasks,
            context=context,
        )
        outputs = produce_outputs(midi_paths, tasks, context=context)
        LOGGER.info("\nDone.")
        LOGGER.info(f"MIDI directory: {context.output_dir}")
        LOGGER.info(f"Logs:           {context.log_dir}")
        if outputs.midi is not None:
            LOGGER.info(f"Merged MIDI:    {outputs.midi}")
        if outputs.mp3 is not None:
            LOGGER.info(f"Mixdown:        {outputs.mp3}")
        context.metadata["status"] = "complete"
        LOGGER.info("run %s completed", context.run_prefix)
        return 0
    except Cancelled:
        context.metadata["status"] = "cancelled"
        context.metadata["error"] = {
            "type": "Cancelled",
            "message": "Cancelled",
        }
        LOGGER.warning("run %s cancelled", context.run_prefix)
        LOGGER.warning("Cancelled.")
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        context.metadata["status"] = "failed"
        context.metadata["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        LOGGER.exception("run %s failed", context.run_prefix)
        LOGGER.error("%s", error)
        return 1
    finally:
        context.metadata["finished_at"] = datetime.now().astimezone().isoformat()
        context.metadata["elapsed_seconds"] = round(
            time.monotonic() - started,
            6,
        )
        try:
            written_metadata = write_run_metadata(
                context.metadata,
                context.metadata_path,
            )
            LOGGER.info(f"Metadata:       {written_metadata}")
        except (OSError, TypeError, ValueError) as metadata_error:
            LOGGER.warning("Could not write metadata: %s", metadata_error)
        context.runner.request_stop()
        if context.staged_input_dir is not None:
            try:
                cleanup_staged_input(context.staged_input_dir)
            except (OSError, RuntimeError) as cleanup_error:
                LOGGER.warning("Could not remove staged input: %s", cleanup_error)


