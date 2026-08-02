"""Stem gate, presence filtering, and hard-zero preprocessing phases."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..adaptive_volume_gate import adaptive_volume_gate, hard_zero_audio_blocks
from ..stem_presence_filter import analyze_stem_presence, assess_stem_presence
from .audio_staging import require_file
from .types import PipelineConfig, PipelineContext, StemTask

LOGGER = logging.getLogger(__name__)

def gate_preset_for_task(task: StemTask, config: PipelineConfig) -> str:
    """Return the explicitly configured per-stem adaptive gate preset."""
    preset = config.gate_preset_for_stem(task.name)
    if preset is None:
        raise ValueError(f"no adaptive gate preset configured for {task.name}")
    return preset


def gated_audio_path(task: StemTask, config: PipelineConfig) -> Path:
    preset = gate_preset_for_task(task, config)
    if config.gate_threshold_dbfs is not None:
        threshold_tag = f"{config.gate_threshold_dbfs:g}dbfs"
    elif config.gate_threshold_offset_db is not None:
        threshold_tag = f"offset{config.gate_threshold_offset_db:+g}db"
    elif preset == "extreme":
        threshold_tag = "+6db"
    else:
        threshold_tag = "auto"
    zero_tag = (
        f"-zero{config.hard_zero_threshold_dbfs:g}dbfs"
        if task.name in config.hard_zero_stems
        else ""
    )
    return task.audio.with_name(
        f"{task.audio.stem}_gate-{preset}-{threshold_tag}{zero_tag}{task.audio.suffix}"
    )


def hard_zero_audio_path(task: StemTask, config: PipelineConfig) -> Path:
    return task.audio.with_name(
        f"{task.audio.stem}_zero{config.hard_zero_threshold_dbfs:g}dbfs"
        f"{task.audio.suffix}"
    )


def direct_preprocessed_audio_path(
    task: StemTask,
    generated: Path,
    context: PipelineContext,
) -> Path:
    """Keep derivatives of a directly used input inside its output directory."""
    if not context.config.treat_original_as_other:
        return generated
    if context.output_dir is None:
        raise RuntimeError("pipeline context has not been initialized")
    return context.output_dir / generated.with_suffix(".wav").name


def apply_adaptive_gates(
    tasks: list[StemTask],
    *,
    context: PipelineContext,
) -> list[StemTask]:
    config = context.config
    if not config.adaptive_gate_presets:
        return tasks
    selected_stems = dict(config.adaptive_gate_presets)
    LOGGER.info(
        "\n== Applying per-stem adaptive gates: "
        + ",".join(f"{stem}:{preset}" for stem, preset in config.adaptive_gate_presets)
        + " ==",
    )
    selected_tasks = [task for task in tasks if task.name in selected_stems]

    def apply_gate(task: StemTask) -> tuple[StemTask, Path, Any] | None:
        preset = gate_preset_for_task(task, config)
        gated = direct_preprocessed_audio_path(
            task,
            gated_audio_path(task, config),
            context,
        )
        if config.mix_only:
            return task, gated, None
        result = adaptive_volume_gate(
            task.audio,
            gated,
            preset=preset,
            threshold_offset_db=config.gate_threshold_offset_db,
            threshold_dbfs=config.gate_threshold_dbfs,
            zero_below_dbfs=(
                config.hard_zero_threshold_dbfs
                if task.name in config.hard_zero_stems
                else None
            ),
        )
        require_file(gated, f"Gated {task.name} stem")
        return task, gated, result

    worker_count = min(config.preprocess_jobs, len(selected_tasks))
    if worker_count:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="gate-worker",
        ) as executor:
            results = list(executor.map(apply_gate, selected_tasks))
    else:
        results = []

    for completed in results:
        if completed is None:
            continue
        task, gated, result = completed
        task.audio = gated
        if result is None:
            continue
        context.metadata["stems"].setdefault(task.name, {})["adaptive_gate"] = (
            result.to_dict()
        )
        LOGGER.info(
            f"  {task.name:<7} threshold "
            f"{result.threshold_dbfs:.2f} dBFS "
            f"({result.threshold_mode}); "
            f"{result.blocks_reduced_over_6db_percent:.1f}% "
            f"blocks reduced >6 dB -> {gated}",
        )
    return tasks


def filter_absent_stems(
    tasks: list[StemTask],
    *,
    context: PipelineContext,
) -> list[StemTask]:
    config = context.config
    if not config.gate_presence_filter:
        return tasks
    if not tasks:
        return tasks
    worker_count = min(config.preprocess_jobs, len(tasks))
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="presence-worker",
    ) as executor:
        analyzed = executor.map(
            analyze_stem_presence,
            (task.audio for task in tasks),
        )
        metrics = {
            task.name: metric
            for task, metric in zip(tasks, analyzed, strict=True)
        }
    for task in tasks:
        context.metadata["stems"].setdefault(task.name, {})["presence_metrics"] = (
            metrics[task.name].to_dict()
        )
    kept: list[StemTask] = []
    LOGGER.info("\n== Assessing stem presence ==")
    for task in tasks:
        decision = assess_stem_presence(metrics[task.name], list(metrics.values()))
        current = decision.metrics
        status = "keep" if decision.present else "skip as absent"
        if decision.present:
            kept.append(task)
        stem_metadata = context.metadata["stems"].setdefault(task.name, {})
        stem_metadata["presence"] = decision.to_dict()
        stem_metadata["selected_for_transcription"] = decision.present
        LOGGER.info(
            f"  {task.name:<7} {status}; RMS "
            f"{current.global_rms_dbfs:.2f} dBFS, p99 "
            f"{current.block_p99_dbfs:.2f} dBFS, active "
            f"{current.coverage_above_minus_40_percent:.3f}%, "
            f"loudest 200ms {current.loudest_200ms_dbfs:.2f} dBFS, "
            f"relative deficit {decision.relative_deficit_db:.2f} dB",
        )
    return kept


def apply_standalone_hard_zero(
    tasks: list[StemTask],
    *,
    context: PipelineContext,
) -> list[StemTask]:
    config = context.config
    selected = [
        task
        for task in tasks
        if task.name in config.hard_zero_stems
        and config.gate_preset_for_stem(task.name) is None
    ]
    if selected:
        LOGGER.info(
            "\n== Hard-zeroing selected stems "
            f"(< {config.hard_zero_threshold_dbfs:g} dBFS) ==",
        )
    for task in selected:
        zeroed = direct_preprocessed_audio_path(
            task,
            hard_zero_audio_path(task, config),
            context,
        )
        if config.mix_only:
            task.audio = zeroed
            continue
        result = hard_zero_audio_blocks(
            task.audio,
            zeroed,
            threshold_dbfs=config.hard_zero_threshold_dbfs,
        )
        require_file(zeroed, f"Hard-zeroed {task.name} stem")
        task.audio = zeroed
        context.metadata["stems"].setdefault(task.name, {})["hard_zero"] = result
        LOGGER.info(
            f"  {task.name:<7} "
            f"{result['blocks_zeroed_percent']:.1f}% blocks zeroed "
            f"-> {zeroed}",
        )
    return tasks


def preprocess_stems(
    tasks: list[StemTask],
    *,
    context: PipelineContext,
) -> list[StemTask]:
    tasks = apply_adaptive_gates(tasks, context=context)
    tasks = filter_absent_stems(tasks, context=context)
    if not tasks:
        raise RuntimeError("all selected stems were rejected as absent")
    return apply_standalone_hard_zero(tasks, context=context)





