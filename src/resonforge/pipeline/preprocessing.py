"""Bounded presence analysis before lazy per-stem transcription."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from resonforge.transcribers.base import StemTask

from ..audio.stem_presence_filter import (
    analyze_stem_presence_buffer,
    assess_stem_presence,
)
from ..scheduler.host_memory import global_host_memory_budget
from .audio_admission import estimate_working_bytes, load_task_audio
from .types import InitializedRun

LOGGER = logging.getLogger(__name__)


def _analyze_presence(task: StemTask, *, run: InitializedRun):
    budget = global_host_memory_budget(run.config.host_memory_budget_mib)
    requested = estimate_working_bytes(task)
    with budget.acquire(requested, stop_event=run.runner.stop_event) as reservation:
        audio = load_task_audio(task)
        metric = analyze_stem_presence_buffer(audio, identity=task.audio)
        return metric, reservation


def preprocess_stems(
    tasks: list[StemTask],
    *,
    run: InitializedRun,
) -> list[StemTask]:
    """Measure raw stems without retaining decoded WAV arrays."""
    if not run.config.gate_presence_filter:
        return tasks
    if not tasks:
        return tasks
    worker_count = min(run.config.preprocess_jobs, len(tasks))
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="presence-worker",
    ) as executor:
        analyzed = executor.map(
            lambda task: _analyze_presence(task, run=run),
            tasks,
        )
        results = {
            task.name: result
            for task, result in zip(tasks, analyzed, strict=True)
        }
    metrics = {stem: result[0] for stem, result in results.items()}
    kept: list[StemTask] = []
    LOGGER.info("\n== Assessing raw stem presence under global RAM budget ==")
    for task in tasks:
        metric, reservation = results[task.name]
        decision = assess_stem_presence(metric, list(metrics.values()))
        stem_metadata = run.metadata["stems"].setdefault(task.name, {})
        stem_metadata["presence_metrics"] = metric.to_dict()
        stem_metadata["presence"] = decision.to_dict()
        stem_metadata["presence_audio_memory"] = {
            "requested_bytes": reservation.requested_bytes,
            "admitted_bytes": reservation.admitted_bytes,
            "wait_seconds": round(reservation.wait_seconds, 6),
        }
        stem_metadata["selected_for_transcription"] = decision.present
        if decision.present:
            kept.append(task)
        current = decision.metrics
        LOGGER.info(
            "  %s %s; RMS %.2f dBFS, p99 %.2f dBFS, active %.3f%%",
            f"{task.name:<7}",
            "keep" if decision.present else "skip as absent",
            current.global_rms_dbfs,
            current.block_p99_dbfs,
            current.coverage_above_minus_40_percent,
        )
    # No "everything was absent" branch: absence is judged *relative* to the
    # other stems (`relative_deficit > 25 dB`), so whenever one stem is absent
    # another is necessarily present, and `kept` can never be empty. A run with
    # nothing to transcribe arrives here already empty — every stem skipped —
    # and the ordinary phases handle that.
    return kept
