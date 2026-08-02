"""Local and cross-stem MIDI postprocessing phase."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import soundfile as sf

from ..midi.cleanup import (
    clamp_midi_duration,
    clean_retrigger_loops,
    filter_excessive_onset_bursts,
    truncate_overlapping_same_pitch_notes,
)
from ..midi.jsonl import DRUMS, GM, convert_jsonl_to_midi, rewrite_midi_program
from ..midi.quiet_onsets import filter_quiet_onsets
from .muscriptor_tempo import collect_muscriptor_tempo_reports, select_tempo_grid
from .transcription import split_transcriber_spec, transcriber_for_stem
from .types import PipelineContext, StemTask

LOGGER = logging.getLogger(__name__)

MAXIMUM_SIMULTANEOUS_ONSETS = {
    "bass": 6,
    "vocals": 5,
    "guitar": 6,
    "drums": 8,
    "piano": 10,
    "other": 12,
    "band": 16,
}


def postprocess_midis(
    midi_paths: dict[str, Path],
    tasks: list[StemTask],
    separated_tasks: list[StemTask],
    *,
    context: PipelineContext,
) -> dict[str, Path]:
    """Create local-clean MIDI, then optional cross-stem global-clean MIDI."""
    config = context.config
    if config.mix_only:
        return midi_paths
    if context.output_dir is None or context.log_dir is None:
        raise RuntimeError("pipeline context has not been initialized")

    LOGGER.info("\n== Creating local-clean MIDI ==")
    local_targets: dict[str, Path] = {}
    for task in tasks:
        raw_midi = midi_paths[task.name]
        backend, _ = split_transcriber_spec(transcriber_for_stem(config, task.name))
        base = raw_midi.name.removesuffix(".raw.midi")
        clean_target = raw_midi.with_name(base + ".clean.midi")
        clean_midi = (
            context.log_dir / f"{context.run_prefix}-{task.name}.local-clean.work.midi"
        )
        clean_midi.unlink(missing_ok=True)
        local_targets[task.name] = clean_target
        local_metadata = (
            context.metadata["stems"]
            .setdefault(task.name, {})
            .setdefault("local_cleanup", {})
        )
        local_metadata["status"] = "running"

        if backend == "muscriptor":
            raw_json = raw_midi.with_name(base + ".raw.json")
            stderr_log = context.log_dir / f"{task.name}.muscriptor.stderr.log"
            conversion_report: dict[str, object] = {}
            convert_jsonl_to_midi(
                raw_json,
                drop_eos_log=(stderr_log if config.midi_cleanup else None),
                out=clean_midi,
                cleanup=config.midi_cleanup,
                verbose=False,
                report=conversion_report,
            )
            local_metadata["conversion_cleanup"] = conversion_report
        else:
            shutil.copy2(raw_midi, clean_midi)

        instruments = [
            instrument.strip()
            for instrument in task.instruments.split(",")
            if instrument.strip()
        ]
        if backend in {"basic-pitch", "mt3"} and len(instruments) == 1:
            instrument = instruments[0]
            if instrument in GM or instrument in DRUMS:
                rewrite_midi_program(clean_midi, instrument)
                local_metadata["program_assignment"] = {
                    "instrument": instrument,
                    "gm_program": (128 if instrument in DRUMS else GM[instrument]),
                }

        local_metadata["raw_midi"] = str(raw_midi.resolve())
        local_metadata["clean_midi"] = str(clean_target.resolve())
        midi_paths[task.name] = clean_midi

    if config.midi_cleanup:
        LOGGER.info("\n== Filtering onsets in near-silent audio ==")
        for task in tasks:
            cleaned = midi_paths[task.name]
            report = filter_quiet_onsets(
                cleaned,
                task.audio,
                cleaned,
                threshold_dbfs=config.silence_threshold_dbfs,
            )
            context.metadata["stems"][task.name]["local_cleanup"][
                "quiet_onset_filter"
            ] = report
            LOGGER.info(
                f"  {task.name:<7} removed "
                f"{report['notes_removed']} quiet onsets "
                f"(< {config.silence_threshold_dbfs:g} dBFS) "
                f"-> {local_targets[task.name]}",
            )

    if config.midi_cleanup:
        LOGGER.info("\n== Filtering excessive onset bursts ==")
        for task in tasks:
            backend, _ = split_transcriber_spec(transcriber_for_stem(config, task.name))
            if backend != "muscriptor":
                continue
            cleaned = midi_paths[task.name]
            report = filter_excessive_onset_bursts(
                cleaned,
                cleaned,
                maximum_onsets=MAXIMUM_SIMULTANEOUS_ONSETS[task.name],
            )
            context.metadata["stems"][task.name]["local_cleanup"][
                "onset_burst_filter"
            ] = report
            LOGGER.info(
                f"  {task.name:<7} removed {report['notes_removed']} notes "
                f"from {report['burst_count']} onset bursts "
                f"-> {local_targets[task.name]}",
            )

        # MuScriptor is normalized during JSONL conversion.
        LOGGER.info("\n== Normalizing local same-pitch overlaps ==")
        for task in tasks:
            backend, _ = split_transcriber_spec(transcriber_for_stem(config, task.name))
            if backend == "muscriptor":
                continue
            cleaned = midi_paths[task.name]
            report = truncate_overlapping_same_pitch_notes(cleaned, cleaned)
            context.metadata["stems"][task.name]["local_cleanup"][
                "same_pitch_overlap_cleanup"
            ] = report

        LOGGER.info("\n== Clamping local-clean MIDI to stem durations ==")
        for task in tasks:
            cleaned = midi_paths[task.name]
            duration_seconds = sf.info(task.audio).duration
            report = clamp_midi_duration(
                cleaned,
                cleaned,
                duration_seconds,
            )
            context.metadata["stems"][task.name]["local_cleanup"]["duration_clamp"] = (
                report
            )
            LOGGER.info(
                f"  {task.name:<7} removed "
                f"{report['notes_removed_after_end']} notes, clipped "
                f"{report['note_offs_clipped']} note-offs at "
                f"{duration_seconds:.6f}s "
                f"-> {local_targets[task.name]}",
            )

    for task in tasks:
        work_midi = midi_paths[task.name]
        clean_target = local_targets[task.name]
        work_path = str(work_midi.resolve())
        target_path = str(clean_target.resolve())
        for report in context.metadata["stems"][task.name]["local_cleanup"].values():
            if not isinstance(report, dict):
                continue
            for key in ("source", "output"):
                if report.get(key) == work_path:
                    report[key] = target_path
        work_midi.replace(clean_target)
        midi_paths[task.name] = clean_target
        context.metadata["stems"][task.name]["local_cleanup"]["status"] = "completed"

    LOGGER.info("\n== Analyzing tempo from MuScriptor stems ==")
    reports = collect_muscriptor_tempo_reports(
        tasks,
        context=context,
    )
    consensus = select_tempo_grid(reports)
    if consensus is None:
        LOGGER.info("  no usable tempo evidence, fallback to 120 BPM")
        context.tempo_result = None
        context.metadata["tempo"] = {
            "tempo_source": "fallback",
            "selected_bpm": 120.0,
            "status": "unavailable",
            "reports": reports,
        }
    else:
        context.tempo_result = consensus
        context.tempo_result["beat_times_seconds"] = []
        LOGGER.info(
            "  BPM "
            f"{context.tempo_result['selected_bpm']:.3f} "
            "(sources: " + ", ".join(context.tempo_result["sources_used"]) + ")",
        )
        context.metadata["tempo"] = {
            **context.tempo_result,
            "tempo_source": context.tempo_result["selected_source"],
        }

    if not config.midi_cleanup or (config.only_other or config.treat_original_as_other):
        reason = (
            "clean-midi disabled"
            if not config.midi_cleanup
            else "cross-stem cleanup unavailable in single-other mode"
        )
        for task in tasks:
            context.metadata["stems"][task.name]["global_cleanup"] = {
                "status": "skipped",
                "reason": reason,
            }
        return midi_paths

    LOGGER.info("\n== Creating global-clean MIDI ==")
    global_targets: dict[str, Path] = {}
    for task in tasks:
        clean_midi = midi_paths[task.name]
        base = clean_midi.name.removesuffix(".clean.midi")
        global_target = clean_midi.with_name(base + ".global.clean.midi")
        global_midi = (
            context.log_dir / f"{context.run_prefix}-{task.name}.global-clean.work.midi"
        )
        global_midi.unlink(missing_ok=True)
        shutil.copy2(clean_midi, global_midi)
        global_targets[task.name] = global_target
        context.metadata["stems"][task.name]["global_cleanup"] = {
            "status": "running",
            "source_clean_midi": str(clean_midi.resolve()),
            "global_clean_midi": str(global_target.resolve()),
        }
        midi_paths[task.name] = global_midi

    if config.midi_cleanup:
        LOGGER.info("\n== Running cross-stem retrigger cleanup ==")
        totals = {
            "before": 0,
            "after": 0,
            "merged": 0,
            "boundaries": 0,
        }
        for task in tasks:
            backend, _ = split_transcriber_spec(transcriber_for_stem(config, task.name))
            if backend != "muscriptor":
                continue
            cleaned = midi_paths[task.name]
            reference_midis = [
                path
                for name, path in midi_paths.items()
                if name != task.name and path.is_file()
            ]
            report = clean_retrigger_loops(
                cleaned,
                task.audio,
                cleaned,
                reference_midis=reference_midis,
                beat_times=(
                    context.tempo_result["beat_times_seconds"]
                    if context.tempo_result is not None
                    else ()
                ),
            )
            midi_paths[task.name] = cleaned
            context.metadata["stems"][task.name]["global_cleanup"][
                "retrigger_cleanup"
            ] = report
            totals["before"] += report["notes_before"]
            totals["after"] += report["notes_after"]
            totals["merged"] += report["notes_merged"]
            totals["boundaries"] += report["boundaries_removed"]
            LOGGER.info(
                f"  {task.name:<7} notes "
                f"{report['notes_before']} -> {report['notes_after']}; "
                f"merged {report['notes_merged']} notes, removed "
                f"{report['boundaries_removed']} boundaries "
                f"-> {global_targets[task.name]}",
            )
        LOGGER.info(
            "  total   notes "
            f"{totals['before']} -> {totals['after']}; "
            f"merged {totals['merged']} notes, removed "
            f"{totals['boundaries']} boundaries",
        )

    if config.midi_cleanup:
        LOGGER.info("\n== Clamping global-clean MIDI to stem durations ==")
        for task in tasks:
            global_midi = midi_paths[task.name]
            duration_seconds = sf.info(task.audio).duration
            report = clamp_midi_duration(
                global_midi,
                global_midi,
                duration_seconds,
            )
            context.metadata["stems"][task.name]["global_cleanup"]["duration_clamp"] = (
                report
            )
            LOGGER.info(
                f"  {task.name:<7} removed "
                f"{report['notes_removed_after_end']} notes, clipped "
                f"{report['note_offs_clipped']} note-offs at "
                f"{duration_seconds:.6f}s "
                f"-> {global_targets[task.name]}",
            )
    for task in tasks:
        work_midi = midi_paths[task.name]
        global_target = global_targets[task.name]
        work_path = str(work_midi.resolve())
        target_path = str(global_target.resolve())
        for report in context.metadata["stems"][task.name]["global_cleanup"].values():
            if not isinstance(report, dict):
                continue
            for key in ("source", "output"):
                if report.get(key) == work_path:
                    report[key] = target_path
        work_midi.replace(global_target)
        midi_paths[task.name] = global_target
        context.metadata["stems"][task.name]["global_cleanup"]["status"] = "completed"
    return midi_paths
