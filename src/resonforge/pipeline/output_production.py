"""Final MIDI merge and optional MP3 mixdown phase."""

from __future__ import annotations

import logging
from pathlib import Path

from ..midi.mixdown import merge_midis, mixdown_midis
from .audio_staging import require_file
from .muscriptor_tempo import merged_bpm
from .types import PipelineContext, PipelineOutputs, StemTask

LOGGER = logging.getLogger(__name__)

STEM_GAINS_DB = {
    "bass": -2.0,
    "drums": -2.0,
    "guitar": -5.0,
    "piano": -5.0,
    "band": -3.5,
    "other": -7.0,
    "vocals": -4.0,
}


def produce_outputs(
    midi_paths: dict[str, Path],
    tasks: list[StemTask],
    *,
    context: PipelineContext,
) -> PipelineOutputs:
    """Merge selected stems and optionally render the MP3 mixdown."""
    config = context.config
    if context.output_dir is None:
        raise RuntimeError("pipeline context has not been initialized")

    midi_files = [midi_paths[task.name] for task in tasks]
    for midi_file in midi_files:
        require_file(midi_file, "MIDI for mixdown")
    merged = context.output_dir / f"{context.run_prefix}-final.mid"
    output_bpm = merged_bpm(context.tempo_result)
    LOGGER.info("\n== Merging final MIDI (velocities unchanged) ==")
    output_beats_per_bar = (
        context.tempo_result.get("beats_per_bar")
        if context.tempo_result is not None
        else None
    )
    bar_phase_seconds = (
        context.tempo_result.get("bar_phase_seconds")
        if context.tempo_result is not None
        else None
    )
    try:
        midi_mode = merge_midis(
            midi_files,
            merged,
            [0.0] * len(midi_files),
            normalize_velocity=None,
            output_bpm=output_bpm,
            track_names=[task.name for task in tasks],
            output_beats_per_bar=output_beats_per_bar,
            bar_phase_seconds=bar_phase_seconds,
        )
    except ValueError:
        midi_mode = merge_midis(
            midi_files,
            merged,
            [0.0] * len(midi_files),
            normalize_velocity=None,
            use_ports=True,
            output_bpm=output_bpm,
            track_names=[task.name for task in tasks],
            output_beats_per_bar=output_beats_per_bar,
            bar_phase_seconds=bar_phase_seconds,
        )
    LOGGER.info(
        f"  merged MIDI -> {merged} ({midi_mode}, {output_bpm:.3f} BPM)",
    )
    context.metadata["outputs"]["final_midi"] = str(merged)
    context.metadata["outputs"]["final_midi_bpm"] = output_bpm
    context.metadata["outputs"]["final_midi_beats_per_bar"] = output_beats_per_bar
    context.metadata["outputs"]["final_midi_bar_phase_seconds"] = bar_phase_seconds

    mix_path = None
    if config.mp3_out:
        mix_path = context.output_dir / f"{context.run_prefix}-final.mp3"
        LOGGER.info("\n== Rendering MIDI mixdown to MP3 ==")
        mixdown_midis(
            midi_files,
            out=mix_path,
            mp3_bitrate=config.mp3_bitrate,
            jobs=config.mix_jobs,
            track_gains_db=",".join(
                str(STEM_GAINS_DB[task.name]) for task in tasks
            ),
            normalize_dbfs=-1.0,
            midi_normalize_velocity=110,
            midi_out=None,
        )
        context.metadata["outputs"]["final_mp3"] = str(mix_path)
    return PipelineOutputs(midi=merged, mp3=mix_path)
