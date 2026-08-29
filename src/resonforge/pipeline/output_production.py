"""Memory-first final MIDI merge and optional output publication."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from resonforge.transcribers.base import StemTask

from ..midi.merge import merge_midi_documents
from ..midi.mixdown import mixdown_midis
from .memory_postprocessing import PostprocessBatchResult
from .muscriptor_tempo import merged_bpm
from .types import InitializedRun, PipelineOutputs

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
    batch: PostprocessBatchResult,
    tasks: list[StemTask],
    *,
    run: InitializedRun,
) -> PipelineOutputs:
    """Merge MIDI in memory and publish only when the output sink is enabled."""
    documents = [batch.stems[task.name].selected_midi for task in tasks]
    output_bpm = merged_bpm(batch.tempo)
    meter = batch.tempo.get("beats_per_bar") if batch.tempo else None
    phase = batch.tempo.get("bar_phase_seconds") if batch.tempo else None
    kwargs = {
        "source_gains_db": [0.0] * len(documents),
        "output_bpm": output_bpm,
        "track_names": [task.name for task in tasks],
        "output_beats_per_bar": meter,
        "bar_phase_seconds": phase,
    }
    try:
        merged, mode = merge_midi_documents(documents, **kwargs)
    except ValueError:
        merged, mode = merge_midi_documents(documents, use_ports=True, **kwargs)

    midi_path = None
    mp3_path = None
    if run.config.publish_files:
        midi_path = merged.write(run.output_dir / "final.mid")
        run.metadata["outputs"]["final_midi"] = str(midi_path)
        if run.config.mp3_out:
            mp3_path = run.output_dir / "final.mp3"
            with tempfile.TemporaryDirectory(prefix="resonforge-mix-") as directory:
                sources = []
                for task, document in zip(tasks, documents, strict=True):
                    source = Path(directory) / f"{task.name}.mid"
                    document.write(source)
                    sources.append(source)
                mixdown_midis(
                    sources,
                    out=mp3_path,
                    mp3_bitrate=run.config.mp3_bitrate,
                    jobs=run.config.mix_jobs,
                    track_gains_db=",".join(
                        str(STEM_GAINS_DB[task.name]) for task in tasks
                    ),
                    normalize_dbfs=-1.0,
                    midi_normalize_velocity=110,
                    midi_out=None,
                )
            run.metadata["outputs"]["final_mp3"] = str(mp3_path)
    run.metadata["outputs"].update(
        {
            "final_midi_bpm": output_bpm,
            "final_midi_beats_per_bar": meter,
            "final_midi_bar_phase_seconds": phase,
            "midi_mode": mode,
            "published": run.config.publish_files,
        }
    )
    return PipelineOutputs(merged, midi_path, mp3_path)
