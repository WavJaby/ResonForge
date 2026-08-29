"""MuScriptor-only MIDI postprocessing over in-memory stage results."""

from __future__ import annotations

from dataclasses import dataclass, replace

from resonforge.transcribers.base import StemTask

from ..midi.jsonl import convert_events_to_midi, notes_to_midi_document
from ..midi.memory_cleanup import (
    AudioCleanupFeatures,
    clamp_notes,
    clean_retrigger_notes,
    filter_quiet_notes,
)
from ..observability.tempo_types import MuscriptorTempoReport, TempoResult
from ..transcribers.base import (
    ProcessedTranscription,
    TranscriptionResult,
)
from .muscriptor_tempo import analyze_muscriptor_tempo, select_tempo_grid
from .types import PipelineConfig

MAXIMUM_SIMULTANEOUS_ONSETS = {
    "bass": 6,
    "vocals": 5,
    "guitar": 6,
    "drums": 8,
    "piano": 10,
    "other": 12,
    "band": 16,
}


@dataclass(frozen=True)
class PostprocessBatchResult:
    stems: dict[str, ProcessedTranscription]
    tempo: TempoResult | None


def postprocess_in_memory(
    transcriptions: dict[str, TranscriptionResult],
    tasks: list[StemTask],
    *,
    config: PipelineConfig,
    audio_features: dict[str, AudioCleanupFeatures] | None = None,
    tempo_reports: dict[str, MuscriptorTempoReport] | None = None,
) -> PostprocessBatchResult:
    """Clean MuScriptor notes without materializing intermediate MIDI files."""
    task_by_stem = {task.name: task for task in tasks}
    features_by_stem = dict(audio_features or {})
    processed: dict[str, ProcessedTranscription] = {}
    for stem, transcription in transcriptions.items():
        task = task_by_stem[stem]
        conversion = convert_events_to_midi(
            transcription.events,
            cleanup=config.midi_cleanup,
            allow_empty=True,
        )
        notes = conversion.notes
        reports: dict[str, object] = {"conversion_cleanup": conversion.report}
        if config.midi_cleanup:
            features = features_by_stem.get(stem)
            if features is None and task.buffer is not None:
                from ..midi.memory_cleanup import extract_audio_cleanup_features

                features = extract_audio_cleanup_features(task.buffer)
            if features is None:
                raise RuntimeError(f"missing compact audio features for {stem}")
            quiet = filter_quiet_notes(
                notes,
                features,
                threshold_dbfs=config.silence_threshold_dbfs,
            )
            notes = quiet.notes
            reports["quiet_onset_filter"] = quiet.report
            # Onset-burst filtering temporarily disabled: it drops onsets past a
            # fixed per-stem cap regardless of whether the audio supports them.
            # Re-enable by restoring this block and the filter_onset_bursts import.
            # burst = filter_onset_bursts(
            #     notes,
            #     maximum_onsets=MAXIMUM_SIMULTANEOUS_ONSETS[stem],
            # )
            # notes = burst.notes
            # reports["onset_burst_filter"] = burst.report
            clamped = clamp_notes(notes, features.duration_seconds)
            notes = clamped.notes
            reports["duration_clamp"] = clamped.report
        clean = notes_to_midi_document(notes)
        processed[stem] = ProcessedTranscription(
            stem=stem,
            events=transcription.events,
            raw_midi=transcription.raw_midi,
            clean_midi=clean,
            selected_midi=clean,
            notes=notes,
            artifacts=transcription.artifacts,
            reports=reports,
            metrics=transcription.metrics,
        )

    selected_tempo_reports = (
        list(tempo_reports.values())
        if tempo_reports is not None
        else [
            analyze_muscriptor_tempo(task, model=config.transcriber_for_stem(task.name))
            for task in tasks
        ]
    )
    tempo = select_tempo_grid(selected_tempo_reports)
    if config.midi_cleanup and not (
        config.only_other or config.treat_original_as_other
    ):
        beat_times = tempo.get("beat_times_seconds", ()) if tempo else ()
        local_notes = {stem: result.notes for stem, result in processed.items()}
        for stem, result in tuple(processed.items()):
            features = features_by_stem.get(stem)
            if features is None and task_by_stem[stem].buffer is not None:
                from ..midi.memory_cleanup import extract_audio_cleanup_features

                features = extract_audio_cleanup_features(task_by_stem[stem].buffer)
            if features is None:
                raise RuntimeError(f"missing compact audio features for {stem}")
            references = (
                note
                for other_stem, notes in local_notes.items()
                if other_stem != stem
                for note in notes
            )
            cleaned = clean_retrigger_notes(
                result.notes,
                features,
                reference_notes=references,
                beat_times=beat_times,
            )
            selected = notes_to_midi_document(cleaned.notes)
            reports = {**result.reports, "retrigger_cleanup": cleaned.report}
            processed[stem] = replace(
                result,
                selected_midi=selected,
                notes=cleaned.notes,
                reports=reports,
            )
    return PostprocessBatchResult(processed, tempo)
