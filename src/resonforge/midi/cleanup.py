"""Conservative cleanup for unsupported, machine-like MIDI retrigger loops."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import mido
import numpy as np

from ..audio_loading import load_mono_audio


@dataclass
class MidiNote:
    channel: int
    pitch: int
    start: float
    end: float
    on_index: int
    off_index: int


@dataclass
class CleanupDecision:
    channel: int
    pitch: int
    start_seconds: float
    end_seconds: float
    note_count: int
    median_interval_ms: float
    interval_cv: float
    audio_onset_support: float
    cross_track_support: float
    beat_grid_support: float
    action: str


def _timed_messages(path: Path) -> tuple[mido.MidiFile, list[tuple[float, mido.Message]]]:
    midi = mido.MidiFile(path)
    tempo = 500_000
    elapsed = 0.0
    timed: list[tuple[float, mido.Message]] = []
    for message in mido.merge_tracks(midi.tracks):
        elapsed += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        timed.append((elapsed, message))
        if message.type == "set_tempo":
            tempo = message.tempo
    return midi, timed


def read_midi_notes(path: str | Path) -> tuple[list[MidiNote], list[tuple[float, mido.Message]]]:
    _, timed = _timed_messages(Path(path))
    active: dict[tuple[int, int], list[tuple[float, int]]] = {}
    notes: list[MidiNote] = []
    for index, (seconds, message) in enumerate(timed):
        if not hasattr(message, "channel"):
            continue
        key = (message.channel, getattr(message, "note", -1))
        is_on = message.type == "note_on" and message.velocity > 0
        is_off = message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        )
        if is_on:
            active.setdefault(key, []).append((seconds, index))
        elif is_off and active.get(key):
            start, on_index = active[key].pop(0)
            if seconds > start:
                notes.append(
                    MidiNote(
                        message.channel,
                        message.note,
                        start,
                        seconds,
                        on_index,
                        index,
                    )
                )
    return notes, timed


def truncate_overlapping_same_pitch_notes(
    midi_path: str | Path,
    output_path: str | Path,
) -> dict:
    """End an active channel/pitch note when the same pitch retriggers."""
    source = Path(midi_path)
    output = Path(output_path)
    source_midi, timed = _timed_messages(source)
    notes, _ = read_midi_notes(source)
    grouped: dict[tuple[int, int], list[MidiNote]] = {}
    for note in notes:
        grouped.setdefault((note.channel, note.pitch), []).append(note)

    end_overrides: dict[int, float] = {}
    truncated = 0
    for same_pitch in grouped.values():
        same_pitch.sort(key=lambda note: (note.start, note.end))
        for current, following in zip(same_pitch, same_pitch[1:], strict=False):
            if following.start < current.end:
                end_overrides[current.off_index] = following.start
                truncated += 1

    # Rebuild: shortened note-off must sort before its retriggering note-on.
    result = mido.MidiFile(ticks_per_beat=source_midi.ticks_per_beat)
    track = mido.MidiTrack()
    result.tracks.append(track)
    tempo = next(
        (
            message.tempo
            for _, message in timed
            if message.type == "set_tempo"
        ),
        500_000,
    )
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    for _, message in timed:
        if message.type in {"program_change", "track_name"}:
            track.append(message.copy(time=0))

    events: list[tuple[float, int, mido.Message]] = []
    for note in notes:
        velocity = int(getattr(timed[note.on_index][1], "velocity", 90))
        end = end_overrides.get(note.off_index, note.end)
        if end <= note.start:
            end = note.start + 0.001
        events.append(
            (
                note.start,
                1,
                mido.Message(
                    "note_on",
                    channel=note.channel,
                    note=note.pitch,
                    velocity=velocity,
                ),
            )
        )
        events.append(
            (
                end,
                0,
                mido.Message(
                    "note_off",
                    channel=note.channel,
                    note=note.pitch,
                    velocity=0,
                ),
            )
        )

    previous_tick = 0
    for seconds, _, message in sorted(events, key=lambda item: (item[0], item[1])):
        tick = round(
            seconds / (tempo / 1_000_000) * result.ticks_per_beat
        )
        track.append(message.copy(time=max(0, tick - previous_tick)))
        previous_tick = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    return {
        "notes_examined": len(notes),
        "overlaps_truncated": truncated,
    }


def _onset_times(audio_path: Path, sample_rate: int = 22_050) -> np.ndarray:
    audio, sr = load_mono_audio(audio_path, sample_rate=sample_rate)
    envelope = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=256)
    frames = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=sr,
        hop_length=256,
        units="frames",
        backtrack=False,
    )
    return librosa.frames_to_time(frames, sr=sr, hop_length=256)


def _support_ratio(
    query: Iterable[float],
    reference: np.ndarray,
    tolerance: float,
) -> float:
    query_array = np.asarray(list(query), dtype=float)
    if query_array.size == 0 or reference.size == 0:
        return 0.0
    indices = np.searchsorted(reference, query_array)
    supported = 0
    for query_time, index in zip(query_array, indices, strict=True):
        distances = []
        if index < reference.size:
            distances.append(abs(float(reference[index]) - query_time))
        if index:
            distances.append(abs(float(reference[index - 1]) - query_time))
        supported += bool(distances and min(distances) <= tolerance)
    return supported / query_array.size


def _beat_grid(beat_times: Iterable[float]) -> np.ndarray:
    """Expand quarter-note beats to common binary and triplet subdivisions."""
    beats = np.asarray(list(beat_times), dtype=float)
    if beats.size < 2:
        return beats
    points = list(beats)
    for left, right in zip(beats, beats[1:], strict=False):
        interval = right - left
        if interval <= 0:
            continue
        for denominator in (2, 3, 4):
            points.extend(
                left + interval * numerator / denominator
                for numerator in range(1, denominator)
            )
    return np.asarray(sorted(set(points)), dtype=float)


def _candidate_runs(notes: list[MidiNote]) -> list[list[MidiNote]]:
    grouped: dict[tuple[int, int], list[MidiNote]] = {}
    for note in notes:
        grouped.setdefault((note.channel, note.pitch), []).append(note)

    candidates: list[list[MidiNote]] = []
    for same_pitch in grouped.values():
        same_pitch.sort(key=lambda note: note.start)
        run: list[MidiNote] = []
        for note in same_pitch:
            if not run:
                run = [note]
                continue
            interval = note.start - run[-1].start
            gap = note.start - run[-1].end
            if 0.06 <= interval <= 0.30 and -0.02 <= gap <= 0.04:
                run.append(note)
            else:
                if len(run) >= 12:
                    candidates.append(run)
                run = [note]
        if len(run) >= 12:
            candidates.append(run)
    return candidates


def _save_without_boundaries(
    source: Path,
    output: Path,
    remove_indices: set[int],
) -> None:
    source_midi, timed = _timed_messages(source)
    _save_timed_messages(
        source_midi,
        timed,
        output,
        remove_indices=remove_indices,
    )


def _save_timed_messages(
    source_midi: mido.MidiFile,
    timed: list[tuple[float, mido.Message]],
    output: Path,
    *,
    remove_indices: set[int] | None = None,
    time_overrides: dict[int, float] | None = None,
) -> None:
    remove_indices = remove_indices or set()
    time_overrides = time_overrides or {}
    result = mido.MidiFile(ticks_per_beat=source_midi.ticks_per_beat)
    track = mido.MidiTrack()
    result.tracks.append(track)
    tempo = 500_000
    previous_seconds = 0.0
    for index, (seconds, message) in enumerate(timed):
        if index in remove_indices or message.type == "end_of_track":
            continue
        seconds = time_overrides.get(index, seconds)
        delta_tick = round(
            (seconds - previous_seconds)
            / (tempo / 1_000_000)
            * result.ticks_per_beat
        )
        copied = message.copy(time=max(0, delta_tick))
        track.append(copied)
        previous_seconds = seconds
        if message.type == "set_tempo":
            tempo = message.tempo
    track.append(mido.MetaMessage("end_of_track", time=0))
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)


def clamp_midi_duration(
    midi_path: str | Path,
    output_path: str | Path,
    duration_seconds: float,
) -> dict:
    """Remove out-of-range onsets and clip note-offs to an audio duration."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    source = Path(midi_path)
    output = Path(output_path)
    in_place = source.resolve() == output.resolve()
    temporary_output: Path | None = None
    write_output = output
    if in_place:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.stem}.",
            suffix=".clamp-tmp.mid",
            dir=output.parent,
        )
        os.close(descriptor)
        temporary_output = Path(temporary_name)
        write_output = temporary_output
    source_midi, timed = _timed_messages(source)
    notes, _ = read_midi_notes(source)
    remove_indices: set[int] = {
        index
        for index, (seconds, message) in enumerate(timed)
        if seconds > duration_seconds and message.type != "end_of_track"
    }
    time_overrides: dict[int, float] = {}
    removed_notes = 0
    clipped_notes = 0
    for note in notes:
        if note.start >= duration_seconds:
            remove_indices.update((note.on_index, note.off_index))
            removed_notes += 1
        elif note.end > duration_seconds:
            remove_indices.discard(note.off_index)
            time_overrides[note.off_index] = duration_seconds
            clipped_notes += 1

    try:
        _save_timed_messages(
            source_midi,
            timed,
            write_output,
            remove_indices=remove_indices,
            time_overrides=time_overrides,
        )
        if in_place:
            os.replace(write_output, output)
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
    return {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "duration_seconds": float(duration_seconds),
        "notes_before": len(notes),
        "notes_removed_after_end": removed_notes,
        "note_offs_clipped": clipped_notes,
    }


def filter_excessive_onset_bursts(
    midi_path: str | Path,
    output_path: str | Path,
    *,
    maximum_onsets: int,
    onset_window_seconds: float = 0.02,
) -> dict:
    """Cap implausibly dense near-simultaneous onsets by removing low velocity."""
    if maximum_onsets < 1:
        raise ValueError("maximum_onsets must be at least 1")
    if onset_window_seconds < 0:
        raise ValueError("onset_window_seconds must be non-negative")

    source = Path(midi_path)
    output = Path(output_path)
    notes, timed = read_midi_notes(source)
    ordered = sorted(notes, key=lambda note: (note.start, note.on_index))
    clusters: list[list[MidiNote]] = []
    cluster: list[MidiNote] = []
    cluster_start = 0.0
    for note in ordered:
        if not cluster or note.start - cluster_start <= onset_window_seconds:
            if not cluster:
                cluster_start = note.start
            cluster.append(note)
        else:
            clusters.append(cluster)
            cluster = [note]
            cluster_start = note.start
    if cluster:
        clusters.append(cluster)

    remove_indices: set[int] = set()
    bursts: list[dict[str, object]] = []
    for notes_at_onset in clusters:
        if len(notes_at_onset) <= maximum_onsets:
            continue
        ranked = sorted(
            notes_at_onset,
            key=lambda note: (
                -int(getattr(timed[note.on_index][1], "velocity", 0)),
                note.on_index,
            ),
        )
        removed = ranked[maximum_onsets:]
        for note in removed:
            remove_indices.update((note.on_index, note.off_index))
        bursts.append(
            {
                "start_seconds": round(notes_at_onset[0].start, 6),
                "onsets_before": len(notes_at_onset),
                "onsets_after": maximum_onsets,
                "removed": [
                    {
                        "channel": note.channel,
                        "pitch": note.pitch,
                        "velocity": int(
                            getattr(timed[note.on_index][1], "velocity", 0)
                        ),
                    }
                    for note in removed
                ],
            }
        )

    _save_without_boundaries(source, output, remove_indices)
    notes_removed = len(remove_indices) // 2
    return {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "maximum_onsets": maximum_onsets,
        "onset_window_ms": onset_window_seconds * 1000.0,
        "notes_before": len(notes),
        "notes_removed": notes_removed,
        "notes_after": len(notes) - notes_removed,
        "burst_count": len(bursts),
        "bursts": bursts,
    }


def clean_retrigger_loops(
    midi_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    *,
    reference_midis: Iterable[str | Path] = (),
    beat_times: Iterable[float] = (),
    onset_tolerance: float = 0.05,
    maximum_audio_support: float = 0.25,
    maximum_cross_support: float = 0.30,
    maximum_interval_cv: float = 0.08,
) -> dict:
    """Merge only long retrigger loops unsupported by audio and other tracks."""

    source = Path(midi_path)
    output = Path(output_path)
    notes, _ = read_midi_notes(source)
    audio_onsets = _onset_times(Path(audio_path))
    cross_onsets = np.asarray(
        sorted(
            note.start
            for reference in reference_midis
            for note in read_midi_notes(Path(reference))[0]
        ),
        dtype=float,
    )
    beat_grid = _beat_grid(beat_times)
    remove_indices: set[int] = set()
    decisions: list[CleanupDecision] = []

    for run in _candidate_runs(notes):
        intervals = np.diff([note.start for note in run])
        median_interval = float(np.median(intervals))
        interval_cv = float(np.std(intervals) / max(median_interval, 1e-9))
        boundaries = [note.start for note in run[1:]]
        audio_support = _support_ratio(boundaries, audio_onsets, onset_tolerance)
        cross_support = _support_ratio(boundaries, cross_onsets, onset_tolerance)
        beat_support = _support_ratio(boundaries, beat_grid, onset_tolerance)
        should_merge = (
            interval_cv <= maximum_interval_cv
            and audio_support <= maximum_audio_support
            and cross_support <= maximum_cross_support
            # Strong beat support requires zero acoustic support.
            and (
                beat_support <= 0.60
                or (audio_support == 0.0 and cross_support == 0.0)
            )
        )
        if should_merge:
            for previous, following in zip(run, run[1:], strict=False):
                remove_indices.add(previous.off_index)
                remove_indices.add(following.on_index)
        decisions.append(
            CleanupDecision(
                channel=run[0].channel,
                pitch=run[0].pitch,
                start_seconds=round(run[0].start, 6),
                end_seconds=round(run[-1].end, 6),
                note_count=len(run),
                median_interval_ms=round(median_interval * 1000.0, 3),
                interval_cv=round(interval_cv, 5),
                audio_onset_support=round(audio_support, 4),
                cross_track_support=round(cross_support, 4),
                beat_grid_support=round(beat_support, 4),
                action="merge_retriggers" if should_merge else "keep",
            )
        )

    _save_without_boundaries(source, output, remove_indices)
    return {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "notes_before": len(notes),
        "notes_after": len(notes) - len(remove_indices) // 2,
        "notes_merged": len(remove_indices) // 2,
        "boundaries_removed": len(remove_indices) // 2,
        "candidate_count": len(decisions),
        "decisions": [asdict(decision) for decision in decisions],
    }
