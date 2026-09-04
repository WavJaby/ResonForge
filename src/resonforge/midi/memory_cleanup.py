"""Pure in-memory cleanup for MuScriptor note intervals."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import librosa
import numpy as np

from ..audio.buffer import AudioBuffer
from .intervals import Note


def _frame_rms_dbfs(
    audio: AudioBuffer,
    *,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    samples = audio.mono(sample_rate)
    rms = librosa.feature.rms(
        y=samples,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    return (
        librosa.frames_to_time(
            np.arange(rms.size), sr=sample_rate, hop_length=hop_length
        ),
        20.0 * np.log10(np.maximum(rms, 1e-12)),
    )


def _onset_times(audio: AudioBuffer, sample_rate: int = 22_050) -> np.ndarray:
    samples = audio.mono(sample_rate)
    envelope = librosa.onset.onset_strength(
        y=samples, sr=sample_rate, hop_length=256
    )
    frames = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=sample_rate,
        hop_length=256,
        units="frames",
        backtrack=False,
    )
    return librosa.frames_to_time(frames, sr=sample_rate, hop_length=256)


def _support_ratio(
    query: Iterable[float],
    reference: np.ndarray,
    tolerance: float,
) -> float:
    """Share of query times with a reference time within `tolerance`."""
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



@dataclass(frozen=True)
class AudioCleanupFeatures:
    """Compact audio evidence retained after the decoded WAV is released."""

    duration_seconds: float
    rms_frame_times: np.ndarray
    rms_frame_dbfs: np.ndarray
    onset_times: np.ndarray


def extract_audio_cleanup_features(audio: AudioBuffer) -> AudioCleanupFeatures:
    frame_times, frame_dbfs = _frame_rms_dbfs(
        audio,
        sample_rate=22_050,
        frame_length=1024,
        hop_length=256,
    )
    return AudioCleanupFeatures(
        duration_seconds=audio.duration_seconds,
        rms_frame_times=frame_times,
        rms_frame_dbfs=frame_dbfs,
        onset_times=_onset_times(audio),
    )


@dataclass(frozen=True)
class NoteCleanupResult:
    notes: tuple[Note, ...]
    report: dict[str, object]


def filter_quiet_notes(
    notes: Iterable[Note],
    features: AudioCleanupFeatures,
    *,
    threshold_dbfs: float,
) -> NoteCleanupResult:
    source = tuple(notes)
    frame_times = features.rms_frame_times
    frame_dbfs = features.rms_frame_dbfs
    kept: list[Note] = []
    removed: list[dict[str, object]] = []
    for note in source:
        left = np.searchsorted(frame_times, max(0.0, note[2] - 0.02), side="left")
        right = np.searchsorted(frame_times, note[2] + 0.08, side="right")
        local = frame_dbfs[left:right]
        onset_dbfs = float(np.max(local)) if local.size else -240.0
        if onset_dbfs < threshold_dbfs:
            removed.append(
                {
                    "instrument": note[0],
                    "pitch": note[1],
                    "start_seconds": round(note[2], 6),
                    "end_seconds": round(note[3], 6),
                    "onset_peak_dbfs": round(onset_dbfs, 3),
                }
            )
        else:
            kept.append(note)
    return NoteCleanupResult(
        tuple(kept),
        {
            "threshold_dbfs": threshold_dbfs,
            "notes_before": len(source),
            "notes_removed": len(removed),
            "notes_after": len(kept),
            "removed": removed,
        },
    )


def filter_onset_bursts(
    notes: Iterable[Note],
    *,
    maximum_onsets: int,
    onset_window_seconds: float = 0.02,
) -> NoteCleanupResult:
    source = tuple(notes)
    ordered = sorted(enumerate(source), key=lambda item: (item[1][2], item[0]))
    clusters: list[list[tuple[int, Note]]] = []
    cluster: list[tuple[int, Note]] = []
    cluster_start = 0.0
    for indexed in ordered:
        if not cluster or indexed[1][2] - cluster_start <= onset_window_seconds:
            if not cluster:
                cluster_start = indexed[1][2]
            cluster.append(indexed)
        else:
            clusters.append(cluster)
            cluster = [indexed]
            cluster_start = indexed[1][2]
    if cluster:
        clusters.append(cluster)
    removed_indexes: set[int] = set()
    bursts: list[dict[str, object]] = []
    for group in clusters:
        if len(group) <= maximum_onsets:
            continue
        removed = group[maximum_onsets:]
        removed_indexes.update(index for index, _note in removed)
        bursts.append(
            {
                "start_seconds": round(group[0][1][2], 6),
                "onsets_before": len(group),
                "onsets_after": maximum_onsets,
                "removed": [
                    {"instrument": note[0], "pitch": note[1]}
                    for _index, note in removed
                ],
            }
        )
    kept = tuple(note for index, note in enumerate(source) if index not in removed_indexes)
    return NoteCleanupResult(
        kept,
        {
            "maximum_onsets": maximum_onsets,
            "onset_window_ms": onset_window_seconds * 1000.0,
            "notes_before": len(source),
            "notes_removed": len(removed_indexes),
            "notes_after": len(kept),
            "burst_count": len(bursts),
            "bursts": bursts,
        },
    )


def clamp_notes(notes: Iterable[Note], duration_seconds: float) -> NoteCleanupResult:
    source = tuple(notes)
    kept: list[Note] = []
    removed = clipped = 0
    for instrument, pitch, start, end in source:
        if start >= duration_seconds:
            removed += 1
            continue
        if end > duration_seconds:
            end = duration_seconds
            clipped += 1
        kept.append((instrument, pitch, start, end))
    return NoteCleanupResult(
        tuple(kept),
        {
            "duration_seconds": duration_seconds,
            "notes_before": len(source),
            "notes_removed_after_end": removed,
            "note_offs_clipped": clipped,
        },
    )


def clean_retrigger_notes(
    notes: Iterable[Note],
    features: AudioCleanupFeatures,
    *,
    reference_notes: Iterable[Note] = (),
    beat_times: Iterable[float] = (),
) -> NoteCleanupResult:
    source = tuple(notes)
    grouped: dict[tuple[str, int], list[tuple[int, Note]]] = {}
    for index, note in enumerate(source):
        grouped.setdefault((note[0], note[1]), []).append((index, note))
    audio_onsets = features.onset_times
    cross_onsets = np.asarray(sorted(note[2] for note in reference_notes), dtype=float)
    beat_grid = _beat_grid(beat_times)
    remove_indexes: set[int] = set()
    replacements: dict[int, Note] = {}
    decisions: list[dict[str, object]] = []
    for same_pitch in grouped.values():
        same_pitch.sort(key=lambda item: item[1][2])
        runs: list[list[tuple[int, Note]]] = []
        run: list[tuple[int, Note]] = []
        for indexed in same_pitch:
            if not run:
                run = [indexed]
                continue
            interval = indexed[1][2] - run[-1][1][2]
            gap = indexed[1][2] - run[-1][1][3]
            if 0.06 <= interval <= 0.30 and -0.02 <= gap <= 0.04:
                run.append(indexed)
            else:
                if len(run) >= 12:
                    runs.append(run)
                run = [indexed]
        if len(run) >= 12:
            runs.append(run)
        for candidate in runs:
            intervals = np.diff([item[1][2] for item in candidate])
            median_interval = float(np.median(intervals))
            interval_cv = float(np.std(intervals) / max(median_interval, 1e-9))
            boundaries = [item[1][2] for item in candidate[1:]]
            audio_support = _support_ratio(boundaries, audio_onsets, 0.05)
            cross_support = _support_ratio(boundaries, cross_onsets, 0.05)
            beat_support = _support_ratio(boundaries, beat_grid, 0.05)
            merge = (
                interval_cv <= 0.08
                and audio_support <= 0.25
                and cross_support <= 0.30
                and (beat_support <= 0.60 or (audio_support == 0 and cross_support == 0))
            )
            if merge:
                first_index, first = candidate[0]
                replacements[first_index] = (first[0], first[1], first[2], candidate[-1][1][3])
                remove_indexes.update(index for index, _note in candidate[1:])
            decisions.append(
                {
                    "instrument": candidate[0][1][0],
                    "pitch": candidate[0][1][1],
                    "start_seconds": round(candidate[0][1][2], 6),
                    "end_seconds": round(candidate[-1][1][3], 6),
                    "note_count": len(candidate),
                    "interval_cv": round(interval_cv, 5),
                    "audio_onset_support": round(audio_support, 4),
                    "cross_track_support": round(cross_support, 4),
                    "beat_grid_support": round(beat_support, 4),
                    "action": "merge_retriggers" if merge else "keep",
                }
            )
    cleaned = tuple(
        replacements.get(index, note)
        for index, note in enumerate(source)
        if index not in remove_indexes
    )
    return NoteCleanupResult(
        cleaned,
        {
            "notes_before": len(source),
            "notes_after": len(cleaned),
            "notes_merged": len(remove_indexes),
            "boundaries_removed": len(remove_indexes),
            "candidate_count": len(decisions),
            "decisions": decisions,
        },
    )
