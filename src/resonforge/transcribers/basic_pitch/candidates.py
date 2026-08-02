"""Posteriorgram candidate generation for the general Basic Pitch decoder."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ._runtime import _basic_pitch_runtime

AUDIO_SAMPLE_RATE = 22050
FFT_HOP = 256
MIDI_OFFSET = 21


@dataclass
class Candidate:
    start_frame: int
    end_frame: int
    pitch: int
    confidence: float
    onset_confidence: float
    note_confidence: float
    contour_confidence: float
    source: str

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


def frames_for_ms(milliseconds: float) -> int:
    return max(1, int(round(milliseconds / 1000.0 * AUDIO_SAMPLE_RATE / FFT_HOP)))


def safe_inferred_onsets(onsets: np.ndarray, notes: np.ndarray) -> np.ndarray:
    """Use Basic Pitch's inferred-onset routine unless the input is degenerate."""
    if not np.any(notes) or float(np.max(notes) - np.min(notes)) <= 1e-9:
        return onsets.copy()
    try:
        inferred = _basic_pitch_runtime().get_infered_onsets(
            onsets.copy(), notes.copy()
        )
    except RuntimeError:
        # Degenerate inference falls back to model onsets.
        return onsets.copy()
    if not np.all(np.isfinite(inferred)):
        return onsets.copy()
    return inferred


def local_peaks(values: np.ndarray, threshold: float) -> np.ndarray:
    if values.size < 3:
        return np.flatnonzero(values >= threshold)
    peak = np.zeros(values.shape, dtype=bool)
    peak[1:-1] = (values[1:-1] >= values[:-2]) & (
        values[1:-1] > values[2:]
    )
    peak[0] = values[0] > values[1]
    peak[-1] = values[-1] >= values[-2]
    return np.flatnonzero(peak & (values >= threshold))


def contour_for_pitch(contours: np.ndarray, pitch_index: int) -> np.ndarray:
    start = max(0, pitch_index * 3)
    end = min(contours.shape[1], start + 3)
    if end <= start:
        return np.zeros(contours.shape[0], dtype=np.float32)
    return np.max(contours[:, start:end], axis=1)


def trace_note_end(
    activation: np.ndarray,
    start: int,
    threshold: float,
    tolerance_frames: int,
) -> int:
    below = 0
    index = start + 1
    while index < activation.size:
        if activation[index] < threshold:
            below += 1
            if below >= tolerance_frames:
                return max(start + 1, index - below + 1)
        else:
            below = 0
        index += 1
    return activation.size


def contiguous_regions(
    mask: np.ndarray,
    gap_tolerance: int,
) -> Iterable[tuple[int, int]]:
    active = np.flatnonzero(mask)
    if not active.size:
        return
    start = int(active[0])
    previous = int(active[0])
    for index_value in active[1:]:
        index = int(index_value)
        if index - previous > gap_tolerance + 1:
            yield start, previous + 1
            start = index
        previous = index
    yield start, previous + 1


def candidate_score(
    onset_confidence: float,
    note_confidence: float,
    contour_confidence: float,
) -> float:
    return float(
        np.clip(
            0.40 * onset_confidence
            + 0.35 * note_confidence
            + 0.25 * contour_confidence,
            0.0,
            1.0,
        )
    )


def generate_candidates(
    output: dict[str, np.ndarray],
    *,
    onset_threshold: float,
    frame_threshold: float,
    frame_only_threshold: float,
    onset_min_frames: int,
    frame_only_min_frames: int,
    tolerance_frames: int,
    frame_only_min_confidence: float,
) -> list[Candidate]:
    notes = np.asarray(output["note"], dtype=np.float32)
    onsets = safe_inferred_onsets(
        np.asarray(output["onset"], dtype=np.float32),
        notes,
    )
    contours = np.asarray(output["contour"], dtype=np.float32)
    n_frames, n_pitches = notes.shape
    covered = np.zeros(notes.shape, dtype=bool)
    candidates: list[Candidate] = []

    onset_locations: list[tuple[float, int, int]] = []
    for pitch_index in range(n_pitches):
        for frame in local_peaks(onsets[:, pitch_index], onset_threshold):
            onset_locations.append(
                (float(onsets[frame, pitch_index]), int(frame), pitch_index)
            )

    # Strong candidates claim their same-pitch frame energy first.
    for onset_value, start, pitch_index in sorted(
        onset_locations, reverse=True
    ):
        end = trace_note_end(
            notes[:, pitch_index],
            start,
            frame_threshold,
            tolerance_frames,
        )
        if end - start < onset_min_frames:
            continue
        note_conf = float(np.mean(notes[start:end, pitch_index]))
        contour_track = contour_for_pitch(contours, pitch_index)
        contour_conf = float(np.mean(contour_track[start:end]))
        score = candidate_score(onset_value, note_conf, contour_conf)
        candidates.append(
            Candidate(
                start,
                end,
                pitch_index + MIDI_OFFSET,
                score,
                onset_value,
                note_conf,
                contour_conf,
                "onset",
            )
        )
        covered[start:end, pitch_index] = True

    # Use the paper's additional-note path only as fallback.
    gap_tolerance = max(1, tolerance_frames // 2)
    for pitch_index in range(n_pitches):
        residual_mask = (
            (notes[:, pitch_index] >= frame_only_threshold)
            & ~covered[:, pitch_index]
        )
        contour_track = contour_for_pitch(contours, pitch_index)
        for start, end in contiguous_regions(residual_mask, gap_tolerance):
            if end - start < frame_only_min_frames:
                continue
            note_conf = float(np.mean(notes[start:end, pitch_index]))
            contour_conf = float(np.mean(contour_track[start:end]))
            onset_conf = float(np.max(onsets[start:end, pitch_index]))
            score = candidate_score(onset_conf, note_conf, contour_conf)
            if score < frame_only_min_confidence:
                continue
            candidates.append(
                Candidate(
                    start,
                    end,
                    pitch_index + MIDI_OFFSET,
                    score,
                    onset_conf,
                    note_conf,
                    contour_conf,
                    "frame",
                )
            )
    return candidates
