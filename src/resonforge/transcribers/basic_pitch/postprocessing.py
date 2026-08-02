"""Pure candidate postprocessing for the general Basic Pitch decoder."""

from __future__ import annotations

from collections.abc import Sequence

from ._runtime import _basic_pitch_runtime
from .candidates import AUDIO_SAMPLE_RATE, FFT_HOP, Candidate


def merge_candidate_values(left: Candidate, right: Candidate) -> Candidate:
    left_duration = max(1, left.duration_frames)
    right_duration = max(1, right.duration_frames)
    total = left_duration + right_duration
    return Candidate(
        start_frame=min(left.start_frame, right.start_frame),
        end_frame=max(left.end_frame, right.end_frame),
        pitch=left.pitch,
        confidence=max(left.confidence, right.confidence),
        onset_confidence=max(left.onset_confidence, right.onset_confidence),
        note_confidence=(
            left.note_confidence * left_duration
            + right.note_confidence * right_duration
        )
        / total,
        contour_confidence=(
            left.contour_confidence * left_duration
            + right.contour_confidence * right_duration
        )
        / total,
        source="onset" if "onset" in (left.source, right.source) else "frame",
    )


def merge_same_pitch(
    candidates: Sequence[Candidate],
    *,
    merge_gap_frames: int,
    onset_threshold: float,
    onset_min_frames: int,
) -> list[Candidate]:
    result: list[Candidate] = []
    for pitch in sorted({candidate.pitch for candidate in candidates}):
        pitch_candidates = sorted(
            (candidate for candidate in candidates if candidate.pitch == pitch),
            key=lambda item: (item.start_frame, -item.confidence),
        )
        merged: list[Candidate] = []
        for candidate in pitch_candidates:
            if not merged:
                merged.append(candidate)
                continue
            previous = merged[-1]
            gap = candidate.start_frame - previous.end_frame
            related = gap <= merge_gap_frames
            if not related:
                merged.append(candidate)
                continue

            is_rearticulation = (
                candidate.source == "onset"
                and candidate.onset_confidence >= onset_threshold
                and candidate.start_frame - previous.start_frame
                >= onset_min_frames
            )
            if is_rearticulation:
                if candidate.start_frame < previous.end_frame:
                    previous.end_frame = candidate.start_frame
                if previous.duration_frames >= onset_min_frames:
                    merged[-1] = previous
                else:
                    merged.pop()
                merged.append(candidate)
            else:
                merged[-1] = merge_candidate_values(previous, candidate)
        result.extend(merged)
    return sorted(result, key=lambda item: (item.start_frame, item.pitch))


def overlap_ratio(inner: Candidate, outer: Candidate) -> float:
    overlap = max(
        0,
        min(inner.end_frame, outer.end_frame)
        - max(inner.start_frame, outer.start_frame),
    )
    return overlap / max(1, inner.duration_frames)


def suppress_weak_harmonics(
    candidates: Sequence[Candidate],
    *,
    score_ratio: float,
    onset_threshold: float,
) -> tuple[list[Candidate], list[Candidate]]:
    rejected: list[Candidate] = []
    kept: list[Candidate] = []
    harmonic_intervals = {12, 19, 24}

    for candidate in sorted(candidates, key=lambda item: item.pitch):
        if candidate.source == "onset" and (
            candidate.onset_confidence >= onset_threshold * 0.8
        ):
            kept.append(candidate)
            continue
        suppress = False
        for lower in candidates:
            if candidate.pitch - lower.pitch not in harmonic_intervals:
                continue
            if overlap_ratio(candidate, lower) < 0.80:
                continue
            if candidate.confidence >= lower.confidence * score_ratio:
                continue
            suppress = True
            break
        if suppress:
            rejected.append(candidate)
        else:
            kept.append(candidate)
    return sorted(kept, key=lambda item: (item.start_frame, item.pitch)), rejected


def candidates_to_seconds(
    candidates: Sequence[Candidate],
    n_frames: int,
) -> list[tuple[float, float, Candidate]]:
    times = _basic_pitch_runtime().model_frames_to_time(n_frames + 1)
    converted = []
    for candidate in candidates:
        start = max(0.0, float(times[min(candidate.start_frame, n_frames)]))
        end = max(
            start + FFT_HOP / AUDIO_SAMPLE_RATE,
            float(times[min(candidate.end_frame, n_frames)]),
        )
        converted.append((start, end, candidate))
    return converted
