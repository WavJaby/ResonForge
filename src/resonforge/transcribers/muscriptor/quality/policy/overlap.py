"""Adaptive overlap configuration and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from muscriptor.tokenizer.notes import NoteEvent


@dataclass(frozen=True)
class OverlapWindow:
    """Resolved replay and verification spans for one chunk boundary."""

    replay_seconds: float = 0.6
    verification_seconds: float = 0.4

    def __post_init__(self) -> None:
        if self.replay_seconds <= 0 or self.verification_seconds <= 0:
            raise ValueError(
                "replay_seconds and verification_seconds must be positive"
            )
        if self.overlap_seconds >= 5:
            raise ValueError("replay plus verification must be less than 5 seconds")

    @property
    def overlap_seconds(self) -> float:
        return self.replay_seconds + self.verification_seconds

    @property
    def stride_seconds(self) -> float:
        return 5.0 - self.overlap_seconds


DEFAULT_OVERLAP_WINDOW = OverlapWindow()


@dataclass(frozen=True)
class OverlapMatch:
    reference_events: int
    candidate_events: int
    matched_events: int
    precision: float | None
    recall: float | None
    f1: float | None
    informative: bool
    score: float | None = None
    onset_score: float | None = None
    offset_score: float | None = None
    open_start_score: float | None = None
    open_end_score: float | None = None
    comparable_evidence: int = 0
    reliability: str = "none"


@dataclass(frozen=True)
class OverlapDiagnosticsEvent:
    """Diagnostics-only agreement for one generated verification suffix."""

    chunk_index: int
    seek_time: float
    condition_origin: float
    overlap_seconds: float
    replay_seconds: float
    verification_seconds: float
    match: OverlapMatch


def match_note_events(
    reference: list[NoteEvent],
    candidate: list[NoteEvent],
    *,
    tolerance_seconds: float = 0.05,
    soft_tolerance_seconds: float = 0.10,
    reference_open_start: set[tuple[int, int]] | None = None,
    candidate_open_start: set[tuple[int, int]] | None = None,
    reference_open_end: set[tuple[int, int]] | None = None,
    candidate_open_end: set[tuple[int, int]] | None = None,
) -> OverlapMatch:
    """Score event timing and sustained-note state in one verification window."""
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative")
    if soft_tolerance_seconds <= 0:
        raise ValueError("soft_tolerance_seconds must be positive")

    unmatched = set(range(len(candidate)))
    matched = 0
    for expected in reference:
        choices = [
            index
            for index in unmatched
            if _same_event(expected, candidate[index])
            and abs(expected.time - candidate[index].time) <= tolerance_seconds
        ]
        if not choices:
            continue
        best = min(
            choices, key=lambda index: abs(expected.time - candidate[index].time)
        )
        unmatched.remove(best)
        matched += 1

    reference_count = len(reference)
    candidate_count = len(candidate)
    informative = bool(reference_count or candidate_count)
    precision = matched / candidate_count if candidate_count else 0.0
    recall = matched / reference_count if reference_count else 0.0
    f1 = (
        None
        if not informative
        else (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
    )
    if not informative:
        precision = None
        recall = None

    reference_onsets = [event for event in reference if _is_onset(event)]
    candidate_onsets = [event for event in candidate if _is_onset(event)]
    reference_offsets = [event for event in reference if not _is_onset(event)]
    candidate_offsets = [event for event in candidate if not _is_onset(event)]
    onset_score, onset_evidence = _soft_event_f1(
        reference_onsets,
        candidate_onsets,
        soft_tolerance_seconds,
    )
    offset_score, offset_evidence = _soft_event_f1(
        reference_offsets,
        candidate_offsets,
        soft_tolerance_seconds,
    )
    open_start_score, open_start_evidence = _set_f1(
        reference_open_start,
        candidate_open_start,
    )
    open_end_score, open_end_evidence = _set_f1(
        reference_open_end,
        candidate_open_end,
    )
    components = (
        (onset_score, 0.45),
        (offset_score, 0.20),
        (open_start_score, 0.15),
        (open_end_score, 0.20),
    )
    available = [(value, weight) for value, weight in components if value is not None]
    score = (
        sum(value * weight for value, weight in available)
        / sum(weight for _, weight in available)
        if available
        else None
    )
    evidence = (
        onset_evidence
        + offset_evidence
        + open_start_evidence
        + open_end_evidence
    )
    reliability = "high" if evidence >= 8 else "medium" if evidence >= 4 else "low"
    if evidence == 0:
        reliability = "none"
    return OverlapMatch(
        reference_events=reference_count,
        candidate_events=candidate_count,
        matched_events=matched,
        precision=precision,
        recall=recall,
        f1=f1,
        informative=score is not None,
        score=score,
        onset_score=onset_score,
        offset_score=offset_score,
        open_start_score=open_start_score,
        open_end_score=open_end_score,
        comparable_evidence=evidence,
        reliability=reliability,
    )


def _same_event(left: NoteEvent, right: NoteEvent) -> bool:
    return (
        left.is_drum == right.is_drum
        and left.program == right.program
        and left.pitch == right.pitch
        and left.velocity == right.velocity
    )


def _is_onset(event: NoteEvent) -> bool:
    return event.is_drum or event.velocity > 0


def _soft_event_f1(
    reference: list[NoteEvent],
    candidate: list[NoteEvent],
    tolerance_seconds: float,
) -> tuple[float | None, int]:
    evidence = max(len(reference), len(candidate))
    if evidence == 0:
        return None, 0
    unmatched = set(range(len(candidate)))
    credit = 0.0
    for expected in reference:
        choices = [
            index
            for index in unmatched
            if _same_event_identity(expected, candidate[index])
            and abs(expected.time - candidate[index].time) <= tolerance_seconds
        ]
        if not choices:
            continue
        best = min(
            choices,
            key=lambda index: abs(expected.time - candidate[index].time),
        )
        unmatched.remove(best)
        distance = abs(expected.time - candidate[best].time)
        credit += 1.0 - distance / tolerance_seconds
    precision = credit / len(candidate) if candidate else 0.0
    recall = credit / len(reference) if reference else 0.0
    if precision + recall == 0:
        return 0.0, evidence
    return 2 * precision * recall / (precision + recall), evidence


def _same_event_identity(left: NoteEvent, right: NoteEvent) -> bool:
    return (
        left.is_drum == right.is_drum
        and left.program == right.program
        and left.pitch == right.pitch
        and _is_onset(left) == _is_onset(right)
    )


def _set_f1(
    reference: set[tuple[int, int]] | None,
    candidate: set[tuple[int, int]] | None,
) -> tuple[float | None, int]:
    if reference is None or candidate is None:
        return None, 0
    evidence = max(len(reference), len(candidate))
    if evidence == 0:
        return None, 0
    matched = len(reference & candidate)
    precision = matched / len(candidate) if candidate else 0.0
    recall = matched / len(reference) if reference else 0.0
    if precision + recall == 0:
        return 0.0, evidence
    return 2 * precision * recall / (precision + recall), evidence
