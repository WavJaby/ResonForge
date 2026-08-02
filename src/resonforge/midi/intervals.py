"""Pure cleanup operations for MuScriptor note events."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

Note: TypeAlias = tuple[str, int, float, float]


@dataclass(frozen=True)
class ChunkCleanupReport:
    notes: list[Note]
    chunks: frozenset[int]
    notes_removed: int


@dataclass(frozen=True)
class IntervalCleanupReport:
    notes: list[Note]
    near_duplicates_merged: int
    overlaps_truncated: int
    durations_clamped: int


def parse_chunk_indexes(value: str | None) -> set[int]:
    """Parse comma-separated, zero-based chunk indexes."""
    if not value:
        return set()
    try:
        chunks = {
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        }
    except ValueError as error:
        raise ValueError(
            "chunk indexes must be comma-separated integers"
        ) from error
    if any(chunk < 0 for chunk in chunks):
        raise ValueError("chunk indexes must be zero or greater")
    return chunks


def parse_missing_eos_chunks(log_text: str) -> set[int]:
    """Extract MuScriptor chunk indexes reported as missing EOS."""
    return {
        int(match)
        for match in re.findall(
            r"chunk\s+(\d+)\s+\(seek=[\d.]+s\)\s+did not emit EOS",
            log_text,
        )
    }


def drop_chunk_notes(
    notes: Iterable[Note],
    chunks: Iterable[int],
    *,
    chunk_seconds: float = 5.0,
) -> ChunkCleanupReport:
    """Remove notes intersecting any rejected fixed-duration chunk."""
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    source = list(notes)
    rejected = frozenset(chunks)
    kept = [
        note
        for note in source
        if not any(
            note[2] < (chunk + 1) * chunk_seconds
            and note[3] > chunk * chunk_seconds
            for chunk in rejected
        )
    ]
    return ChunkCleanupReport(
        notes=kept,
        chunks=rejected,
        notes_removed=len(source) - len(kept),
    )


def sanitize_note_intervals(
    notes: Iterable[Note],
    *,
    minimum_duration: float = 0.02,
    maximum_duration: float = 10.0,
) -> IntervalCleanupReport:
    """Make intervals safe for MIDI channel-and-pitch note-off semantics."""
    if minimum_duration <= 0:
        raise ValueError("minimum_duration must be positive")
    if maximum_duration < minimum_duration:
        raise ValueError(
            "maximum_duration must be at least minimum_duration"
        )

    grouped: dict[tuple[str, int], list[list[float]]] = {}
    durations_clamped = 0
    for instrument, pitch, on_value, off_value in notes:
        on = float(on_value)
        original_off = float(off_value)
        off = min(max(original_off, on + minimum_duration), on + maximum_duration)
        if off != original_off:
            durations_clamped += 1
        grouped.setdefault((instrument, pitch), []).append([on, off])

    cleaned: list[Note] = []
    duplicates = 0
    overlaps = 0
    for (instrument, pitch), intervals in grouped.items():
        intervals.sort()
        normalized: list[list[float]] = []
        for on, off in intervals:
            if normalized and on - normalized[-1][0] < minimum_duration:
                normalized[-1][1] = max(normalized[-1][1], off)
                duplicates += 1
                continue
            if normalized and on < normalized[-1][1]:
                normalized[-1][1] = on
                overlaps += 1
            normalized.append([on, off])
        cleaned.extend(
            (instrument, pitch, on, off)
            for on, off in normalized
            if off > on
        )
    cleaned.sort(key=lambda note: (note[2], note[0], note[1], note[3]))
    return IntervalCleanupReport(
        notes=cleaned,
        near_duplicates_merged=duplicates,
        overlaps_truncated=overlaps,
        durations_clamped=durations_clamped,
    )
