"""MIDI, CSV, and baseline artifacts for the general Basic Pitch decoder."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ._runtime import _basic_pitch_runtime
from .candidates import Candidate, frames_for_ms

PROGRAMS = {
    "bass": 33,  # Electric Bass (finger)
    "guitar": 27,  # Electric Guitar (clean)
    "piano": 4,  # Electric Piano 1
}


def write_midi(
    notes: Sequence[tuple[float, float, Candidate]],
    output_path: Path,
    stem: str,
) -> None:
    pretty_midi = _basic_pitch_runtime().pretty_midi
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(
        program=PROGRAMS.get(stem, PROGRAMS["piano"]),
        name=f"{stem} - Basic Pitch general",
    )
    for start, end, candidate in notes:
        velocity = int(np.clip(round(candidate.confidence * 127), 1, 127))
        instrument.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=candidate.pitch,
                start=start,
                end=end,
            )
        )
    midi.instruments.append(instrument)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))


def max_polyphony(notes: Sequence[tuple[float, float, Candidate]]) -> int:
    events: list[tuple[float, int]] = []
    for start, end, _ in notes:
        events.append((start, 1))
        events.append((end, -1))
    active = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def write_csv(
    notes: Sequence[tuple[float, float, Candidate]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "start_s",
                "end_s",
                "pitch_midi",
                "confidence",
                "onset_confidence",
                "note_confidence",
                "contour_confidence",
                "source",
            ]
        )
        for start, end, candidate in notes:
            writer.writerow(
                [
                    f"{start:.6f}",
                    f"{end:.6f}",
                    candidate.pitch,
                    f"{candidate.confidence:.6f}",
                    f"{candidate.onset_confidence:.6f}",
                    f"{candidate.note_confidence:.6f}",
                    f"{candidate.contour_confidence:.6f}",
                    candidate.source,
                ]
            )


def note_count_by_source(candidates: Sequence[Candidate]) -> dict[str, int]:
    return {
        source: sum(candidate.source == source for candidate in candidates)
        for source in ("onset", "frame")
    }


def save_baseline(
    output: dict[str, np.ndarray],
    output_path: Path,
    *,
    onset_threshold: float,
    frame_threshold: float,
    onset_min_ms: float,
) -> int:
    baseline_midi, baseline_notes = _basic_pitch_runtime().model_output_to_notes(
        output,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=frames_for_ms(max(120.0, onset_min_ms)),
        include_pitch_bends=False,
        melodia_trick=True,
    )
    baseline_midi.write(str(output_path))
    return len(baseline_notes)
