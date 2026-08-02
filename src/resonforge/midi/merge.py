"""Merge MIDI sources while preserving real-time event positions."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import mido
import numpy as np

OUTPUT_TICKS_PER_BEAT = 480


def midi_text(value: str) -> str:
    """Return text that mido can safely write to a MIDI text field."""
    try:
        value.encode("latin-1")
    except UnicodeEncodeError:
        return quote(value, safe=" -_.[]()")
    return value


def merge_midis(
    inputs: list[Path],
    output: Path,
    source_gains_db: list[float],
    normalize_velocity: int | None,
    use_ports: bool = False,
    output_bpm: float = 120.0,
    track_names: list[str] | None = None,
    output_beats_per_bar: int | None = None,
    bar_phase_seconds: float | None = None,
) -> str:
    """Combine MIDI files at time zero, preserving real-time event positions."""
    if not np.isfinite(output_bpm) or output_bpm <= 0:
        raise ValueError(f"output BPM must be positive and finite: {output_bpm}")
    if len(source_gains_db) != len(inputs):
        raise ValueError("source_gains_db must contain one value per input")
    if track_names is not None and len(track_names) != len(inputs):
        raise ValueError("track_names must contain exactly one name per input")
    if output_beats_per_bar is not None and output_beats_per_bar <= 0:
        raise ValueError("output beats per bar must be positive")
    if bar_phase_seconds is not None and (
        not np.isfinite(bar_phase_seconds) or bar_phase_seconds < 0
    ):
        raise ValueError("bar phase must be non-negative and finite")
    output_tempo = mido.bpm2tempo(output_bpm)
    combined = mido.MidiFile(ticks_per_beat=OUTPUT_TICKS_PER_BEAT)
    tempo_track = mido.MidiTrack()
    tempo_track.append(
        mido.MetaMessage("set_tempo", tempo=output_tempo, time=0)
    )
    if output_beats_per_bar is not None:
        tempo_track.append(
            mido.MetaMessage(
                "time_signature",
                numerator=output_beats_per_bar,
                denominator=4,
                time=0,
            )
        )
    if bar_phase_seconds is not None:
        tempo_track.append(
            mido.MetaMessage(
                "marker",
                text=f"resonforge:bar_phase={bar_phase_seconds:.6f}",
                time=0,
            )
        )
    combined.tracks.append(tempo_track)

    free_channels = iter(channel for channel in range(16) if channel != 9)
    channel_map: dict[tuple[int, int], int] = {}
    note_messages: list[mido.Message] = []

    for source_index, path in enumerate(inputs):
        velocity_gain = 10.0 ** (source_gains_db[source_index] / 20.0)
        source = mido.MidiFile(path)
        source_track = mido.merge_tracks(source.tracks)
        output_track = mido.MidiTrack()
        track_name = (
            track_names[source_index]
            if track_names is not None
            else path.stem
        )
        output_track.append(
            mido.MetaMessage("track_name", name=midi_text(track_name), time=0)
        )
        if use_ports:
            output_track.append(
                mido.MetaMessage("midi_port", port=source_index, time=0)
            )

        tempo = 500_000
        elapsed_seconds = 0.0
        previous_tick = 0
        for message in source_track:
            elapsed_seconds += mido.tick2second(
                message.time,
                source.ticks_per_beat,
                tempo,
            )
            if message.type == "set_tempo":
                tempo = message.tempo
                continue
            if message.is_meta:
                continue

            # Wall-clock seconds -> destination BPM grid for cross-stem alignment.
            target_tick = round(
                elapsed_seconds
                / (output_tempo / 1_000_000)
                * OUTPUT_TICKS_PER_BEAT
            )
            copied = message.copy(time=target_tick - previous_tick)
            previous_tick = target_tick

            if hasattr(copied, "channel"):
                if use_ports:
                    new_channel = copied.channel
                elif copied.channel == 9:
                    new_channel = 9
                else:
                    key = (source_index, copied.channel)
                    if key not in channel_map:
                        try:
                            channel_map[key] = next(free_channels)
                        except StopIteration:
                            raise ValueError(
                                "merged MIDI needs more than 15 melodic channels"
                            ) from None
                    new_channel = channel_map[key]
                copied = copied.copy(channel=new_channel)
            if copied.type == "note_on" and copied.velocity > 0:
                copied.velocity = max(
                    1,
                    min(127, round(copied.velocity * velocity_gain)),
                )
                note_messages.append(copied)
            output_track.append(copied)

        output_track.append(mido.MetaMessage("end_of_track", time=0))
        combined.tracks.append(output_track)

    if normalize_velocity is not None and note_messages:
        highest = max(message.velocity for message in note_messages)
        scale = normalize_velocity / highest
        for message in note_messages:
            message.velocity = max(
                1,
                min(127, round(message.velocity * scale)),
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output)
    return "multi-port" if use_ports else "single-port"
