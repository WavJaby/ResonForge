"""Convert MuScriptor JSONL events to MIDI and optional stereo previews."""

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mido

from .document import MidiDocument
from .intervals import (
    Note,
    drop_chunk_notes,
    parse_chunk_indexes,
    parse_missing_eos_chunks,
    sanitize_note_intervals,
)

# Known MuScriptor group -> GM program; unknown groups fail instead of guessing.
GM = {
    "acoustic_piano": 0, "electric_piano": 4, "chromatic_percussion": 12,
    "organ": 19, "acoustic_guitar": 24,
    "clean_electric_guitar": 27, "distorted_electric_guitar": 30, "electric_bass": 33,
    "acoustic_bass": 32, "voice": 52, "string_ensemble": 48, "synth_strings": 50,
    "violin": 40, "viola": 41, "cello": 42, "contrabass": 43,
    "orchestral_harp": 46, "orchestra_hit": 55,
    "flutes": 73, "soprano_and_alto_sax": 65, "tenor_and_baritone_sax": 66,
    "tenor_sax": 66, "baritone_sax": 67, "trumpet": 56,
    "trombone": 57, "french_horn": 60, "brass_section": 61,
    "english_horn": 69, "synth_lead": 80, "synth_pad": 88, "timpani": 47,
    "mallets": 12, "harp": 46, "accordion": 21, "banjo": 105, "clarinet": 71, "oboe": 68,
    "bassoon": 70, "tuba": 58, "synth_bass": 38, "harpsichord": 6,
    "program_98": 98,
}
DRUMS = {"drums"}
TPB = 480


class MidiConversionError(RuntimeError):
    """Raised when MuScriptor events cannot produce a usable MIDI file."""


@dataclass(frozen=True)
class MidiConversionResult:
    document: MidiDocument
    notes: tuple[Note, ...]
    report: dict[str, object]


def convert_events_to_midi(
    events,
    *,
    rejected_chunks: set[int] | frozenset[int] = frozenset(),
    maximum_note_seconds: float = 10.0,
    cleanup: bool = False,
    bpm: float = 120.0,
    allow_empty: bool = False,
) -> MidiConversionResult:
    """Convert MuScriptor payloads without creating an intermediate file."""
    notes = events_to_notes(events, allow_empty=allow_empty)
    input_notes = len(notes)
    chunk_report = drop_chunk_notes(notes, rejected_chunks)
    notes = chunk_report.notes
    if not notes and not allow_empty:
        raise MidiConversionError("no notes remain after rejected chunks")
    if cleanup:
        interval_report = sanitize_note_intervals(
            notes,
            maximum_duration=maximum_note_seconds,
        )
        notes = interval_report.notes
        duplicates = interval_report.near_duplicates_merged
        overlaps = interval_report.overlaps_truncated
        durations_clamped = interval_report.durations_clamped
    else:
        duplicates = overlaps = durations_clamped = 0
    return MidiConversionResult(
        document=notes_to_midi_document(notes, bpm=bpm),
        notes=tuple(notes),
        report={
            "input_notes": input_notes,
            "output_notes": len(notes),
            "missing_eos_chunks": sorted(rejected_chunks),
            "missing_eos_removed": chunk_report.notes_removed,
            "near_duplicates_merged": duplicates,
            "overlaps_truncated": overlaps,
            "durations_clamped": durations_clamped,
        },
    )


@dataclass(frozen=True)
class MidiConversionConfig:
    jsonl: Path
    out: Path | None = None
    auralize: Path | None = None
    audio: Path | None = None
    soundfont: Path | None = None
    only: str | None = None
    drop_eos_log: Path | None = None
    drop_chunks: str | None = None
    max_note_seconds: float = 10.0
    cleanup: bool = False


def load_notes(path: Path) -> list[Note]:
    return events_to_notes(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def events_to_notes(events, *, allow_empty: bool = False) -> list[Note]:
    """Resolve MuScriptor event payloads into note intervals in memory."""
    starts: dict[int, dict] = {}
    notes: list[Note] = []
    for ev in events:
        if ev["type"] == "start":
            starts[ev["index"]] = ev
        elif ev["type"] == "end":
            # CLI: index reference; Python API: nested start event.
            s = ev.get("start_event") or starts.get(ev.get("start_event_index"))
            if s is None:
                raise ValueError(f"end without start: {ev}")
            notes.append((s["instrument"], s["pitch"], s["start_time"], ev["end_time"]))
    if not notes and not allow_empty:
        raise MidiConversionError("no notes in jsonl")
    return notes


def notes_to_midi_document(notes, bpm=120.0) -> MidiDocument:
    spb = 60.0 / bpm
    instruments = sorted({n[0] for n in notes})
    unknown = [i for i in instruments if i not in GM and i not in DRUMS]
    if unknown:
        raise MidiConversionError(
            f"no GM mapping for {unknown} -- extend the GM map"
        )

    mid = mido.MidiFile(ticks_per_beat=TPB)
    tempo_trk = mido.MidiTrack()
    tempo_trk.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    mid.tracks.append(tempo_trk)
    channels = {}
    ch_iter = iter([c for c in range(16) if c != 9])
    for inst in instruments:
        channels[inst] = 9 if inst in DRUMS else next(ch_iter)

    for inst in instruments:
        trk = mido.MidiTrack()
        mid.tracks.append(trk)
        trk.append(mido.MetaMessage("track_name", name=inst, time=0))
        ch = channels[inst]
        if inst not in DRUMS:
            trk.append(mido.Message("program_change", channel=ch, program=GM[inst], time=0))
        evs = []
        for i, p, on, off in notes:
            if i != inst:
                continue
            if off <= on:
                off = on + 0.05  # Onset-only drum -> nominal duration.
            evs.append((on, "on", p))
            evs.append((off, "off", p))
        evs.sort(key=lambda e: (e[0], e[1] == "on"))  # Same tick: off -> on.
        # Round absolute ticks; rounding deltas accumulates timing drift.
        prev_tick = 0
        for t, kind, p in evs:
            tick = int(round(t / spb * TPB))
            dt = tick - prev_tick
            prev_tick = tick
            if kind == "on":
                trk.append(mido.Message("note_on", channel=ch, note=p, velocity=90, time=dt))
            else:
                trk.append(mido.Message("note_off", channel=ch, note=p, velocity=0, time=dt))
    return MidiDocument.from_mido(mid)


def to_midi(notes, out: Path, bpm=120.0):
    document = notes_to_midi_document(notes, bpm=bpm)
    document.write(out)
    return sorted({note[0] for note in notes})


def rewrite_midi_program(
    midi_path: Path | str,
    instrument: str,
    *,
    output_path: Path | str | None = None,
) -> Path:
    """Collapse a stem MIDI to one MuScriptor-compatible instrument track.

    Basic Pitch exports Electric Piano 1 regardless of the source stem. This
    post-processing merges model-assigned tracks and channels because the
    pipeline input is already a single separated stem. Note timing and
    velocity are preserved. Drums are moved to MIDI channel 10.
    """
    if instrument not in GM and instrument not in DRUMS:
        raise ValueError(f"no GM mapping for {instrument!r}")

    source = Path(midi_path)
    destination = Path(output_path) if output_path is not None else source
    midi = mido.MidiFile(source)

    target_channel = 9 if instrument in DRUMS else 0
    merged = mido.merge_tracks(midi.tracks)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=instrument, time=0))
    if instrument not in DRUMS:
        track.append(
            mido.Message(
                "program_change",
                channel=target_channel,
                program=GM[instrument],
                time=0,
            )
        )

    pending_time = 0
    for message in merged:
        pending_time += message.time
        if message.type in {"program_change", "track_name", "end_of_track"}:
            continue
        rewritten = message.copy(time=pending_time)
        if not rewritten.is_meta and hasattr(rewritten, "channel"):
            rewritten.channel = target_channel
        track.append(rewritten)
        pending_time = 0
    track.append(mido.MetaMessage("end_of_track", time=pending_time))
    midi.tracks = [track]

    destination.parent.mkdir(parents=True, exist_ok=True)
    midi.save(destination)
    return destination


def auralize(mid_path: Path, audio_path: Path, out_wav: Path, soundfont: Path):
    import numpy as np
    import soundfile as sf

    orig, sr_o = sf.read(audio_path, dtype="float32", always_2d=True)
    with tempfile.TemporaryDirectory() as td:
        synth_wav = Path(td) / "synth.wav"

        r = subprocess.run(
            ["fluidsynth", "-ni", "-F", str(synth_wav), "-r", str(sr_o),
             str(soundfont), str(mid_path)],
            capture_output=True, text=True)
        if r.returncode != 0 or not synth_wav.exists():
            raise MidiConversionError(
                f"fluidsynth failed: {r.stderr[-500:]}"
            )
        synth, _ = sf.read(synth_wav, dtype="float32", always_2d=True)
    L = orig.mean(axis=1)
    R = synth.mean(axis=1)
    n = max(len(L), len(R))
    L = np.pad(L, (0, n - len(L)))
    R = np.pad(R, (0, n - len(R)))
    R *= 0.9 / max(1e-6, np.abs(R).max())
    sf.write(out_wav, np.stack([L, R], axis=1), sr_o)


def find_soundfont() -> Path:
    # Share MuScriptor's HF_HOME-aware cache resolution.
    try:
        from muscriptor.utils.auralization import _resolve_soundfont
        return _resolve_soundfont(None)
    except Exception as exc:
        raise MidiConversionError(
            "could not resolve MuseScore_General.sf2 -- pass --soundfont\n"
            f"{exc}"
        ) from exc


def convert_jsonl_to_midi(
    jsonl: Path,
    *,
    out: Path | None = None,
    auralize: Path | None = None,
    audio: Path | None = None,
    soundfont: Path | None = None,
    only: str | None = None,
    drop_eos_log: Path | None = None,
    drop_chunks: str | None = None,
    max_note_seconds: float = 10.0,
    cleanup: bool = False,
    verbose: bool = True,
    report: dict[str, object] | None = None,
) -> Path:
    """Convert MuScriptor JSONL output to MIDI without going through a CLI."""
    config = MidiConversionConfig(
        jsonl=Path(jsonl),
        out=Path(out) if out is not None else None,
        auralize=Path(auralize) if auralize is not None else None,
        audio=Path(audio) if audio is not None else None,
        soundfont=Path(soundfont) if soundfont is not None else None,
        only=only,
        drop_eos_log=Path(drop_eos_log) if drop_eos_log is not None else None,
        drop_chunks=drop_chunks,
        max_note_seconds=max_note_seconds,
        cleanup=cleanup,
    )
    return _convert_jsonl(config, verbose=verbose, report=report)


def _convert_jsonl(
    config: MidiConversionConfig,
    *,
    verbose: bool = True,
    report: dict[str, object] | None = None,
) -> Path:
    out = config.out or config.jsonl.with_suffix(".mid")
    notes = load_notes(config.jsonl)
    input_notes = len(notes)
    eos_removed = 0
    bad_chunks = parse_chunk_indexes(config.drop_chunks)
    if config.drop_eos_log:
        if not config.drop_eos_log.is_file():
            raise MidiConversionError(
                f"EOS log not found: {config.drop_eos_log}"
            )
        log_text = config.drop_eos_log.read_text(
            encoding="utf-8", errors="replace"
        )
        bad_chunks.update(parse_missing_eos_chunks(log_text))
    if bad_chunks:
        chunk_report = drop_chunk_notes(notes, bad_chunks)
        notes = chunk_report.notes
        eos_removed = chunk_report.notes_removed
        if verbose:
            print(
                f"cleaned MIDI: removed {eos_removed} "
                f"{'note' if eos_removed == 1 else 'notes'} "
                "from missing-EOS chunks"
            )
        if not notes:
            raise MidiConversionError(
                "nothing left after dropping bad chunks"
            )
    if config.only:
        keep = {s.strip() for s in config.only.split(",")}
        before = len(notes)
        notes = [n for n in notes if n[0] in keep]
        if verbose:
            print(f"filter {sorted(keep)}: {before} -> {len(notes)} notes")
        if not notes:
            raise MidiConversionError("nothing left after filter")
    if config.max_note_seconds <= 0:
        raise ValueError("--max-note-seconds must be positive")
    if config.cleanup:
        interval_report = sanitize_note_intervals(
            notes,
            maximum_duration=config.max_note_seconds,
        )
        notes = interval_report.notes
        duplicates = interval_report.near_duplicates_merged
        overlaps = interval_report.overlaps_truncated
        durations_clamped = interval_report.durations_clamped
    else:
        duplicates = 0
        overlaps = 0
        durations_clamped = 0
    cleaned = []
    if duplicates:
        cleaned.append(
            f"merged {duplicates} near-duplicate "
            f"{'note' if duplicates == 1 else 'notes'}"
        )
    if overlaps:
        cleaned.append(
            f"truncated {overlaps} overlapping "
            f"{'note' if overlaps == 1 else 'notes'}"
        )
    if cleaned and verbose:
        print(f"cleaned MIDI: {', '.join(cleaned)}")
    instruments = to_midi(notes, out)
    if report is not None:
        report.update(
            {
                "input_notes": input_notes,
                "output_notes": len(notes),
                "missing_eos_chunks": sorted(bad_chunks),
                "missing_eos_removed": eos_removed,
                "near_duplicates_merged": duplicates,
                "overlaps_truncated": overlaps,
                "durations_clamped": durations_clamped,
            }
        )
    if verbose:
        print(f"{out}  ({len(notes)} notes, instruments: {', '.join(instruments)})")

    if config.auralize:
        if not config.audio:
            raise MidiConversionError("--auralize needs --audio <original>")
        sf2 = config.soundfont or find_soundfont()
        auralize(out, config.audio, config.auralize, sf2)
        if verbose:
            print(f"{config.auralize}  (L=original, R=synth)")
    return out
