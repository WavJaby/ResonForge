"""Remove MIDI notes whose onsets occur in near-silent source audio."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from ..audio_loading import load_mono_audio
from .cleanup import _save_without_boundaries, read_midi_notes


def _frame_rms_dbfs(
    audio_path: Path,
    *,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    audio, sr = load_mono_audio(audio_path, sample_rate=sample_rate)
    if audio.size == 0:
        raise ValueError(f"empty audio: {audio_path}")
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    dbfs = 20.0 * np.log10(np.maximum(rms, 1e-12))
    times = librosa.frames_to_time(
        np.arange(rms.size),
        sr=sr,
        hop_length=hop_length,
    )
    return times, dbfs


def filter_quiet_onsets(
    midi_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    *,
    threshold_dbfs: float = -50.0,
    sample_rate: int = 22_050,
    frame_length: int = 1024,
    hop_length: int = 256,
    lookbehind_seconds: float = 0.02,
    lookahead_seconds: float = 0.08,
) -> dict:
    """Remove notes starting in quiet audio while preserving earlier sustains."""

    if threshold_dbfs > 0:
        raise ValueError("quiet-onset threshold must be 0 dBFS or lower")
    source = Path(midi_path)
    output = Path(output_path)
    notes, _ = read_midi_notes(source)
    frame_times, frame_dbfs = _frame_rms_dbfs(
        Path(audio_path),
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    remove_indices: set[int] = set()
    removed: list[dict] = []
    for note in notes:
        left = np.searchsorted(
            frame_times,
            max(0.0, note.start - lookbehind_seconds),
            side="left",
        )
        right = np.searchsorted(
            frame_times,
            note.start + lookahead_seconds,
            side="right",
        )
        local = frame_dbfs[left:right]
        onset_dbfs = float(np.max(local)) if local.size else -240.0
        if onset_dbfs < threshold_dbfs:
            remove_indices.update((note.on_index, note.off_index))
            removed.append(
                {
                    "channel": note.channel,
                    "pitch": note.pitch,
                    "start_seconds": round(note.start, 6),
                    "end_seconds": round(note.end, 6),
                    "onset_peak_dbfs": round(onset_dbfs, 3),
                }
            )

    _save_without_boundaries(source, output, remove_indices)
    return {
        "source": str(source.resolve()),
        "audio": str(Path(audio_path).resolve()),
        "output": str(output.resolve()),
        "threshold_dbfs": float(threshold_dbfs),
        "notes_before": len(notes),
        "notes_removed": len(removed),
        "notes_after": len(notes) - len(removed),
        "removed": removed,
    }
