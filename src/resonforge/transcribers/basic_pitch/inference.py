"""Basic Pitch window inference over the project's audio loader."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ...audio_loading import load_mono_audio


def predict_model_output(
    audio_path: str | Path,
    model: Any,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Run Basic Pitch windows without its audioread-based file loader."""
    from basic_pitch.constants import AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP
    from basic_pitch.inference import unwrap_output, window_audio_file

    overlapping_frames = 30
    overlap_len = overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len
    audio, _ = load_mono_audio(audio_path, sample_rate=AUDIO_SAMPLE_RATE)
    original_length = audio.shape[0]
    if original_length == 0:
        raise RuntimeError(f"Basic Pitch received empty audio: {audio_path}")
    padded = np.concatenate(
        (np.zeros(overlap_len // 2, dtype=np.float32), audio)
    )
    total = max(1, math.ceil(padded.shape[0] / hop_size))
    if progress is not None:
        progress(0, total)

    output: dict[str, list[np.ndarray]] = {
        "note": [],
        "onset": [],
        "contour": [],
    }
    completed = 0
    for window, _ in window_audio_file(padded, hop_size):
        batch = np.expand_dims(window, axis=0)
        for name, values in model.predict(batch).items():
            output[name].append(values)
        completed += 1
        if progress is not None:
            progress(completed, total)

    if completed == 0:
        raise RuntimeError(f"Basic Pitch received empty audio: {audio_path}")
    return {
        name: unwrap_output(
            np.concatenate(values),
            original_length,
            overlapping_frames,
        )
        for name, values in output.items()
    }
