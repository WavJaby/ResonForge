"""Deterministic libsndfile audio loading for analysis paths."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_mono_audio(
    path: str | Path,
    *,
    sample_rate: int,
) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 and resample with SoXR when needed."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    source = Path(path)
    audio, source_rate = sf.read(source, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if source_rate != sample_rate and mono.size:
        mono = librosa.resample(
            mono,
            orig_sr=source_rate,
            target_sr=sample_rate,
            res_type="soxr_hq",
        )
    return np.asarray(mono, dtype=np.float32), sample_rate
