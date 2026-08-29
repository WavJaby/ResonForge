"""Immutable in-memory audio passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioBuffer:
    samples: np.ndarray
    sample_rate: int

    @classmethod
    def read(cls, path: str | Path) -> AudioBuffer:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if not samples.size:
            raise ValueError(f"empty audio: {path}")
        samples.setflags(write=False)
        return cls(samples, sample_rate)

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    def replace(self, samples: np.ndarray) -> AudioBuffer:
        value = np.asarray(samples, dtype=np.float32)
        value.setflags(write=False)
        return AudioBuffer(value, self.sample_rate)

    def mono(self, sample_rate: int) -> np.ndarray:
        mono = np.mean(self.samples, axis=1, dtype=np.float32)
        if self.sample_rate != sample_rate and mono.size:
            mono = librosa.resample(
                mono,
                orig_sr=self.sample_rate,
                target_sr=sample_rate,
                res_type="soxr_hq",
            )
        return np.asarray(mono, dtype=np.float32)
