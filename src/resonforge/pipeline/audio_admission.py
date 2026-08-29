"""Lazy stem decoding and bounded in-memory preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf

from resonforge.transcribers.base import StemTask

from ..audio.buffer import AudioBuffer
from ..scheduler.host_memory import HostMemoryBudget, HostMemoryReservation
from .types import PipelineConfig


@dataclass(frozen=True)
class AdmittedStemAudio:
    task: StemTask
    preprocessing: dict[str, object]
    reservation: HostMemoryReservation


def _source_paths(task: StemTask) -> tuple[Path, ...]:
    return task.sources or (task.audio,)


def estimate_working_bytes(task: StemTask) -> int:
    """Conservatively estimate decode + mix/gate working arrays."""
    decoded = 0
    for path in _source_paths(task):
        info = sf.info(path)
        decoded += info.frames * info.channels * 4
    # One decoded source plus gate/mix output and analysis/resampling headroom.
    return max(1, decoded * 3)


def load_task_audio(task: StemTask) -> AudioBuffer:
    """Decode one stem, or lazily combine its source stems."""
    sources = _source_paths(task)
    if len(sources) == 1:
        return AudioBuffer.read(sources[0])
    loaded = [AudioBuffer.read(path) for path in sources]
    if len({audio.sample_rate for audio in loaded}) != 1:
        raise RuntimeError("band stems have mismatched sample rates")
    if len({audio.samples.shape[1] for audio in loaded}) != 1:
        raise RuntimeError("band stems have mismatched channel counts")
    length = max(len(audio.samples) for audio in loaded)
    combined = np.zeros((length, loaded[0].samples.shape[1]), dtype=np.float32)
    for audio in loaded:
        combined[: len(audio.samples)] += audio.samples
    peak = float(np.max(np.abs(combined)))
    if peak > 0.99:
        combined *= 0.99 / peak
    return loaded[0].replace(combined)


def prepare_admitted_audio(
    task: StemTask,
    *,
    config: PipelineConfig,
    budget: HostMemoryBudget,
    stop_event,
):
    """Yield a task carrying audio only while its global lease is held."""
    requested = estimate_working_bytes(task)
    lease = budget.acquire(requested, stop_event=stop_event)

    class _PreparedLease:
        def __enter__(self) -> AdmittedStemAudio:
            reservation = lease.__enter__()
            try:
                audio = load_task_audio(task)
                metrics: dict[str, object] = {}
                return AdmittedStemAudio(
                    replace(task, buffer=audio),
                    metrics,
                    reservation,
                )
            except BaseException:
                lease.__exit__(None, None, None)
                raise

        def __exit__(self, *args: object) -> None:
            lease.__exit__(*args)

    return _PreparedLease()
