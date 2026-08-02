"""In-process MT3 model loading, inference, and progress routing."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np

MT3_PEAK_TARGET = 10 ** (-1.0 / 20.0)
_PROGRESS: ContextVar[Callable[[int, int], None] | None] = ContextVar(
    "mt3_progress",
    default=None,
)
_MODELS: dict[tuple[str, str], Any] = {}
_MODEL_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_DEVICE_LOCKS: dict[str, threading.Lock] = {}
_LOAD_LOCK = threading.Lock()


def invalid_midi_programs(
    valid_programs: tuple[int, ...] | frozenset[int],
) -> tuple[int, ...]:
    valid = frozenset(valid_programs)
    if any(program < 0 or program > 127 for program in valid):
        raise ValueError("MIDI programs must be in the inclusive range 0..127")
    return tuple(program for program in range(128) if program not in valid)


def condition_audio(
    samples: np.ndarray,
    *,
    peak_target: float = MT3_PEAK_TARGET,
) -> tuple[np.ndarray, float, bool]:
    conditioned = np.asarray(samples, dtype=np.float32)
    if conditioned.size == 0:
        raise ValueError("MT3 audio input is empty")
    if not np.all(np.isfinite(conditioned)):
        conditioned = np.nan_to_num(
            conditioned,
            copy=True,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    input_peak = float(np.max(np.abs(conditioned)))
    normalized = input_peak > peak_target
    if normalized and input_peak > 0.0:
        conditioned = conditioned * (peak_target / input_peak)
    return conditioned, input_peak, normalized


def _report(completed: int, total: int) -> None:
    callback = _PROGRESS.get()
    if callback is not None:
        callback(completed, total)


def _mt3_pytorch_forward(self: Any, features: np.ndarray) -> np.ndarray:
    import torch

    if not self._model_loaded:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    inputs = torch.from_numpy(features).float().to(self._device)
    total = len(inputs)
    _report(0, total)
    outputs = []
    for completed, batch in enumerate(inputs, start=1):
        batch = batch.unsqueeze(0)
        with torch.no_grad():
            result = self._model.generate(
                inputs=batch,
                max_length=self.MAX_GENERATION_LENGTH,
                num_beams=1,
                do_sample=False,
                length_penalty=0.4,
                eos_token_id=self._model.config.eos_token_id,
                early_stopping=False,
            )
        outputs.append(self._postprocess_batch(result).cpu().numpy())
        _report(completed, total)
    max_length = max(output.shape[1] for output in outputs)
    padded = [
        np.pad(
            output,
            ((0, 0), (0, max_length - output.shape[1])),
            constant_values=-1,
        )
        for output in outputs
    ]
    return np.concatenate(padded, axis=0)


def _yourmt3_forward(self: Any, features: Any) -> Any:
    if self.model is None:
        raise RuntimeError("Model not loaded")
    features = features.to(self.device_str)
    batch_size = 8
    total = int(features.shape[0])
    _report(0, total)
    predictions = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        result = self.model.inference_file(
            bsz=batch_size,
            audio_segments=features[start:end],
        )
        batch_predictions = result[0] if isinstance(result, tuple) else result
        predictions.extend(batch_predictions)
        _report(end, total)
    return predictions


def _instrument_model(model: Any, model_name: str) -> None:
    if getattr(model, "_resonforge_progress", False):
        return
    if model_name == "mt3_pytorch":
        model.forward = MethodType(_mt3_pytorch_forward, model)
    elif model_name == "yourmt3":
        model.forward = MethodType(_yourmt3_forward, model)
    else:
        raise ValueError(f"unsupported MT3 model: {model_name}")
    model._resonforge_progress = True


def _load_model(model_name: str, device: str, checkpoint_dir: Path) -> Any:
    key = (model_name, device)
    cached = _MODELS.get(key)
    if cached is not None:
        return cached
    with _LOAD_LOCK:
        cached = _MODELS.get(key)
        if cached is None:
            os.environ["MT3_CHECKPOINT_DIR"] = str(checkpoint_dir.resolve())
            from mt3_infer import load_model

            cached = load_model(model_name, device=device, cache=False)
            _instrument_model(cached, model_name)
            _MODELS[key] = cached
            _MODEL_LOCKS[key] = threading.Lock()
            _DEVICE_LOCKS.setdefault(device, threading.Lock())
    return cached


def transcribe_audio(
    audio: Path,
    output: Path,
    *,
    model_name: str,
    device: str,
    checkpoint_dir: Path,
    valid_programs: tuple[int, ...] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    if not audio.is_file():
        raise FileNotFoundError(f"audio input does not exist: {audio}")
    output.parent.mkdir(parents=True, exist_ok=True)

    from mt3_infer.utils.audio import load_audio

    samples, sample_rate = load_audio(str(audio))
    samples, _, _ = condition_audio(samples)
    key = (model_name, device)
    model = _load_model(model_name, device, checkpoint_dir)
    token = _PROGRESS.set(progress_callback)
    try:
        with _DEVICE_LOCKS[device], _MODEL_LOCKS[key]:
            if valid_programs is not None:
                configure = getattr(model, "set_valid_programs", None)
                if configure is not None:
                    configure(valid_programs)
            midi = model.transcribe(samples, sr=sample_rate)
    finally:
        _PROGRESS.reset(token)
    midi.save(str(output))
    if not output.is_file():
        raise RuntimeError(f"MT3 did not create: {output}")
