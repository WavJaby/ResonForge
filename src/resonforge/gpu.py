"""Automatic GPU compute policy for transcription subprocesses."""

from __future__ import annotations

import torch


def muscriptor_dtype(gpu: int | None = None) -> str:
    """Select float32 for Pascal GPUs and float16 for Turing or newer."""
    if not torch.cuda.is_available():
        return "float32"

    device = torch.cuda.current_device() if gpu is None else gpu
    major, _minor = torch.cuda.get_device_capability(device)
    return "float16" if major >= 7 else "float32"
