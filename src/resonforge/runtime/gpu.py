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


# **`cuda_graph_pool_usage` deleted 2026-08-27.** It split the CUDA graph
# private pool into its reserved and allocated halves so a ledger could
# subtract them without double counting. Nothing called it: the graph pool's
# bytes are reserved from the driver through the caching allocator, so
# `mem_get_info` free has already excluded them by the time anything reads it,
# and no separate accounting is needed. `cuda_graph_cache_pool_bytes_peak`
# still records what a run held.
