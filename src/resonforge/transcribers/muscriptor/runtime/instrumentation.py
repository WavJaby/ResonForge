"""Timing the phases a generation session runs, on the device's own clock.

Separate from both sides because both are measured by it, and because what it
reports is easy to over-read: these are **CUDA events on the session's stream**,
so a span covers everything the stream did between them -- not the device time
of the phase. The one exception is a span around a graph replay, where the host
does not participate at all.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager

import torch


class _CudaPhaseRecorder:
    """Collect CUDA phase time without introducing synchronization barriers."""

    def __init__(self, device: torch.device) -> None:
        self.enabled = device.type == "cuda" and torch.cuda.is_available()
        self.pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.completed: Counter[str] = Counter()

    @contextmanager
    def measure(self, phase: str):
        if not self.enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.pending.append((phase, start, end))

    def drain(self) -> dict[str, float]:
        unresolved = []
        for phase, start, end in self.pending:
            if end.query():
                self.completed[phase] += start.elapsed_time(end)
            else:
                unresolved.append((phase, start, end))
        self.pending = unresolved
        result = dict(self.completed)
        self.completed.clear()
        return result

