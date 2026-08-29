"""Process-global admission control for large in-memory audio buffers."""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass

from resonforge.scheduler.preparation_service import in_preparation_worker


@dataclass(frozen=True)
class HostMemoryReservation:
    """One admitted allocation, released when its context exits."""

    requested_bytes: int
    admitted_bytes: int
    wait_seconds: float


class _Lease(AbstractContextManager[HostMemoryReservation]):
    def __init__(self, budget: HostMemoryBudget, reservation: HostMemoryReservation):
        self._budget = budget
        self._reservation = reservation
        self._released = False

    def __enter__(self) -> HostMemoryReservation:
        return self._reservation

    def __exit__(self, *_args: object) -> None:
        if not self._released:
            self._released = True
            self._budget._release(self._reservation.admitted_bytes)


class HostMemoryBudget:
    """FIFO byte budget shared by all sessions in one server process."""

    def __init__(self, capacity_bytes: int):
        if capacity_bytes <= 0:
            raise ValueError("host memory budget must be positive")
        self.capacity_bytes = capacity_bytes
        self._condition = threading.Condition()
        self._used_bytes = 0
        self._peak_bytes = 0
        self._waiters: deque[object] = deque()

    def acquire(
        self,
        requested_bytes: int,
        *,
        stop_event: threading.Event | None = None,
    ) -> _Lease:
        if requested_bytes <= 0:
            raise ValueError("requested host memory must be positive")
        if in_preparation_worker():
            # The budget is released by stems whose region producer runs on a
            # preparation lane. Waiting here on a lane worker therefore waits
            # for work that cannot start, and the process wedges silently
            # (R7). Admission belongs on the thread that consumes the audio.
            raise RuntimeError(
                "host memory admission cannot block a preparation worker"
            )
        admitted = min(requested_bytes, self.capacity_bytes)
        started = time.monotonic()
        waiter = object()
        with self._condition:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("host memory admission cancelled")
            self._waiters.append(waiter)
            while self._waiters[0] is not waiter or self._used_bytes + admitted > self.capacity_bytes:
                if stop_event is not None and stop_event.is_set():
                    self._waiters.remove(waiter)
                    self._condition.notify_all()
                    raise RuntimeError("host memory admission cancelled")
                self._condition.wait(timeout=0.1)
            self._waiters.popleft()
            self._used_bytes += admitted
            self._peak_bytes = max(self._peak_bytes, self._used_bytes)
            self._condition.notify_all()
        return _Lease(
            self,
            HostMemoryReservation(
                requested_bytes=requested_bytes,
                admitted_bytes=admitted,
                wait_seconds=time.monotonic() - started,
            ),
        )

    def try_acquire_exact(self, requested_bytes: int) -> _Lease | None:
        """Reserve the full request without blocking scheduler progress."""
        if requested_bytes <= 0:
            raise ValueError("requested host memory must be positive")
        with self._condition:
            if (
                requested_bytes > self.capacity_bytes
                or self._waiters
                or self._used_bytes + requested_bytes > self.capacity_bytes
            ):
                return None
            self._used_bytes += requested_bytes
            self._peak_bytes = max(self._peak_bytes, self._used_bytes)
        return _Lease(
            self,
            HostMemoryReservation(
                requested_bytes=requested_bytes,
                admitted_bytes=requested_bytes,
                wait_seconds=0.0,
            ),
        )

    def _release(self, admitted_bytes: int) -> None:
        with self._condition:
            self._used_bytes -= admitted_bytes
            self._condition.notify_all()

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            return {
                "capacity_bytes": self.capacity_bytes,
                "used_bytes": self._used_bytes,
                "peak_bytes": self._peak_bytes,
            }


_LOCK = threading.Lock()
_GLOBAL_BUDGET: HostMemoryBudget | None = None


def global_host_memory_budget(capacity_mib: int) -> HostMemoryBudget:
    """Return the single process budget, rejecting conflicting server config."""
    global _GLOBAL_BUDGET
    capacity_bytes = capacity_mib * 1024**2
    with _LOCK:
        if _GLOBAL_BUDGET is None:
            _GLOBAL_BUDGET = HostMemoryBudget(capacity_bytes)
        elif _GLOBAL_BUDGET.capacity_bytes != capacity_bytes:
            raise RuntimeError(
                "host memory budget is process-global and already configured as "
                f"{_GLOBAL_BUDGET.capacity_bytes // 1024**2} MiB"
            )
        return _GLOBAL_BUDGET


def current_host_memory_budget() -> HostMemoryBudget | None:
    """Return the configured process budget without creating a second policy."""
    with _LOCK:
        return _GLOBAL_BUDGET


def shutdown_host_memory_budget() -> None:
    """Reset the process service after all sessions have stopped."""
    global _GLOBAL_BUDGET
    with _LOCK:
        if _GLOBAL_BUDGET is not None and _GLOBAL_BUDGET.snapshot()["used_bytes"]:
            raise RuntimeError("cannot reset host memory budget while leases are active")
        _GLOBAL_BUDGET = None
