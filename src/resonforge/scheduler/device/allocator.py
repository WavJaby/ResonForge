"""Process-global GPU affinity for concurrently running pipeline sessions."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionDeviceSnapshot:
    session_id: str
    gpu: int | None
    eligible_gpus: tuple[int, ...]
    active_sessions_on_gpu: int


class SessionDeviceLease:
    """Keep one pipeline session on one GPU until its transcription ends."""

    def __init__(
        self,
        allocator: SessionDeviceAllocator,
        snapshot: SessionDeviceSnapshot,
    ) -> None:
        self._allocator = allocator
        self.snapshot = snapshot
        self._closed = False
        self._lock = threading.Lock()

    @property
    def gpu(self) -> int | None:
        return self.snapshot.gpu

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._allocator._release(self.snapshot.session_id, self.snapshot.gpu)


class SessionDeviceAllocator:
    """Assign each song to one least-loaded eligible device."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_by_gpu: dict[int | None, int] = {}
        self._active_sessions: dict[str, int | None] = {}

    def acquire(
        self,
        session_id: str,
        eligible_gpus: tuple[int, ...],
    ) -> SessionDeviceLease:
        eligible = tuple(dict.fromkeys(eligible_gpus))
        candidates: tuple[int | None, ...] = eligible or (None,)
        with self._lock:
            if session_id in self._active_sessions:
                raise RuntimeError(f"session already has a GPU assignment: {session_id}")
            gpu = min(candidates, key=lambda item: self._active_by_gpu.get(item, 0))
            active = self._active_by_gpu.get(gpu, 0) + 1
            self._active_by_gpu[gpu] = active
            self._active_sessions[session_id] = gpu
        return SessionDeviceLease(
            self,
            SessionDeviceSnapshot(session_id, gpu, eligible, active),
        )

    def _release(self, session_id: str, gpu: int | None) -> None:
        with self._lock:
            assigned = self._active_sessions.pop(session_id, object())
            if assigned != gpu:
                raise RuntimeError(f"invalid GPU assignment release: {session_id}")
            remaining = self._active_by_gpu[gpu] - 1
            if remaining:
                self._active_by_gpu[gpu] = remaining
            else:
                del self._active_by_gpu[gpu]

    def snapshot(self) -> dict[int | None, int]:
        with self._lock:
            return dict(self._active_by_gpu)


_GLOBAL_ALLOCATOR = SessionDeviceAllocator()


def acquire_session_device(
    session_id: str,
    eligible_gpus: tuple[int, ...],
) -> SessionDeviceLease:
    return _GLOBAL_ALLOCATOR.acquire(session_id, eligible_gpus)


def global_device_allocator_snapshot() -> dict[int | None, int]:
    return _GLOBAL_ALLOCATOR.snapshot()
