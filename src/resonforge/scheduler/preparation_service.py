"""Process-global bounded CPU work service shared by pipeline sessions."""

from __future__ import annotations

import atexit
import os
import queue
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Literal

PreparationLane = Literal["critical", "background"]


@dataclass(frozen=True)
class _DaemonWork:
    future: Future[Any]
    function: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


_LANE_MARKER = threading.local()


def in_preparation_worker() -> bool:
    """True on a preparation lane thread.

    Every lane is bounded, and lane work is what releases the resources lane
    work waits for. A blocking wait taken *on* a worker can therefore close a
    cycle that no watchdog sees -- R7 wedged four songs exactly that way. Waits
    that can be starved this way test this and refuse.
    """
    return getattr(_LANE_MARKER, "active", False)


class _DaemonExecutor:
    """Bounded daemon workers that cannot retain a terminal pipeline process."""

    _STOP = object()

    def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
        self._queue: queue.Queue[_DaemonWork | object] = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._threads = tuple(
            threading.Thread(
                target=self._worker,
                name=f"{thread_name_prefix}_{index}",
                daemon=True,
            )
            for index in range(max_workers)
        )
        for thread in self._threads:
            thread.start()

    def submit(
        self,
        function: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        future: Future[Any] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("preparation executor is closed")
            self._queue.put(_DaemonWork(future, function, args, kwargs))
        return future

    def _worker(self) -> None:
        _LANE_MARKER.active = True
        while True:
            work = self._queue.get()
            if work is self._STOP:
                return
            assert isinstance(work, _DaemonWork)
            if not work.future.set_running_or_notify_cancel():
                continue
            try:
                result = work.function(*work.args, **work.kwargs)
            except BaseException as error:
                work.future.set_exception(error)
            else:
                work.future.set_result(result)

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                if cancel_futures:
                    retained: list[object] = []
                    while True:
                        try:
                            work = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if isinstance(work, _DaemonWork):
                            work.future.cancel()
                        else:
                            retained.append(work)
                    for work in retained:
                        self._queue.put(work)
                for _thread in self._threads:
                    self._queue.put(self._STOP)
        if wait:
            for thread in self._threads:
                thread.join()


class PreparationLease:
    """Session-scoped view whose cancellation never affects another run."""

    def __init__(self, service: GlobalPreparationService, session_id: str) -> None:
        self._service = service
        self.session_id = session_id
        self._futures: set[Future[Any]] = set()
        self._lock = threading.Lock()

    def submit(
        self,
        lane: PreparationLane,
        function: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        future = self._service.submit(lane, function, *args, **kwargs)
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._completed)
        return future

    def _completed(self, future: Future[Any]) -> None:
        with self._lock:
            self._futures.discard(future)

    def cancel_pending(self) -> int:
        with self._lock:
            futures = tuple(self._futures)
        return sum(future.cancel() for future in futures)

    def close(self, *, cancel_pending: bool = False) -> None:
        if cancel_pending:
            self.cancel_pending()


class GlobalPreparationService:
    """Separate latency-sensitive audio work from slow background analysis."""

    def __init__(self, *, critical_workers: int, background_workers: int) -> None:
        if critical_workers < 1 or background_workers < 1:
            raise ValueError("preparation worker counts must be positive")
        self._critical = _DaemonExecutor(
            max_workers=critical_workers,
            thread_name_prefix="audio-prepare",
        )
        self._background = _DaemonExecutor(
            max_workers=background_workers,
            thread_name_prefix="audio-analysis",
        )

    def submit(
        self,
        lane: PreparationLane,
        function: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        executor = self._critical if lane == "critical" else self._background
        return executor.submit(function, *args, **kwargs)

    def close(self, *, cancel_pending: bool = True, wait: bool = True) -> None:
        self._critical.shutdown(wait=wait, cancel_futures=cancel_pending)
        self._background.shutdown(wait=wait, cancel_futures=cancel_pending)


_LOCK = threading.Lock()
_SERVICE: GlobalPreparationService | None = None


CRITICAL_WORKERS_VARIABLE = "RESONFORGE_CRITICAL_PREPARE_WORKERS"
BACKGROUND_WORKERS_VARIABLE = "RESONFORGE_BACKGROUND_PREPARE_WORKERS"


def _default_background_workers() -> int:
    """Width of the slow-analysis lane.

    **One, and unlike the bare `1` it replaces this is now measured.** Six
    background analyses are submitted within a millisecond of each other and
    queue on one thread, so the queue wait is real and tens of seconds long.

    **Widening it drains that queue and buys nothing**, because the analyses
    themselves then run **2.2x slower** -- they compete with the critical
    preparation lane and the model workers for the same cores. Wall clock is
    identical. Arms: `docs/scheduler-fill/README.md`.

    **The queue is not on the critical path**: the four stems that start
    immediately keep the device fed while piano waits, so piano starting sooner
    changes nothing. Exactly the shape `_default_critical_workers` records for
    its own lane, arrived at independently.

    So: one, because widening costs CPU contention for no throughput. The
    override is here so a host whose analyses are slower relative to its GPU can
    be re-priced without re-deriving this experiment.
    """
    override = os.environ.get(BACKGROUND_WORKERS_VARIABLE)
    if override:
        workers = int(override)
        if workers < 1:
            raise ValueError(
                f"{BACKGROUND_WORKERS_VARIABLE} must be positive, got {workers}"
            )
        return workers
    return 1


def _default_critical_workers() -> int:
    """Width of the latency-sensitive preparation lane.

    **Widening this is not a throughput lever, and that is measured.** A wider
    lane does drain the producer queue -- the queue is real -- but the queue is
    only a few percent of a session, so draining it buys about that much and
    nothing more. It is also **not** the cap on ready rows: the 4 was suspected
    because `--song-jobs` throughput peaks at four, and that correspondence was
    a coincidence.

    The ceiling stays core-relative and stays a ceiling: 16 is the widest
    setting anything has been measured at, and the background lane above shows
    what oversubscription costs when it does bite. The override stays because
    the bound is host-dependent -- re-price it on the next host rather than
    re-deriving the experiment. Arms and numbers: `docs/scheduler-fill/`.
    """
    override = os.environ.get(CRITICAL_WORKERS_VARIABLE)
    if override:
        workers = int(override)
        if workers < 1:
            raise ValueError(
                f"{CRITICAL_WORKERS_VARIABLE} must be positive, got {workers}"
            )
        return workers
    return max(1, min(16, os.cpu_count() or 1))


def acquire_preparation_service(
    *, session_id: str | None = None
) -> PreparationLease:
    """Acquire the process singleton used by CLI and future server requests."""
    global _SERVICE
    with _LOCK:
        if _SERVICE is None:
            _SERVICE = GlobalPreparationService(
                critical_workers=_default_critical_workers(),
                background_workers=_default_background_workers(),
            )
        service = _SERVICE
    return PreparationLease(service, session_id or uuid.uuid4().hex[:8])


def shutdown_preparation_service() -> None:
    global _SERVICE
    with _LOCK:
        service = _SERVICE
        _SERVICE = None
    if service is not None:
        service.close(wait=False)


atexit.register(shutdown_preparation_service)
