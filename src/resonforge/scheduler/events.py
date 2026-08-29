"""Typed, thread-safe scheduler execution event collection."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, TypedDict


class SchedulerSlotState(TypedDict, total=False):
    session_id: int
    slot: int
    sequence: int | None
    job_type: str | None
    paused: bool
    pipeline_session_id: str | None


class SchedulerEvent(TypedDict, total=False):
    action: str
    start_seconds: float
    end_seconds: float
    resource: str
    target_device: str
    lane: str
    thread: str
    session_id: int
    session_label: str
    model: str
    active_before: int
    active_after: int
    occupied_before: int
    occupied_after: int
    capacity: int
    target_capacity: int
    previous_target_capacity: int
    capacity_budget_bytes: int
    capacity_used_bytes: int
    capacity_row_bytes: int
    capacity_pressure: float
    capacity_snapshot_bytes: int
    resident_bytes: int
    displaced_count: int
    graph_static_bytes: int
    owned_total_bytes: int
    batch_size: int
    pending_jobs: int
    prefill_jobs: int
    slots_before: list[SchedulerSlotState]
    slots_after: list[SchedulerSlotState]
    pipeline_session_id: str | None
    pipeline_session_ids: list[str]
    start_wall_ns: int
    end_wall_ns: int
    view_owner_session_id: str


class SchedulerEventCollector:
    """Collect run-relative scheduler events without serialization concerns."""

    def __init__(self) -> None:
        self._started_at = time.perf_counter()
        self._started_wall_ns = time.time_ns()
        self._events: deque[SchedulerEvent] = deque(maxlen=10_000)
        self._lock = threading.Lock()

    def record(self, event: SchedulerEvent) -> None:
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> list[SchedulerEvent]:
        with self._lock:
            return list(self._events)

    def relative_time(self, value: float) -> float:
        return round(value - self._started_at, 6)

    def wall_ns(self, relative_seconds: float) -> int:
        return self._started_wall_ns + round(relative_seconds * 1_000_000_000)

    def record_span(
        self,
        action: str,
        started_at: float,
        finished_at: float,
        *,
        resource: str,
        lane: str,
        **details: Any,
    ) -> None:
        self.record(
            {
                "action": action,
                "start_seconds": self.relative_time(started_at),
                "end_seconds": self.relative_time(finished_at),
                "resource": resource,
                "lane": lane,
                "thread": threading.current_thread().name,
                **details,
            }
        )
