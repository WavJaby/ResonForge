"""Run-scoped sink for scheduler outcomes and execution spans."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from resonforge.scheduler.model_types import ModelTaskTiming

from .events import SchedulerEvent, SchedulerEventCollector


@dataclass
class SchedulerObserver:
    """Collect scheduler observations behind one stable interface."""

    trace_spans: bool = False
    # Where durable spans go. Injected so the scheduler never imports the
    # observability package it writes into; `None` keeps spans in memory only.
    durable_sink: Callable[[dict[str, Any]], None] | None = None
    _events: SchedulerEventCollector = field(default_factory=SchedulerEventCollector)
    _timings: list[ModelTaskTiming] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def task_completed(self, timing: ModelTaskTiming) -> None:
        with self._lock:
            self._timings.append(timing)

    @property
    def timings(self) -> tuple[ModelTaskTiming, ...]:
        with self._lock:
            return tuple(self._timings)

    def record(self, event: SchedulerEvent) -> None:
        if self.trace_spans:
            self._events.record(event)
            if self.durable_sink is None:
                return
            durable = dict(event)
            durable["start_wall_ns"] = self._events.wall_ns(
                float(event["start_seconds"])
            )
            durable["end_wall_ns"] = self._events.wall_ns(
                float(event["end_seconds"])
            )
            self.durable_sink(durable)

    def record_span(self, *args: Any, **kwargs: Any) -> None:
        if self.trace_spans:
            action, started_at, finished_at = args[:3]
            event = {
                "action": action,
                "start_seconds": self._events.relative_time(started_at),
                "end_seconds": self._events.relative_time(finished_at),
                "resource": kwargs.pop("resource"),
                "lane": kwargs.pop("lane"),
                "thread": threading.current_thread().name,
                **kwargs,
            }
            self.record(event)

    def relative_time(self, value: float) -> float:
        return self._events.relative_time(value)

    @property
    def events(self) -> tuple[SchedulerEvent, ...]:
        return tuple(self._events.snapshot()) if self.trace_spans else ()
