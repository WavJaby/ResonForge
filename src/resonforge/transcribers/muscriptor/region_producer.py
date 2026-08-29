"""CPU-side region producer state machine for deferred model generation."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from resonforge.runtime.process_runner import Cancelled


@dataclass(frozen=True)
class RegionStep:
    payloads: tuple[dict[str, object], ...]
    completed_delta: int
    done: bool
    generation_request: object | None = None
    ready_order: str | None = None


@dataclass(frozen=True)
class ProducerHandoff:
    """Signal that a weight-free producer view is ready for CPU execution."""


@dataclass
class RegionProducerSession:
    """Own one region iterator across CPU prepare and model result resumes."""

    create_events: Callable[[object], Iterator[object]]
    event_to_payload: Callable[[object], dict[str, object] | None]
    events: Iterator[object] | None = None
    completed: int = 0
    owner_thread: int | None = None
    cancelled: Callable[[], bool] = lambda: False
    generation_result: object | None = None
    ready_prefix: str = ""
    generation_requests: int = 0
    trace_collector: object | None = None
    producer_model: object | None = None
    producer_handoff_ready: bool = False

    def close(self) -> None:
        if self.events is None:
            return
        close = getattr(self.events, "close", None)
        if close is not None:
            close()

    def handoff(self, model_obj: object) -> ProducerHandoff | RegionStep:
        """Detach a producer view, falling back to same-lane advance for doubles."""
        producer_view = getattr(type(model_obj), "producer_view", None)
        if callable(producer_view):
            self.producer_model = model_obj.producer_view()
            self.producer_handoff_ready = True
            return ProducerHandoff()
        self.producer_model = model_obj
        return self.advance()

    def advance(self) -> RegionStep:
        """Consume one model result and stop at the next scheduling boundary."""
        from muscriptor.events import ProgressEvent
        from muscriptor.generation_batch import (
            GenerationControlRequest,
            GenerationRequest,
        )
        from muscriptor.recovery_runtime import (
            RecoveryCandidateGroupRequest,
            RecoveryCandidateSpec,
        )

        if self.cancelled():
            self.close()
            raise Cancelled()
        if self.producer_model is None:
            raise RuntimeError("region producer has not received a producer view")

        thread_id = threading.get_ident()
        if self.owner_thread is None:
            self.owner_thread = thread_id
        elif self.owner_thread != thread_id:
            if not self.producer_handoff_ready:
                raise RuntimeError("region session moved before producer handoff")
            self.owner_thread = thread_id
        if self.events is None:
            self.events = self.create_events(self.producer_model)

        payloads: list[dict[str, object]] = []
        terminal_delta = 0
        while True:
            if self.cancelled():
                self.close()
                raise Cancelled()
            try:
                if self.generation_result is None:
                    event = next(self.events)
                else:
                    result = self.generation_result
                    self.generation_result = None
                    event = self.events.send(result)
            except StopIteration:
                return RegionStep(tuple(payloads), terminal_delta, True)
            if isinstance(
                event,
                (
                    GenerationRequest,
                    GenerationControlRequest,
                    RecoveryCandidateSpec,
                    RecoveryCandidateGroupRequest,
                ),
            ):
                ready_order = f"{self.ready_prefix}:{self.generation_requests:06d}"
                self.generation_requests += 1
                if self.trace_collector is not None and isinstance(
                    event, RecoveryCandidateSpec
                ):
                    # The trace fields live on the inner request; the spec is
                    # only the routing envelope.
                    event = dataclasses.replace(
                        event,
                        request=dataclasses.replace(
                            event.request,
                            trace_collector=self.trace_collector,
                            trace_context=(type(event).__name__, ready_order),
                        ),
                    )
                elif self.trace_collector is not None and isinstance(
                    event, GenerationRequest
                ):
                    event = dataclasses.replace(
                        event,
                        trace_collector=self.trace_collector,
                        trace_context=(type(event).__name__, ready_order),
                    )
                return RegionStep(
                    payloads=tuple(payloads),
                    completed_delta=terminal_delta,
                    done=False,
                    generation_request=event,
                    ready_order=ready_order,
                )
            if isinstance(event, ProgressEvent):
                if event.completed <= self.completed:
                    continue
                delta = event.completed - self.completed
                self.completed = event.completed
                if event.completed >= event.total:
                    terminal_delta += delta
                    continue
                return RegionStep(
                    payloads=tuple(payloads),
                    completed_delta=delta,
                    done=False,
                )
            payload = self.event_to_payload(event)
            if payload is not None:
                payloads.append(payload)
