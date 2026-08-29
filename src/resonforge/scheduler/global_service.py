"""Process-wide MuScriptor model scheduler shared by pipeline sessions."""

from __future__ import annotations

import atexit
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, wait
from typing import Any

from .device.environment import detect_device_environment
from .device.lease import (
    CoreLeaseFailure,
    PhysicalDeviceId,
    SharedDeviceCoreLease,
    acquire_shared_device_core_lease,
    shutdown_device_core_leases,
)
from .model_workers import ModelWorkerRegistry, SchedulerLivenessError
from .observer import SchedulerObserver


class GlobalSchedulerPoisonedError(RuntimeError):
    """The process CUDA context failed and requires a process restart."""


class GlobalDeviceOwnershipError(RuntimeError):
    """Another ResonForge core owns the requested physical device."""


class GlobalSchedulerLease:
    """Session view with cancellation limited to futures owned by that session."""

    def __init__(
        self,
        registry: ModelWorkerRegistry[Any],
        session_id: str,
        *,
        acquire_device_owner: Callable[[str], None] | None = None,
    ) -> None:
        self._registry = registry
        self.session_id = session_id
        self._futures: set[Future[Any]] = set()
        self._devices: set[str] = set()
        self._lock = threading.Lock()
        self._timing_start = len(registry.timings)
        self._acquire_device_owner = acquire_device_owner

    def _track(self, future: Future[Any], *, device: str) -> Future[Any]:
        with self._lock:
            self._futures.add(future)
            self._devices.add(device)
        future.add_done_callback(
            lambda completed, selected_device=device: self._completed(
                completed,
                selected_device,
            )
        )
        return future

    def _completed(self, future: Future[Any], device: str) -> None:
        with self._lock:
            self._futures.discard(future)
        if future.cancelled():
            return
        error = future.exception()
        if error is not None and _is_fatal_cuda_error(error):
            _poison_global_scheduler(self._registry, error, device=device)

    @staticmethod
    def _key_device(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        key = args[0] if args else kwargs["key"]
        return str(key.device)

    def submit(self, *args: Any, **kwargs: Any) -> Future[Any]:
        kwargs.setdefault("pipeline_session_id", self.session_id)
        return self._track(
            self._registry.submit(*args, **kwargs),
            device=self._key_device(args, kwargs),
        )

    def declare_arenas(self, declaration: Any) -> Future[Any]:
        if self._acquire_device_owner is not None:
            self._acquire_device_owner(str(declaration.device))
        future = self._registry.declare_arenas(
            declaration,
            pipeline_session_id=self.session_id,
        )
        return self._track(future, device=str(declaration.device))

    def submit_batch(self, *args: Any, **kwargs: Any) -> Future[Any]:
        kwargs.setdefault("pipeline_session_id", self.session_id)
        return self._track(
            self._registry.submit_batch(*args, **kwargs),
            device=self._key_device(args, kwargs),
        )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run one operation on the worker that owns a lane's model.

        Forwarded because a caller holding a lease has no other way to reach a
        loaded model -- registering an activation-export sink on it, for
        instance -- and reaching for the registry directly would bypass the
        session's cancellation scope.
        """
        return self._registry.run(*args, **kwargs)

    def submit_prepared(self, submission: Any, **kwargs: Any) -> Future[Any]:
        """Queue a submission that already exists.

        `submit_batch` takes the fields; this takes the object. A caller that
        already holds one had to unpack it argument by argument to re-submit,
        which made the field list something three layers kept in step by hand --
        and they did not: `kv_pool` reached the submission and was dropped by
        the re-submission, so the KV pool was wired end to end and never
        declared.
        """
        kwargs.setdefault("pipeline_session_id", self.session_id)
        return self._track(
            self._registry.submit_prepared(submission, **kwargs),
            device=str(submission.key.device),
        )

    def submit_batch_group(self, *args: Any, **kwargs: Any) -> Future[Any]:
        kwargs.setdefault("pipeline_session_id", self.session_id)
        return self._track(
            self._registry.submit_batch_group(*args, **kwargs),
            device=self._key_device(args, kwargs),
        )

    def bind_producer_decision_waits(
        self,
        key: Any,
        handles: tuple[object, ...],
        future: Future[Any],
        **kwargs: Any,
    ) -> Future[Any]:
        return self._track(
            self._registry.bind_producer_decision_waits(
                key,
                handles,
                future,
                **kwargs,
                pipeline_session_id=self.session_id,
            ),
            device=str(key.device),
        )

    def transfer_producer_decision_waits(
        self,
        key: Any,
        handles: tuple[object, ...],
        future: Future[Any],
        **kwargs: Any,
    ) -> Future[Any]:
        return self._track(
            self._registry.transfer_producer_decision_waits(
                key,
                handles,
                future,
                **kwargs,
                pipeline_session_id=self.session_id,
            ),
            device=str(key.device),
        )

    def release_dependency_claim(self, key: Any, token: Any) -> None:
        self._registry.release_dependency_claim(
            key,
            token,
            pipeline_session_id=self.session_id,
        )

    def preload(self, *args: Any, **kwargs: Any) -> Future[Any]:
        kwargs.setdefault("pipeline_session_id", self.session_id)
        return self._track(
            self._registry.preload(*args, **kwargs),
            device=self._key_device(args, kwargs),
        )

    def wait_for_first(
        self,
        futures: set[Future[Any]] | dict[Future[Any], Any],
    ) -> tuple[set[Future[Any]], set[Future[Any]]]:
        """Wait for work or raise the exact terminal error for any used device."""
        work = set(futures)
        with self._lock:
            devices = tuple(self._devices)
        terminals = {
            self._registry.device_terminal_future(device) for device in devices
        }
        done, pending = wait(work | terminals, return_when=FIRST_COMPLETED)
        for terminal in terminals & done:
            terminal.result()
        return done & work, pending & work

    def record_waterfall_span(self, *args: Any, **details: Any) -> None:
        details.setdefault("pipeline_session_id", self.session_id)
        self._registry.record_waterfall_span(*args, **details)

    @property
    def timings(self) -> tuple[Any, ...]:
        return self._registry.timings[self._timing_start :]

    def cancel_pending(self) -> int:
        with self._lock:
            futures = tuple(self._futures)
        return sum(future.cancel() for future in futures)

    def close(self, *, cancel_pending: bool = False) -> None:
        if cancel_pending:
            self.cancel_pending()
        with self._lock:
            devices = tuple(self._devices)
        for device in devices:
            self._registry.release_arena_declaration(
                device,
                pipeline_session_id=self.session_id,
            )


_LOCK = threading.Lock()
_REGISTRY: ModelWorkerRegistry[Any] | None = None
_LOADER: Callable[[Any], Any] | None = None
_POISONED_ERROR: BaseException | None = None
_DEVICE_LEASES: dict[PhysicalDeviceId, SharedDeviceCoreLease] = {}


def _acquire_global_device_ownership(device: str) -> None:
    environment = detect_device_environment(device)
    if environment.device_type != "cuda":
        return
    if not environment.device_uuid:
        raise GlobalDeviceOwnershipError(
            f"physical device UUID unavailable for {device}; ownership denied"
        )
    physical = PhysicalDeviceId(environment.device_uuid)
    with _LOCK:
        if physical in _DEVICE_LEASES:
            return
        acquired = acquire_shared_device_core_lease(physical)
        if isinstance(acquired, CoreLeaseFailure):
            owner = acquired.observed_owner
            owner_text = (
                "unknown"
                if owner is None
                else f"pid={owner.pid} start={owner.start_token}"
            )
            raise GlobalDeviceOwnershipError(
                f"{acquired.reason}: {physical.value} owner={owner_text}"
            )
        _DEVICE_LEASES[physical] = acquired


def global_device_ownership_snapshot() -> tuple[dict[str, object], ...]:
    with _LOCK:
        leases = tuple(_DEVICE_LEASES.values())
    return tuple(
        {
            "device_uuid": lease.snapshot.device.value,
            "owner_pid": lease.snapshot.owner.pid,
            "owner_start_token": lease.snapshot.owner.start_token,
            "lock_path": str(lease.snapshot.lock_path),
        }
        for lease in sorted(
            leases,
            key=lambda item: item.snapshot.device.value,
        )
    )


def _is_fatal_cuda_error(error: BaseException) -> bool:
    if isinstance(error, SchedulerLivenessError):
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "device-side assert triggered",
            "cuda error: an illegal memory access",
            "cuda error: unspecified launch failure",
            "operation failed due to a previous error during capture",
        )
    )


def _poison_global_scheduler(
    registry: ModelWorkerRegistry[Any],
    error: BaseException,
    *,
    device: str,
) -> None:
    global _POISONED_ERROR
    with _LOCK:
        if _REGISTRY is not registry or _POISONED_ERROR is not None:
            return
        _POISONED_ERROR = error
    registry.terminate_device(device, error)
    registry.close(cancel_pending=True, wait=False)


def acquire_global_scheduler(
    loader: Callable[[Any], Any],
    *,
    session_id: str | None = None,
    durable_sink: Callable[[dict[str, Any]], None] | None = None,
) -> GlobalSchedulerLease:
    """Acquire the process singleton; device lanes remain internal to it.

    `durable_sink` receives trace spans. The caller supplies it because the
    scheduler must not import the observability package it writes into; without
    one, spans stay in memory.
    """
    global _LOADER, _REGISTRY
    with _LOCK:
        if _POISONED_ERROR is not None:
            raise GlobalSchedulerPoisonedError(
                "global scheduler CUDA context is poisoned; restart the process"
            ) from _POISONED_ERROR
        if _REGISTRY is None:
            _LOADER = loader
            _REGISTRY = ModelWorkerRegistry(
                loader,
                observer=SchedulerObserver(
                    trace_spans=True,
                    durable_sink=durable_sink,
                ),
            )
        elif _LOADER is not loader:
            raise RuntimeError("global scheduler already uses a different model loader")
        registry = _REGISTRY
    return GlobalSchedulerLease(
        registry,
        session_id or uuid.uuid4().hex[:8],
        acquire_device_owner=_acquire_global_device_ownership,
    )


def shutdown_global_scheduler() -> None:
    global _LOADER, _POISONED_ERROR, _REGISTRY
    with _LOCK:
        registry = _REGISTRY
        poisoned = _POISONED_ERROR is not None
        _REGISTRY = None
        _LOADER = None
        _POISONED_ERROR = None
    if registry is not None:
        registry.close(cancel_pending=True, wait=not poisoned)
    with _LOCK:
        device_leases = tuple(_DEVICE_LEASES.values())
        _DEVICE_LEASES.clear()
    for lease in device_leases:
        lease.close()
    shutdown_device_core_leases()


atexit.register(shutdown_global_scheduler)
