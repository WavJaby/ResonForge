"""Typed job failures at the service boundary (D1-5, S2).

Two things were lost when a job failed. The traceback: `_run_job` stored
`f"{type(e).__name__}: {e}"`, which for an unattended deployment is the only
record anyone ever sees. And the *kind*: a caller could not tell "you asked for
something impossible" from "this device is unusable now", so a retry loop had no
way to behave differently, and both looked like the same string.

Full per-job isolation is server-round work (S2 defers D5-3). Classifying is
what makes the deferral safe: the blast radius is bounded by deployment shape,
and the caller can at least see which radius it hit.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from enum import StrEnum


class FailureKind(StrEnum):
    """What a caller should do about it — the only reason to distinguish."""

    # The request cannot succeed as written. Retrying is pointless; fix it.
    INVALID_REQUEST = "invalid_request"
    # This job could not be given what it needed. Another job, or this one
    # later, may well succeed — the device is fine.
    RESOURCE = "resource"
    # The device is unusable until the process restarts. Every job routed to it
    # will fail the same way, so retrying in place makes it worse, not better.
    DEVICE_POISONED = "device_poisoned"
    # The operator asked for it.
    CANCELLED = "cancelled"
    # Unclassified. Deliberately not folded into RESOURCE: an unknown failure
    # that a retry loop treats as transient is an unknown failure that repeats.
    INTERNAL = "internal"


@dataclass(frozen=True)
class JobFailure:
    """One job's failure, with enough detail to diagnose it after the fact."""

    kind: FailureKind
    # Empty when the failure did not arrive as an exception at this boundary —
    # the pipeline catches its own and returns a message. Naming a type that was
    # never raised would make the record read like an exception it is not.
    type_name: str
    message: str
    traceback_text: str

    def __str__(self) -> str:
        return f"{self.type_name}: {self.message}" if self.type_name else self.message

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": str(self.kind),
            "type": self.type_name,
            "message": self.message,
            "traceback": self.traceback_text,
        }


def _kind_of(error: BaseException) -> FailureKind:
    # Imported here: the classification is the one place the application layer
    # needs to name scheduler and transport types, and importing them at module
    # scope would drag torch into every caller that only wanted the enum.
    from resonforge.application.request import ConfigurationError
    from resonforge.runtime.process_runner import Cancelled

    if isinstance(error, Cancelled):
        return FailureKind.CANCELLED
    if isinstance(error, ConfigurationError | FileNotFoundError):
        return FailureKind.INVALID_REQUEST

    try:
        from resonforge.scheduler.device.block_pool import PoolCapacityError
        from resonforge.scheduler.global_service import (
            GlobalDeviceOwnershipError,
            GlobalSchedulerPoisonedError,
        )
    except ImportError:  # scheduler deps absent in a transport-only environment
        return FailureKind.INTERNAL

    if isinstance(error, GlobalSchedulerPoisonedError):
        return FailureKind.DEVICE_POISONED
    if isinstance(error, PoolCapacityError | GlobalDeviceOwnershipError):
        # A device too small to hold one row, or one already owned: this job
        # cannot proceed, but the device is not damaged.
        return FailureKind.RESOURCE
    return FailureKind.INTERNAL


def classify_failure(error: BaseException) -> JobFailure:
    """Capture one exception as a typed, fully recorded job failure."""
    return JobFailure(
        kind=_kind_of(error),
        type_name=type(error).__name__,
        message=str(error),
        traceback_text="".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    )
