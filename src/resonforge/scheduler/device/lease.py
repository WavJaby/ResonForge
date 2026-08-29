"""Exclusive cross-process ownership for one physical GPU."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO

from resonforge.runtime.paths import temporary_data_root


@dataclass(frozen=True, order=True)
class PhysicalDeviceId:
    """Stable device identity; CUDA ordinals are process-local aliases."""

    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("physical device identity cannot be empty")


@dataclass(frozen=True, order=True)
class CoreIdentity:
    """One ResonForge core process, protected against PID reuse."""

    pid: int
    start_token: str

    def __post_init__(self) -> None:
        if self.pid < 1 or not self.start_token:
            raise ValueError("core identity requires PID and start token")


@dataclass(frozen=True)
class DeviceCoreLeaseSnapshot:
    device: PhysicalDeviceId
    owner: CoreIdentity
    lock_path: Path


class CoreLeaseFailureKind(StrEnum):
    DEVICE_OWNED = "device_owned"
    LOCK_UNAVAILABLE = "lock_unavailable"


@dataclass(frozen=True)
class CoreLeaseFailure:
    kind: CoreLeaseFailureKind
    device: PhysicalDeviceId
    reason: str
    observed_owner: CoreIdentity | None = None


class DeviceCoreLease:
    """Held OS lock; closing or process death releases device ownership."""

    def __init__(self, snapshot: DeviceCoreLeaseSnapshot, stream: BinaryIO) -> None:
        self.snapshot = snapshot
        self._stream = stream
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            _unlock(self._stream)
        finally:
            self._stream.close()

    def __enter__(self) -> DeviceCoreLease:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class SharedDeviceCoreLease:
    """One session reference to the process-owned physical-device lease."""

    def __init__(
        self,
        registry: DeviceCoreLeaseRegistry,
        snapshot: DeviceCoreLeaseSnapshot,
    ) -> None:
        self._registry = registry
        self.snapshot = snapshot
        self._closed = False
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._registry._release(self.snapshot.device)


class DeviceCoreLeaseRegistry:
    """Share one OS device lock among sessions in this core process."""

    def __init__(self, *, lock_root: Path, owner: CoreIdentity) -> None:
        self._lock_root = lock_root
        self._owner = owner
        self._lock = threading.Lock()
        self._leases: dict[PhysicalDeviceId, tuple[DeviceCoreLease, int]] = {}

    def acquire(
        self,
        device: PhysicalDeviceId,
    ) -> SharedDeviceCoreLease | CoreLeaseFailure:
        with self._lock:
            current = self._leases.get(device)
            if current is None:
                acquired = acquire_device_core_lease(
                    device,
                    self._owner,
                    lock_root=self._lock_root,
                )
                if isinstance(acquired, CoreLeaseFailure):
                    return acquired
                lease, references = acquired, 0
            else:
                lease, references = current
            self._leases[device] = (lease, references + 1)
            return SharedDeviceCoreLease(self, lease.snapshot)

    def _release(self, device: PhysicalDeviceId) -> None:
        with self._lock:
            current = self._leases.get(device)
            if current is None:
                raise RuntimeError("device core lease is not owned")
            lease, references = current
            if references == 1:
                del self._leases[device]
                lease.close()
            else:
                self._leases[device] = (lease, references - 1)

    def close(self) -> None:
        with self._lock:
            leases = tuple(lease for lease, _references in self._leases.values())
            self._leases.clear()
        for lease in leases:
            lease.close()


_PROCESS_START_TOKEN = f"{time.time_ns():x}"
# A lock must not outlive the boot, so a temporary directory is the right
# fallback here — unlike the calibration store, which raises rather than degrade
# into a location that is silently emptied.
_DEFAULT_LOCK_ROOT = temporary_data_root() / "ResonForge" / "device-locks"
_GLOBAL_REGISTRY = DeviceCoreLeaseRegistry(
    lock_root=_DEFAULT_LOCK_ROOT,
    owner=CoreIdentity(os.getpid(), _PROCESS_START_TOKEN),
)


def acquire_shared_device_core_lease(
    device: PhysicalDeviceId,
) -> SharedDeviceCoreLease | CoreLeaseFailure:
    return _GLOBAL_REGISTRY.acquire(device)


def shutdown_device_core_leases() -> None:
    _GLOBAL_REGISTRY.close()


def acquire_device_core_lease(
    device: PhysicalDeviceId,
    owner: CoreIdentity,
    *,
    lock_root: Path,
) -> DeviceCoreLease | CoreLeaseFailure:
    """Try once to own a device; contention is typed and never polled."""
    try:
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = lock_root / _lock_name(device)
        owner_path = lock_path.with_suffix(".owner.json")
        stream = lock_path.open("a+b")
    except OSError as error:
        return CoreLeaseFailure(
            CoreLeaseFailureKind.LOCK_UNAVAILABLE,
            device,
            f"device lock storage unavailable: {error}",
        )
    try:
        if not _try_lock(stream):
            observed = _read_owner(owner_path)
            stream.close()
            return CoreLeaseFailure(
                CoreLeaseFailureKind.DEVICE_OWNED,
                device,
                "another ResonForge core owns the physical device",
                observed,
            )
        _write_owner(owner_path, device, owner)
    except OSError as error:
        stream.close()
        return CoreLeaseFailure(
            CoreLeaseFailureKind.LOCK_UNAVAILABLE,
            device,
            f"device lock operation failed: {error}",
        )
    return DeviceCoreLease(
        DeviceCoreLeaseSnapshot(device, owner, lock_path.resolve()),
        stream,
    )


def _lock_name(device: PhysicalDeviceId) -> str:
    digest = hashlib.sha256(device.value.encode("utf-8")).hexdigest()[:24]
    return f"resonforge-device-{digest}.lock"


def _write_owner(
    path: Path,
    device: PhysicalDeviceId,
    owner: CoreIdentity,
) -> None:
    payload = json.dumps(
        {
            "device": device.value,
            "pid": owner.pid,
            "start_token": owner.start_token,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read_owner(path: Path) -> CoreIdentity | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return CoreIdentity(int(payload["pid"]), str(payload["start_token"]))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError):
        return None


if os.name == "nt":
    import msvcrt

    def _try_lock(stream: BinaryIO) -> bool:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.seek(0)
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(stream: BinaryIO) -> None:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(stream: BinaryIO) -> bool:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock(stream: BinaryIO) -> None:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
