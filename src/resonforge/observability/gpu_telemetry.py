"""Run-scoped NVIDIA GPU telemetry backed by NVML."""

from __future__ import annotations

import csv
import gzip
import hashlib
import logging
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pynvml
import torch

from .artifacts import RunArtifactRegistry

LOGGER = logging.getLogger(__name__)
_CSV_FIELDS = (
    "index",
    "uuid",
    "name",
    "timestamp",
    "utilization.gpu [%]",
    "utilization.memory [%]",
    "memory.used [MiB]",
    "power.draw [W]",
    "clocks.current.sm [MHz]",
    "clocks.current.memory [MHz]",
)
_NVML_LOCK = threading.Lock()
_NVML_USERS = 0


def _acquire_nvml() -> None:
    global _NVML_USERS
    with _NVML_LOCK:
        if _NVML_USERS == 0:
            pynvml.nvmlInit()
        _NVML_USERS += 1


def _release_nvml() -> None:
    global _NVML_USERS
    with _NVML_LOCK:
        if _NVML_USERS == 0:
            return
        _NVML_USERS -= 1
        if _NVML_USERS == 0:
            pynvml.nvmlShutdown()


def _text(value: str | bytes) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _optional(call, *args: object) -> object | None:
    try:
        return call(*args)
    except pynvml.NVMLError:
        return None


def _logical_gpu_uuid(index: int) -> str | None:
    try:
        properties = torch.cuda.get_device_properties(index)
    except (AssertionError, RuntimeError, ValueError):
        return None
    value = getattr(properties, "uuid", None)
    if value is None:
        return None
    # Torch reports a bare UUID; NVML only resolves the canonical GPU-/MIG-
    # prefixed form and returns "Not Found" for the bare one.
    text = str(value)
    return text if text.startswith(("GPU-", "MIG-")) else f"GPU-{text}"


@dataclass(frozen=True)
class _GpuHandle:
    logical_index: int
    handle: Any
    uuid: str
    name: str


def _resolve_handles(gpu_ids: tuple[int, ...]) -> tuple[_GpuHandle, ...]:
    indices = gpu_ids or tuple(range(torch.cuda.device_count()))
    handles: list[_GpuHandle] = []
    for index in indices:
        uuid = _logical_gpu_uuid(index)
        try:
            handle = (
                pynvml.nvmlDeviceGetHandleByUUID(uuid)
                if uuid
                else pynvml.nvmlDeviceGetHandleByIndex(index)
            )
            nvml_uuid = _text(pynvml.nvmlDeviceGetUUID(handle))
            name = _text(pynvml.nvmlDeviceGetName(handle))
        except pynvml.NVMLError as error:
            LOGGER.warning("GPU telemetry skipped cuda:%d: %s", index, error)
            continue
        handles.append(_GpuHandle(index, handle, nvml_uuid, name))
    return tuple(handles)


@dataclass
class GpuTelemetryMonitor:
    """Own one run-local NVML sampling thread and finalize its artifact."""

    raw_path: Path
    interval_ms: int
    handles: tuple[_GpuHandle, ...]
    stop_event: threading.Event
    thread: threading.Thread

    @classmethod
    def start(
        cls,
        output_dir: Path,
        log_dir: Path,
        run_prefix: str,
        *,
        interval_ms: int,
        gpu_ids: tuple[int, ...],
    ) -> GpuTelemetryMonitor | None:
        del log_dir
        try:
            _acquire_nvml()
            handles = _resolve_handles(gpu_ids)
        except (OSError, pynvml.NVMLError) as error:
            LOGGER.warning("GPU telemetry disabled: NVML is unavailable: %s", error)
            return None
        if not handles:
            _release_nvml()
            LOGGER.warning("GPU telemetry disabled: no requested NVIDIA GPU was found")
            return None
        raw_path = output_dir / f"{run_prefix}-gpu.csv"
        stop_event = threading.Event()
        monitor = cls(
            raw_path=raw_path,
            interval_ms=interval_ms,
            handles=handles,
            stop_event=stop_event,
            thread=threading.Thread(),
        )
        monitor.thread = threading.Thread(
            target=monitor._sample,
            name=f"gpu-telemetry-{run_prefix}",
            daemon=True,
        )
        monitor.thread.start()
        LOGGER.info("GPU telemetry started: %d ms -> %s", interval_ms, raw_path)
        return monitor

    def _sample(self) -> None:
        try:
            with self.raw_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
                writer.writeheader()
                while True:
                    timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
                    for gpu in self.handles:
                        writer.writerow(_sample_gpu(gpu, timestamp))
                    stream.flush()
                    if self.stop_event.wait(self.interval_ms / 1000):
                        break
        except OSError:
            LOGGER.exception("GPU telemetry sampler stopped after an output error")

    def finalize(
        self,
        registry: RunArtifactRegistry | None = None,
    ) -> dict[str, object]:
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            LOGGER.warning("GPU telemetry sampler did not stop within 5 seconds")
        _release_nvml()
        descriptor = summarize_gpu_csv(self.raw_path, self.interval_ms)
        output = self.raw_path.with_suffix(".csv.gz")
        with self.raw_path.open("rb") as source, gzip.open(
            output,
            "wb",
            compresslevel=9,
        ) as target:
            shutil.copyfileobj(source, target)
        self.raw_path.unlink()
        artifact = (
            registry.describe(
                output,
                kind="gpu_telemetry",
                encoding="gzip+csv",
                schema_version=1,
            )
            if registry is not None
            else {
                "path": output.name,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "encoding": "gzip+csv",
            }
        )
        descriptor.update(artifact)
        LOGGER.info(
            "GPU telemetry finalized: %d samples -> %s",
            descriptor["samples"],
            output,
        )
        return descriptor


def _sample_gpu(gpu: _GpuHandle, timestamp: str) -> dict[str, object]:
    utilization = _optional(pynvml.nvmlDeviceGetUtilizationRates, gpu.handle)
    memory = _optional(pynvml.nvmlDeviceGetMemoryInfo, gpu.handle)
    power_mw = _optional(pynvml.nvmlDeviceGetPowerUsage, gpu.handle)
    sm_clock = _optional(
        pynvml.nvmlDeviceGetClockInfo,
        gpu.handle,
        pynvml.NVML_CLOCK_SM,
    )
    memory_clock = _optional(
        pynvml.nvmlDeviceGetClockInfo,
        gpu.handle,
        pynvml.NVML_CLOCK_MEM,
    )
    return {
        "index": gpu.logical_index,
        "uuid": gpu.uuid,
        "name": gpu.name,
        "timestamp": timestamp,
        "utilization.gpu [%]": (
            "" if utilization is None else f"{utilization.gpu} %"
        ),
        "utilization.memory [%]": (
            "" if utilization is None else f"{utilization.memory} %"
        ),
        "memory.used [MiB]": (
            "" if memory is None else f"{memory.used / 1024**2:.3f} MiB"
        ),
        "power.draw [W]": (
            "" if power_mw is None else f"{float(power_mw) / 1000:.3f} W"
        ),
        "clocks.current.sm [MHz]": (
            "" if sm_clock is None else f"{sm_clock} MHz"
        ),
        "clocks.current.memory [MHz]": (
            "" if memory_clock is None else f"{memory_clock} MHz"
        ),
    }


def _number(value: str) -> float | None:
    token = value.strip().split(maxsplit=1)[0]
    try:
        return float(token)
    except ValueError:
        return None


def summarize_gpu_csv(path: Path, interval_ms: int) -> dict[str, object]:
    """Return compact whole-run and active-GPU summaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    utilization = [
        value
        for row in rows
        if (value := _number(row.get("utilization.gpu [%]", ""))) is not None
    ]
    active = [value for value in utilization if value > 0]
    memory = [
        value
        for row in rows
        if (value := _number(row.get("memory.used [MiB]", ""))) is not None
    ]
    power = [
        value
        for row in rows
        if (value := _number(row.get("power.draw [W]", ""))) is not None
    ]

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    return {
        "interval_ms": interval_ms,
        "samples": len(rows),
        "gpu_mean_percent": mean(utilization),
        "gpu_active_mean_percent": mean(active),
        "peak_memory_mib": max(memory) if memory else None,
        "mean_power_w": mean(power),
        "peak_power_w": max(power) if power else None,
    }
