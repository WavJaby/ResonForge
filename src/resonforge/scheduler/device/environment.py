"""Physical device identity, read without allocating anything.

This is what survived `runtime_profiles.py`. That module existed to key stored
measurements by an environment fingerprint; with the pool there are no stored
measurements, but the *device* still has to be identified for two reasons that
have nothing to do with prediction: the global scheduler takes an exclusive
core lease per physical GPU, and the pool sizes itself against a real device.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceEnvironment:
    """One device's identity and size."""

    device_type: str
    device_name: str
    device_uuid: str
    compute_capability: str
    total_memory_bytes: int


def detect_device_environment(device: str) -> DeviceEnvironment:
    """Read one device identity without allocating model tensors."""
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return DeviceEnvironment(
            device_type=resolved.type,
            device_name=resolved.type,
            device_uuid="",
            compute_capability="",
            total_memory_bytes=0,
        )
    properties = torch.cuda.get_device_properties(resolved)
    major, minor = torch.cuda.get_device_capability(resolved)
    return DeviceEnvironment(
        device_type="cuda",
        device_name=str(properties.name),
        device_uuid=str(getattr(properties, "uuid", "")),
        compute_capability=f"{major}.{minor}",
        total_memory_bytes=int(properties.total_memory),
    )
