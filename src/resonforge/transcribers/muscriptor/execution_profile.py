"""MuScriptor execution-profile resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RuntimeName = Literal["torch-eager", "cuda-eager", "cuda-graphs"]
PrefillRuntimeName = Literal["eager", "cuda-graphs"]
ConcreteBackendName = Literal["torch-dense", "cuda-contiguous"]
DEFAULT_GRAPH_BUCKET_SIZE = 64
DEFAULT_GRAPH_CACHE_SIZE = 16
# A runtime names a point on TWO independent axes, enumerated so an illegal point can't be spelled:
#   * how a decode step produces tokens -- replay a captured CUDA graph, or issue the kernels per step;
#   * which attention kernel runs -- our contiguous varlen CUDA kernel, or the dense gather in PyTorch.
# Paging is on NEITHER axis. `CudaContiguousBackend` subclasses `TorchDenseBackend` and inherits `init_state`'s block allocation, `reset_rows` and `copy_rows_from` unchanged
# -- both lines page, both draw the same pool, same row bookkeeping. The CUDA backend overrides `init_state`, `complete_kv` and `attend`.
#
# One constraint, one direction: capture binds cuda-contiguous's physical state layout, so replay requires it. The reverse was never constrained.
# Eager decode over the contiguous kernel is legal and until 2026-08-26 had no name. It is `cuda-eager`: what separates "is this about graphs?" from "is this about the kernel?",
# and the only line a CUDA profiler can be pointed at against the production kernel (kineto fails inside a graph replay).
EXECUTION_LINES: dict[RuntimeName, tuple[bool, ConcreteBackendName]] = {
    # runtime: (decode replays a captured graph, attention kernel)
    "torch-eager": (False, "torch-dense"),
    "cuda-eager": (False, "cuda-contiguous"),
    "cuda-graphs": (True, "cuda-contiguous"),
}
# Prefill is a THIRD axis, independent of both. It carries no kernel choice -- a prefill runs whichever attention backend the decode line selected -- so it has exactly two points and constrains neither other axis.
# Kept separate from `--muscriptor-runtime` because that name means *decode*, and one name spanning both would move two variables at once in every prefill/decode comparison. The mistake `cuda-eager` exists to undo.
# What captured prefill is worth is a property of the HOST, not the code: BAIR 10.30 -> 4.63 ms at width 1, the dev box 2%, because the forward already saturates a compute-6.1 card in fp32.
# That plus devices where capture is unavailable is why this is a flag and not a constant.
PREFILL_LINES: dict[PrefillRuntimeName, bool] = {
    # prefill runtime: replays a captured graph
    "eager": False,
    "cuda-graphs": True,
}


@dataclass(frozen=True)
class MuscriptorExecutionProfile:
    """Concrete, validated execution settings for one model lane."""

    runtime: RuntimeName
    resolved_backend: ConcreteBackendName
    cuda_graphs: bool
    # Defaulted unlike the decode axis, to the line needing no capture. Every production profile comes from `resolve_...` and states both;
    # a hand-built profile is a fixture for scheduler code that doesn't read this axis, and making those state it would dress them up as prefill tests.
    # This is the RESOLVED line, not the requested one -- a device that can't capture resolves to `eager`.
    prefill_runtime: PrefillRuntimeName = "eager"
    prefill_graphs: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime": self.runtime,
            "resolved_backend": self.resolved_backend,
            "cuda_graphs": self.cuda_graphs,
            "prefill_runtime": self.prefill_runtime,
            "prefill_graphs": self.prefill_graphs,
        }


def _device_is_cuda(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


def resolve_muscriptor_execution_profile(
    *,
    runtime: str,
    device: str,
    prefill: str = "cuda-graphs",
) -> MuscriptorExecutionProfile:
    """Resolve one concrete profile before model weights are loaded."""
    if runtime not in EXECUTION_LINES:
        raise ValueError(f"unknown MuScriptor runtime: {runtime}")
    if prefill not in PREFILL_LINES:
        raise ValueError(f"unknown MuScriptor prefill runtime: {prefill}")
    replays, backend = EXECUTION_LINES[runtime]
    if replays and backend != "cuda-contiguous":
        raise ValueError(
            "graph capture binds the contiguous physical layout; "
            f"{backend} cannot replay"
        )
    if backend == "cuda-contiguous" and not _device_is_cuda(device):
        raise ValueError(f"{backend} requires a CUDA device, got {device}")
    prefill_replays = PREFILL_LINES[prefill]
    # A device that can't capture resolves to eager rather than RAISING -- the one place this axis behaves unlike the decode axis.
    # The difference is what each names: a runtime names a KERNEL, and a dense kernel replaying a graph is impossible, so it must raise.
    # The prefill line names only whether to capture; capture is a device capability, and accommodating a device that lacks it is why this flag exists.
    # Not silent: the resolved profile carries `eager` into run metadata beside the requested value in `effective_flags`, and `_graph_prefill_row` counts a `not-cuda` refusal either way.
    # Captured prefill is NOT tied to `cuda-contiguous` the way captured decode is -- a prefill writes its KV through the block table, which both backends inherit, so eager decode + captured prefill is a legal arm.
    if prefill_replays and not _device_is_cuda(device):
        prefill, prefill_replays = "eager", False
    return MuscriptorExecutionProfile(
        runtime=runtime,  # type: ignore[arg-type]
        resolved_backend=backend,
        cuda_graphs=replays,
        prefill_runtime=prefill,  # type: ignore[arg-type]
        prefill_graphs=prefill_replays,
    )
