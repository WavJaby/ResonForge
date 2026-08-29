"""Exact trace comparisons for batch-invariant generation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TensorDifference:
    step: int
    operation: str
    reference_shape: tuple[int, ...]
    candidate_shape: tuple[int, ...]
    max_abs: float | None


class DecodeTraceRecorder:
    """Capture one compact row at ordered Transformer operation boundaries."""

    def __init__(self, lm: nn.Module, *, row: int = 0) -> None:
        self.row = row
        self.operation_order: list[str] = []
        self.tensors: dict[str, list[torch.Tensor]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for index, layer in enumerate(lm.transformer.layers):
            self._register(f"layer.{index}.norm1", layer.norm1)
            self._register_input(
                f"layer.{index}.attention.input", layer.self_attn
            )
            self._register_input(
                f"layer.{index}.attention.core", layer.self_attn.out_proj
            )
            self._register(
                f"layer.{index}.attention.out_proj", layer.self_attn.out_proj
            )
            self._register(f"layer.{index}.attention", layer.self_attn)
            self._register(f"layer.{index}.norm2", layer.norm2)
            self._register(f"layer.{index}.linear1", layer.linear1)
            self._register(f"layer.{index}.linear2", layer.linear2)
            self._register(f"layer.{index}.output", layer)
        if lm.out_norm is not None:
            self._register("output.norm", lm.out_norm)
        self._register("output.logits", lm.linear)

    def _register(self, name: str, module: nn.Module) -> None:
        self.operation_order.append(name)
        self.tensors[name] = []

        def capture(_module, _inputs, output) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} returned a non-tensor output")
            if tensor.shape[0] <= self.row:
                raise IndexError(f"trace row {self.row} is absent from {name}")
            self.tensors[name].append(tensor[self.row].detach().cpu().clone())

        self._handles.append(module.register_forward_hook(capture))

    def _register_input(self, name: str, module: nn.Module) -> None:
        self.operation_order.append(name)
        self.tensors[name] = []

        def capture(_module, inputs) -> None:
            tensor = inputs[0]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} received a non-tensor input")
            if tensor.shape[0] <= self.row:
                raise IndexError(f"trace row {self.row} is absent from {name}")
            self.tensors[name].append(tensor[self.row].detach().cpu().clone())

        self._handles.append(module.register_forward_pre_hook(capture))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> DecodeTraceRecorder:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def first_tensor_difference(
    reference: DecodeTraceRecorder,
    candidate: DecodeTraceRecorder,
) -> TensorDifference | None:
    """Return the earliest decode-step/operation mismatch in two traces."""
    if reference.operation_order != candidate.operation_order:
        raise ValueError("trace operation order differs")
    maximum_steps = max(
        (
            max(len(reference.tensors[name]), len(candidate.tensors[name]))
            for name in reference.operation_order
        ),
        default=0,
    )
    for step in range(maximum_steps):
        for name in reference.operation_order:
            reference_values = reference.tensors[name]
            candidate_values = candidate.tensors[name]
            if step >= len(reference_values) or step >= len(candidate_values):
                reference_shape = (
                    tuple(reference_values[step].shape)
                    if step < len(reference_values)
                    else ()
                )
                candidate_shape = (
                    tuple(candidate_values[step].shape)
                    if step < len(candidate_values)
                    else ()
                )
                return TensorDifference(
                    step,
                    name,
                    reference_shape,
                    candidate_shape,
                    None,
                )
            left = reference_values[step]
            right = candidate_values[step]
            bitwise_equal = left.shape == right.shape and torch.equal(
                left.view(torch.uint8), right.view(torch.uint8)
            )
            if not bitwise_equal:
                max_abs = None
                if left.shape == right.shape:
                    max_abs = float(
                        (left.float() - right.float()).abs().max().item()
                    )
                return TensorDifference(
                    step,
                    name,
                    tuple(left.shape),
                    tuple(right.shape),
                    max_abs,
                )
    return None


def first_token_difference(
    reference: tuple[int, ...],
    candidate: tuple[int, ...],
) -> int | None:
    """Return the first unequal token index, including a length mismatch."""
    for index, (left, right) in enumerate(zip(reference, candidate, strict=False)):
        if left != right:
            return index
    return None if len(reference) == len(candidate) else min(
        len(reference), len(candidate)
    )
