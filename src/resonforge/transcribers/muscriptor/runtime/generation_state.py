"""State row mapping for fixed-slot classifier-free generation."""

from __future__ import annotations

import torch
from muscriptor.modules.streaming import ModelState, copy_state_rows
from torch import nn

ConditionTensors = dict[str, tuple[torch.Tensor, torch.Tensor]]


def slot_state_row_indices(
    slot: int,
    width: int,
    *,
    cfg_enabled: bool,
) -> tuple[int, ...]:
    """Which physical rows one logical slot owns, on the host.

    The mapping lives here and `slot_state_rows` builds its tensor from it, so
    the rule has one definition. Callers that need the integers -- and the
    per-layer install loop does, once per layer -- take them from here instead
    of reading them back off the device.
    """
    if width < 1:
        raise ValueError("width must be positive")
    if not 0 <= slot < width:
        raise IndexError("slot is outside the physical batch")
    return (slot, slot + width) if cfg_enabled else (slot,)


def slot_state_rows(
    slot: int,
    width: int,
    *,
    cfg_enabled: bool,
    device: torch.device | str,
) -> torch.Tensor:
    """Map one logical slot to its conditional and optional null rows."""
    rows = slot_state_row_indices(slot, width, cfg_enabled=cfg_enabled)
    return torch.tensor(rows, dtype=torch.long, device=device)


def copy_prepared_slot_state(
    model: nn.Module,
    model_state: ModelState,
    prepared_state: ModelState,
    *,
    slot: int,
    width: int,
    cfg_enabled: bool,
    device: torch.device | str,
    source_slot: int = 0,
    source_width: int = 1,
) -> None:
    """Transplant one prefilled row into one stable logical slot.

    Both sides are "logical slot `n` of a width-`w` batch", so both use
    `slot_state_rows`. The source defaults to a batch-one state; a batched
    prefill passes its own row and width instead, which is what lets a whole
    prefill batch be admitted from the tensor it was produced in rather than
    copied into a private per-row buffer first.
    """
    destination_host_rows = slot_state_row_indices(
        slot, width, cfg_enabled=cfg_enabled
    )
    source_host_rows = slot_state_row_indices(
        source_slot, source_width, cfg_enabled=cfg_enabled
    )
    destination_rows = torch.tensor(
        destination_host_rows, dtype=torch.long, device=device
    )
    source_rows = torch.tensor(source_host_rows, dtype=torch.long, device=device)
    copy_state_rows(
        model,
        model_state,
        prepared_state,
        destination_rows,
        source_rows,
        destination_host_rows=destination_host_rows,
        source_host_rows=source_host_rows,
    )


def allocate_condition_slots(
    prepared: ConditionTensors,
    width: int,
    *,
    cfg_enabled: bool,
) -> ConditionTensors:
    """Allocate fixed-width condition buffers matching batch-one encoding."""
    source_width = 2 if cfg_enabled else 1
    target_width = width * source_width
    result: ConditionTensors = {}
    for name, (condition, mask) in prepared.items():
        if condition.shape[0] != source_width or mask.shape[0] != source_width:
            raise ValueError(f"condition {name!r} has an invalid prepared batch size")
        result[name] = (
            condition.new_zeros((target_width, *condition.shape[1:])),
            mask.new_zeros((target_width, *mask.shape[1:])),
        )
    return result


def copy_prepared_slot_conditions(
    conditions: ConditionTensors,
    prepared: ConditionTensors,
    *,
    slot: int,
    width: int,
    cfg_enabled: bool,
) -> None:
    """Copy conditional/null tensors into one canonical fixed slot."""
    if conditions.keys() != prepared.keys():
        raise ValueError("prepared condition names do not match the fixed batch")
    destination_rows = slot_state_rows(
        slot,
        width,
        cfg_enabled=cfg_enabled,
        device=next(iter(prepared.values()))[0].device,
    )
    source_width = 2 if cfg_enabled else 1
    source_rows = torch.arange(
        source_width,
        dtype=torch.long,
        device=destination_rows.device,
    )
    for name, destination_pair in conditions.items():
        source_pair = prepared[name]
        for destination, source in zip(
            destination_pair, source_pair, strict=True
        ):
            destination.index_copy_(0, destination_rows, source[source_rows])
