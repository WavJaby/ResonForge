"""What one logical row costs this backend in device bytes, beyond its pages.

Since A3 an arena reserves no KV page at admission -- it draws one per layer
when a row crosses a block boundary, and gives them all back when the row ends
-- so a row's *device-byte* cost is only the state that is not KV. Measured on
the shipped `small` model, that is:

* one int64 cursor in the model state per physical row: **32 bytes for a batch
  of four, identical at KV lengths 128, 1,024 and 4,096**
* five int64 scalars and a one-byte sampling mask the session owns per logical
  row
* a one-byte forbidden-token mask per vocabulary entry

None of it varies with KV length, so none of it needs measuring.

**What this module used to be, and why it stopped being necessary.** It
allocated a state at two probe lengths, fit a line through them, and memoised
the coefficients per (model, batch). The slope it was resolving was *entirely*
the KV: measured, the KV was **100.3%** of the row it priced, and with the KV
removed the fit raises "model state did not grow with KV length" because there
is nothing left to fit. A probe, a linear solve, a weak-keyed memo and two
throwaway allocations, to price 8 bytes.

The KV is bounded by the page pool now, in rows rather than bytes --
`_kv_row_capacity`. Two resources, two conservation laws.
"""

from __future__ import annotations

from torch import nn

from resonforge.scheduler.device.block_pool import ArenaCost

# The shared decode cursor `bind_continuous_decode_metadata` allocates, one
# int64 per *physical* row -- so classifier-free guidance doubles it, which is
# why the batch width appears below.
_ROW_CURSOR_BYTES = 8

# Per-row tensors outside the model state whose size is known without any
# prepared row: `sequence`, `sampling_seeds`, `sampling_positions`,
# `temporal_floors` and `temporal_checkpoint_floors` are one int64 each, beside
# a one-byte `sampling_mask`.
_FIXED_ROW_SCALARS = 5 * 8 + 1


def measure_arena_cost(
    lm: nn.Module,
    *,
    kv_capacity: int,
    cfg_enabled: bool,
    minimum_width: int = 1,
) -> ArenaCost:
    """Device bytes one logical row occupies for as long as it is resident.

    `kv_capacity` no longer changes the answer, and the parameter stays because
    it is the declared shape of the scheduler's `arena_cost` callback. The
    length-dependent part of a row is its KV, and that is pages now -- priced
    against the page pool, not against this budget. Validated rather than
    silently ignored: a caller passing a nonsensical capacity is still asking
    for something it will not get.
    """
    if kv_capacity < 1:
        raise ValueError("kv capacity must be positive")
    cfg_width = 2 if cfg_enabled else 1
    return ArenaCost(
        row_bytes=(
            _ROW_CURSOR_BYTES * cfg_width
            + _FIXED_ROW_SCALARS
            + int(getattr(lm, "card", 0))
        ),
        minimum_width=minimum_width,
    )
