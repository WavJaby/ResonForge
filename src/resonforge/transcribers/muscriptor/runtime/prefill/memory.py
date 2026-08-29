"""Every device allocation prefill makes, and what it costs before it is made.

The direction this exists to reverse: the rest of the system sizes itself from a READING (`measured_transient_bytes` is the high-water of what already happened, consulted to plan what happens next).
A ledger can't be built on that -- the answer arrives after the question.

So this module states prefill's demand BEFORE allocating, from shapes the caller already holds, and is the only place prefill takes device memory.
When a central ledger exists the sequence becomes `demand_bytes() -> ask -> allocate()` with no reading in it, and this module changes only by routing `allocate` through the ledger.

Computable vs calibrated:
  * the STATE is exact arithmetic in both rows and length -- no calibration, no reading. (The only excess ever measured was allocator segment rounding, which doesn't scale and vanishes at another length.)
  * the FORWARD is not, and splits in two: a fixed working set moving with neither rows nor length, and a per-row cost proportional to length. Both are the coefficients below.

! Fit against rows at ONE length and the answer is ~4x too small -- that form reads as a constant per row and isn't one, and it under-reserves exactly at production sequence length.
  Two lengths are the minimum: one can't separate a constant from a rate, and three tidy points at a single length produced a confident 4x error. Points and fits: docs/vram-accounting.md.

! Every literal below is TEMPORARY. The two coefficients are one host, one model, one backend, one sequence length.
  Intended replacement: a calibration step measuring them per `(model, backend)` at load time. Until then they're overridable so a disagreeing device isn't silently mispriced.

The peak these describe is the WHOLE ADMISSION -- prefill's forward, the first token, the hot admit -- because that's the peak a reserve has to cover.
If prefill and decode ever become separately budgeted, re-measure with the boundary in the right place; the name says `ADMISSION` not `PREFILL` so that can't be forgotten.
"""

from __future__ import annotations

import torch
from muscriptor.modules.paged_kv import (
    blocks_for_tokens,
    mapped_whole,
    padded_length_for,
    page_bytes_for,
)
from muscriptor.modules.streaming import ModelState, init_states, release_state_blocks

#: Bytes the admission path peaks at regardless of rows or length -- the forward's fixed working set.
#: CALIBRATED, not derived: 202.3 MiB at one length against 201.3 at twice it.
ADMISSION_FIXED_BYTES = 202 * 1024**2

#: Bytes one row adds to that peak per 1k positions of its prefill, ABOVE the state the row keeps.
#: CALIBRATED: 66.5 and 65.0 MiB at the two measured lengths. The lower is taken -- the higher came from the shorter sequence, where the fixed part is a larger share of a smaller total and the split is least determined.
#: Per THOUSAND POSITIONS, not per row: a flat per-row figure fitted at one length underestimates ~4x at the production `kv_capacity` of 2048.
ADMISSION_PER_ROW_PER_1K_POSITIONS_BYTES = 65 * 1024**2

_fixed_bytes = ADMISSION_FIXED_BYTES
_per_row_1k_bytes = ADMISSION_PER_ROW_PER_1K_POSITIONS_BYTES


def set_admission_coefficients(
    *, fixed_bytes: int, per_row_per_1k_positions_bytes: int
) -> None:
    """Re-price the calibrated half for a device or model that disagrees.

    Here because the alternative is a host silently mispricing a peak it cannot
    see, which on Windows is a 4.9x slowdown with no error and on Linux is an
    out-of-memory kill.
    """
    global _fixed_bytes, _per_row_1k_bytes
    if fixed_bytes < 0 or per_row_per_1k_positions_bytes < 0:
        raise ValueError("admission coefficients cannot be negative")
    _fixed_bytes = int(fixed_bytes)
    _per_row_1k_bytes = int(per_row_per_1k_positions_bytes)


def state_bytes(lm, *, rows: int, sequence_length: int) -> int:
    """Bytes the prefill state itself occupies. Exact, and verified exact.

    Same arithmetic the KV supply is sized from (`padded_length_for` then `blocks_for_tokens`, once per layer), so a prefill state and the pool that funds it can never be sized from two different answers.
    """
    if rows < 1 or sequence_length < 1:
        raise ValueError("a prefill state needs at least one row and one position")
    attention = lm.transformer.layers[0].self_attn
    dtype = getattr(attention.attention_backend, "cache_dtype", None) or (
        attention.in_proj_weight.dtype
    )
    per_row_blocks = blocks_for_tokens(padded_length_for(int(sequence_length)))
    page = page_bytes_for(
        num_heads=attention.num_heads,
        head_dim=attention.dim_per_head,
        dtype=dtype,
    )
    return per_row_blocks * len(lm.transformer.layers) * page * int(rows)


def demand_bytes(lm, *, rows: int, sequence_length: int) -> int:
    """Peak device bytes this admission will reach, before any of it is taken.

    The question a ledger has to be asked, in the only order that works -- the caller knows `rows` and `sequence_length` before it allocates anything.
    """
    return (
        state_bytes(lm, rows=rows, sequence_length=sequence_length)
        + _fixed_bytes
        + _per_row_1k_bytes * int(rows) * int(sequence_length) // 1000
    )


def allocate_state(lm, *, rows: int, sequence_length: int) -> ModelState:
    """Take the prefill state. The one place prefill allocates device memory.

    From the KV page pool, mapped whole, from wherever the supply has pages.
    Used to be `private_allocation()` -- outside the pool entirely -- not because prefill's bytes should be unaccounted, but because the pool lent only PAGED ranges while prefill wrote through `slab`, which needs consecutive blocks.

    Two mechanism gaps stood in for requirements there, both closed. Prefill draws from the pool, so its KV is inside the one accounting that owns KV.
    And prefill addresses through its block table, so what it needs is every block MAPPED before it writes (`mapped_whole`), not every block ADJACENT -- which is what `contiguous_allocation` said, and what let a supply refuse a prefill with thousands of pages free.

    Source and destination of a hot admit are still ranges of the SAME tensor, which is what the page-transfer install needs; that never depended on adjacency, only on one block numbering.

    Raises `KVPagesExhausted` only when the supply is genuinely empty.
    ! that exception is still uncaught between here and the scheduler (docs/HANDOFF.md). What changed is that fragmentation can no longer raise it.
    """
    with mapped_whole():
        return init_states(
            lm, batch_size=int(rows), sequence_length=int(sequence_length)
        )


def release_state(state: ModelState) -> None:
    """Give a prefill state's blocks back and empty it. Paired with `allocate_state`."""
    release_state_blocks(state)
    state.clear()


def observe_admission_peak(device: torch.device | str) -> int:
    """Peak bytes this process has allocated since the last reset, for calibration.

    Deliberately NOT used to size anything. It's the instrument the coefficients above were measured with, kept beside them so re-measuring needs no new code
    -- and kept out of the sizing path so this module can't drift back into planning from a reading.
    """
    return torch.cuda.max_memory_allocated(device)
