"""One place that decides how a device's free bytes are divided.

Replaces three call sites reading `torch.cuda.mem_get_info` independently, each subtracting its own reserve, with the KV supply sized as `min(what one session needs, what the device has spare)`
-- so whichever lane declared first took everything above the reserve and later declarations became silent no-ops (`resize` refuses while pages are lent).
Reached in production: the RECOVERY lane opened several times wider than the PRIMARY lane that had a full queue behind it, ran nothing, and the primary then deadlocked with zero page rows.
Nothing chose that; it fell out of declaration order.

So the split is by ROLE, declared up front, independent of which model fills it -- `small`+`medium` and `medium`+`large` divide the device identically.

Three reserves, each subtracted exactly once, at DEVICE scope:
  * `process`  -- passed in from `block_pool.reserve_bytes`. Nothing here re-derives it. That answers "how far above its steady allocation does this process go", a different question from the next.
  * `headroom` -- consumers this process doesn't own and can't measure: desktop, browser, the separation subprocess.
                  EVIDENCE MAY NOT LOWER THIS ONE -- the evidence is about us, not them. Without it the device was sized with zero allowance for anything outside the process,
                  and on Windows the overflow spills to system memory at ~4.9x with no error at all.
  * `prefill`  -- the seam, and deliberately ZERO today. Prefill allocates after the pool takes its share, so a pool sized without it is sized wrong;
                  but `reserve_bytes` is a *process-wide* high-water and prefill's admission peak is already inside that reading.
                  Filling it today subtracts the same bytes twice -- measured 2026-08-27: free 8519 MiB, process 3754, headroom 512, prefill 4636, divisible **0**, every lane at width 1.
                  Kept as a parameter because the fix is to make `reserve_bytes` stop being process-wide: decode's transient and prefill's peak computed by their own modules,
                  `measured_transient_bytes` retired, leaving only `inactive_split_bytes` -- the one quantity that genuinely can't be computed before allocating.

! the first wiring attempt gave this module its OWN fragmentation reserve while `reserve_bytes` already carried one, and charged prefill's device-level peak from a per-lane function.
  Same error twice: a quantity subtracted twice is a card reporting itself smaller than it is, and the arena that then fails to open looks like a capacity fact.

Shares are normalised over the families PRESENT: a device with one lane has no second family to protect, and handing that lane two thirds would leave a third of the card unusable
-- the opposite of the standing goal that no row sits empty. The caller says which roles are present; the absent family's share is zero and the two still sum to the divisible remainder.

Does not allocate and does not read the device -- callers pass the reading in. That keeps "what is free" one fact obtained once by whoever owns the device,
rather than three answers from three `mem_get_info` calls that can disagree by whatever happened in between.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Room left for consumers outside this process: the desktop, a browser, the
#: separation subprocess. **Not measurable from inside**, which is exactly why
#: it is a setting and why no measurement is allowed to reduce it.
DEFAULT_DEVICE_HEADROOM_BYTES = 512 * 1024**2

#: How the divisible remainder splits between the two model families. Primary runs the work; recovery exists to be available when an anomaly needs it, and on measured runs holds rows without running them.
#: By role, not model name, so swapping `small`+`medium` for `medium`+`large` divides the device the same way.
#: ! TEMPORARY and known to be arbitrary. No measurement chose 2:1 -- an owner decision standing in until demand-driven sizing exists. It replaces something worse: whichever lane declared first taking everything.
PRIMARY_FAMILY_SHARE = 2 / 3
RECOVERY_FAMILY_SHARE = 1 / 3

#: `bootstrap` sits with recovery because both are the SECOND model -- same weights, so separate shares would divide a supply they can't both hold at once.
PRIMARY_ROLES = frozenset({"primary"})
RECOVERY_ROLES = frozenset({"recovery", "bootstrap", "fresh"})


@dataclass(frozen=True)
class DeviceBudget:
    """What one device's free bytes were divided into, and why."""

    free_bytes: int
    process_reserve_bytes: int
    headroom_bytes: int
    prefill_reserve_bytes: int
    divisible_bytes: int
    primary_bytes: int
    recovery_bytes: int

    @property
    def reserved_bytes(self) -> int:
        return (
            self.process_reserve_bytes
            + self.headroom_bytes
            + self.prefill_reserve_bytes
        )


_headroom = DEFAULT_DEVICE_HEADROOM_BYTES


def set_reserves(*, headroom_bytes: int | None = None) -> None:
    """Re-price the one flat reserve for a device or workload that disagrees.

    One, not two. Fragmentation belongs to `block_pool.reserve_bytes`, which measures it; a second constant here is what made the first wiring attempt subtract the same bytes twice.
    """
    global _headroom
    if headroom_bytes is not None:
        if headroom_bytes < 0:
            raise ValueError("device headroom cannot be negative")
        _headroom = int(headroom_bytes)


#: For the caller that knows a second family is present but not which one. Both claim, the conservative reading.
ALL_ROLES = frozenset(PRIMARY_ROLES | RECOVERY_ROLES)


def family_share(role: str) -> float:
    """The share of the divisible remainder this role's family may claim."""
    if role in PRIMARY_ROLES:
        return PRIMARY_FAMILY_SHARE
    if role in RECOVERY_ROLES:
        return RECOVERY_FAMILY_SHARE
    return _unknown(role)


def divide(
    free_bytes: int,
    *,
    process_reserve_bytes: int = 0,
    prefill_reserve_bytes: int = 0,
    roles_present: frozenset[str] | set[str] = ALL_ROLES,
) -> DeviceBudget:
    """Divide one device reading into reserves and the present families' shares.

    `free_bytes` is passed in rather than read here on purpose (module docstring), and so is `process_reserve_bytes` -- `block_pool.reserve_bytes`, the one place transient and fragmentation are priced.
    `prefill_reserve_bytes` is the seam, and 0 while `reserve_bytes` already covers that peak.

    `roles_present` decides the denominator: a family nobody occupies gets nothing, so a single-lane device is divided whole rather than left a third empty.
    """
    if free_bytes < 0:
        raise ValueError("a device cannot have negative free bytes")
    if process_reserve_bytes < 0:
        raise ValueError("a process reserve cannot be negative")
    if prefill_reserve_bytes < 0:
        raise ValueError("a prefill reserve cannot be negative")
    unknown = set(roles_present) - ALL_ROLES
    if unknown:
        raise ValueError(
            f"unknown lane role {sorted(unknown)[0]!r}; roles are {sorted(ALL_ROLES)}"
        )
    if not roles_present:
        raise ValueError("a device with no lane on it has nothing to divide")
    reserved = (
        int(process_reserve_bytes) + _headroom + int(prefill_reserve_bytes)
    )
    divisible = max(0, int(free_bytes) - reserved)
    has_primary = bool(set(roles_present) & PRIMARY_ROLES)
    has_recovery = bool(set(roles_present) & RECOVERY_ROLES)
    if has_primary and has_recovery:
        primary = int(divisible * PRIMARY_FAMILY_SHARE)
    else:
        primary = divisible if has_primary else 0
    return DeviceBudget(
        free_bytes=int(free_bytes),
        process_reserve_bytes=int(process_reserve_bytes),
        headroom_bytes=_headroom,
        prefill_reserve_bytes=int(prefill_reserve_bytes),
        divisible_bytes=divisible,
        primary_bytes=primary,
        # the remainder, not a second multiplication -- the two shares can then never sum to more than there is, whether one family is present or both
        recovery_bytes=divisible - primary,
    )


def lane_bytes(budget: DeviceBudget, role: str) -> int:
    """What one role's family may claim from an already-divided budget."""
    return (
        budget.primary_bytes if role in PRIMARY_ROLES else budget.recovery_bytes
        if role in RECOVERY_ROLES
        else _unknown(role)
    )


def _unknown(role: str):
    raise ValueError(
        f"unknown lane role {role!r}; roles are "
        f"{sorted(PRIMARY_ROLES | RECOVERY_ROLES)}"
    )
