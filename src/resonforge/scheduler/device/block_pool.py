"""Device-wide block pool, byte-denominated (P1).

Replaces per-model arena + solved width. No prediction -- ask what fits *now* against a live device read, answer is one comparison.

Invariants:
  * blocks are BYTES not rows. small/medium row cost 2.27x apart -> each lane draws its own count, no cross-model pricing.
  * a declared lane keeps a floor. Subtraction, not a split: sizing lane L ignores bytes declared-but-not-resident lanes need, else whoever opens first takes the card.
  * a bundle claims 0 blocks. Members draw pages inside an arena that already exists => device blocks have exactly one claimant => nothing to hold-and-wait on.

Whole-row granularity, so an arena is still one contiguous allocation and the KV layout is unchanged. Blocks quantise the accounting, not the tensor.

Since A3 an arena does not reserve a row's full length -- pages drawn at block boundaries, returned at row end.
So the guarantee is "system makes progress" (a starved row releases), NOT "every session finishes uninterfered with"; an individual row may be preempted and rebuilt exactly.
Preemption is what removes hold-and-wait; starvation it does not, and starvation is neither bounded nor measured here.

Still owned here: never admit past the point one row's floor can't be funded (`own_floor_blocks`) -- a supply too small for one row is starved, not deadlocked, and preempting everything does not help.
Before/after table + why the previous design guessed: `docs/paging/`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

# Block quantum, and since P2 also one KV page: BLOCK_TOKENS(16) positions of one layer @ heads=8 dim=64 fp16 K+V = 16*2048 = 32 KiB.
# Allocator and addressing MUST agree on the unit, or they're two allocators arguing over the same memory.
# Not derivable here -- bytes/token is model geometry, this pool is model-agnostic. Layer count changes how many blocks a row takes, never how big one is.
BLOCK_BYTES = 32 * 1024

# No device-proportional reserve -- it scales with the card, not with the arena being opened.
# docs/vram-accounting.md 5b-1.


# Seed for widths nothing has decoded at yet.
# The transient can't be computed before allocating (depends which kernels run), and the first width on a fresh process would otherwise be priced against 0 -> CUDA OOM.
# 64 MiB = per-row avg at production width, this host.
# It's an avg of a mostly-fixed cost => wrong at every width but one, so `transient_bytes_for` overwrites it with a real reading per row-count and never interpolates (R6).
# Fits + the widths they disagree at: docs/vram-accounting.md. Override: --decode-transient-per-row-mib
BOOTSTRAP_TRANSIENT_PER_ROW_BYTES = 64 * 1024**2
_bootstrap_transient_per_row_bytes = BOOTSTRAP_TRANSIENT_PER_ROW_BYTES

# No per-row cap on the measured price -- wrong at one end whatever you set it to, and readings are already scoped to the row count they were taken at.
# Counter was 0 over a full run => demolition candidate, not a guard. docs/vram-accounting.md 5b-1.


def set_bootstrap_transient_per_row_bytes(value: int) -> None:
    """Seed every pool created after this call. Zero restores the old behaviour."""
    global _bootstrap_transient_per_row_bytes
    if value < 0:
        raise ValueError("transient per row cannot be negative")
    _bootstrap_transient_per_row_bytes = int(value)


class PoolCapacityError(RuntimeError):
    """This device cannot hold one row of this lane, however empty it is.

    The only terminal capacity outcome left -- everything the old design failed on was a bad guess, this is a fact about the hardware.
    Reported against an *empty* pool so a transient shortage can't be misread as one.
    """


def reserve_bytes(
    *,
    resident_rows: int,
    transient_bytes: int = 0,
    fragmentation_bytes: int = 0,
) -> int:
    """Device bytes held back from the pool, for rows already resident.

    * transient -- how far a decode step goes above steady allocation, measured at the row count it is spent at (`transient_bytes_for`). Never a per-row ratio: R6.
    * fragmentation -- `inactive_split_bytes`: free per the allocator, unusable for a contiguous arena, and the only term here not computable before allocating.

    Tiles with `_effective_cost`, doesn't overlap it: that charges rows *about to be added*, this covers rows *already resident*. The boundary is the argument, so no double-counting.

    No `total_bytes` on purpose -- a card-proportional reserve can't be right on two cards. docs/vram-accounting.md 5b-1.
    """
    if resident_rows < 0:
        raise ValueError("a device cannot hold a negative number of rows")
    return max(0, int(transient_bytes)) + max(0, int(fragmentation_bytes))


@dataclass(frozen=True)
class ArenaCost:
    """What one logical row of one model costs, + the band the lane will use.

    row_bytes -- arena one logical row holds while resident. Measured, never from a profile.
    minimum_width -- residency the lane needs to be useful at all (recovery: 1 row). The floor the pool won't let another lane eat.
    maximum_width -- rows this lane can ever put to work. Backend POLICY, not a prediction: a bootstrap lane one recovery group deep can't use 12 rows however free the card is.

    Without `maximum_width`, "width is whatever fits" = whichever lane opens first takes the device.
    Measured on this workload: recovery opened at t=3.7s and took 11 rows it never used past 1, primary opened at t=15s and got 1.
    The old design spent `budget_shares` + a demand-weighted solve on this; a declared band is the same guarantee with no guessing,
    because the backend states what it will use instead of predicting what will fit.
    """

    row_bytes: int
    minimum_width: int = 1
    maximum_width: int | None = None

    def __post_init__(self) -> None:
        if self.row_bytes < 1:
            raise ValueError("arena row bytes must be positive")
        if self.minimum_width < 1:
            raise ValueError("arena minimum width must be positive")
        if self.maximum_width is not None and self.maximum_width < self.minimum_width:
            raise ValueError("arena maximum width cannot be below its minimum")

    @property
    def floor_bytes(self) -> int:
        return self.row_bytes * self.minimum_width


def blocks_for(byte_count: int, *, block_bytes: int = BLOCK_BYTES) -> int:
    """Blocks that hold `byte_count`, rounding up."""
    if byte_count <= 0:
        return 0
    return -(-int(byte_count) // block_bytes)


@dataclass(frozen=True)
class PoolView:
    """One immutable reading of a device pool.

    Free blocks come from a live device query, not a ledger of what we handed out: an arena that exists is already gone from the device's free memory, so a ledger would double-count or drift.
    Pool state is only what's *promised and not yet taken*.
    """

    device: str
    block_bytes: int = BLOCK_BYTES
    # below the water line, held by nothing right now
    free_blocks: int = 0
    # held by arenas whose sessions are owner-free -> closable on demand
    reclaimable_blocks: int = 0
    # declared lanes with no arena yet, in blocks of minimum residency
    floor_blocks: int = 0
    # off CUDA, or the device refused the query -> nothing sized, nothing refused
    unbounded: bool = False

    def available_blocks(self, *, own_floor_blocks: int = 0) -> int:
        """Blocks one caller may claim without stranding a declared lane.

        own_floor_blocks = what this caller already holds a place for; a lane doesn't reserve against its own floor. Adds to the *caller's* budget, never the device's.

        KV is NOT in this budget. It was, twice -- via `ArenaCost.row_bytes` and credited back via `own_supply_blocks`.
        Since A3 an arena reserves no page at admission, so KV is bounded by the page pool alone.
        Two resources, two conservation laws; this one is bytes.

        NOT clamped at 0, deliberately. Declared floors can exceed what's free (another process took memory after we declared), and clamping turns "less than nothing" into "nothing",
        so `admits(0)` would say yes to a caller asking if its own floors are still covered.
        """
        return (
            self.free_blocks
            + self.reclaimable_blocks
            - self.floor_blocks
            + own_floor_blocks
        )

    def admits(self, blocks: int, *, own_floor_blocks: int = 0) -> bool:
        """The whole admission test. One comparison, no safe-sequence search."""
        if self.unbounded:
            return True
        return blocks <= self.available_blocks(own_floor_blocks=own_floor_blocks)


def demand_width(  # noqa: D417 - `held` is documented as deliberately unread
    *,
    held: int,
    demand_high_water: int,
    minimum_width: int,
    affordable: int,
) -> int:
    """Width a lane should hold: what it has used, bounded by what fits.

    P4's decision, and deliberately NOT a share of the device.
    A share needs a denominator in a currency both lanes are priced in, and they aren't -- a `medium` row is 24 layers of 1024-wide pages vs 14 of 768,
    so "half the rows" is two different amounts of memory.
    Asking each lane what it *used* needs no denominator; the device bound afterwards decides whether it may have it.

    high-water, not instantaneous demand. Ready width is lumpy (mean 5.26 vs peak 13, same lane) -> following it directly resizes on every dip.
    High-water since the last resize = "what this lane needed while holding this width": a measurement, same shape as `observe_transient`,
    and what damps this without the hysteresis DESIGN.md wants measured first.

    `held` is not read: a lane's current width must not influence what it asks for next, or one given too much stays wide because it is wide.
    It's in the signature so callers can't silently pass demand where they meant occupancy -- capacity, occupied, active and ready are four different numbers.
    """
    if minimum_width < 1:
        raise ValueError("a lane needs at least one row")
    return max(minimum_width, min(demand_high_water, affordable))


def width_that_fits(
    cost: ArenaCost,
    available_blocks: int,
    *,
    ceiling: int,
    block_bytes: int = BLOCK_BYTES,
) -> int:
    """The widest arena this lane can open now, at or above its own floor.

    Returns 0 when not even the floor fits -- a refusal to admit, not a narrower admission.
    An arena below `minimum_width` can't do the lane's job and would have to be reclaimed to make room for one that can.
    """
    if cost.maximum_width is not None:
        ceiling = min(ceiling, cost.maximum_width)
    ceiling = max(ceiling, cost.minimum_width)
    width = min(ceiling, max(0, available_blocks) * block_bytes // cost.row_bytes)
    return width if width >= cost.minimum_width else 0


class DeviceBlockPool:
    """Promises outstanding against one device + its declared lanes.

    NOT a ledger of live allocations -- those show up in the device's own free memory, and counting them twice is how the old design sized widths against a water line nothing enforced.
    """

    def __init__(self, device: str, *, block_bytes: int = BLOCK_BYTES) -> None:
        self.device = device
        self.block_bytes = block_bytes
        self._lock = threading.Lock()
        self._lanes: dict[str, ArenaCost] = {}
        self._transient_high_water = 0
        # Priced at the width it's spent at, never divided by rows (R6).
        # Keyed by row count, high-water *within* each count: the transient has a big fixed term and isn't linear in rows, so a ratio fitted at one width is wrong at every other.
        # What the ratio cost: docs/PROJECT_MEMORY.md.
        self._transient_by_rows: dict[int, int] = {}
        # row count the current CUDA peak belongs to. None until first sample -> first width also gets a rebase.
        self._transient_width: int | None = None
        # Does the bootstrap ever bind a width? Histograms can't say -- they show it in force, not whether a width was picked while it was.
        self._width_decisions = 0
        # what the counter above can't say: did bootstrap pricing change the width. See `note_width_outcome`.
        self._width_outcomes = 0
        self._width_outcomes_diverged = 0
        self._width_outcome_worst_blocks = 0
        # instantaneous, not high-water: a closed lane holds no rows, and reserving for rows nobody has is how the old flat bootstrap held a third of the card against nothing.
        self._resident_rows = 0
        self._transient_measurement_started = False
        self._fragmentation_high_water = 0

    # -- declared lanes ---------------------------------------------------
    def declare_lane(self, model: str, cost: ArenaCost) -> None:
        """Record what one co-resident lane costs and what floor it keeps."""
        with self._lock:
            self._lanes[model] = cost

    def lane(self, model: str) -> ArenaCost | None:
        with self._lock:
            return self._lanes.get(model)

    def lanes(self) -> dict[str, ArenaCost]:
        with self._lock:
            return dict(self._lanes)

    def floor_blocks(
        self,
        *,
        resident: frozenset[str] = frozenset(),
        exclude: str | None = None,
    ) -> int:
        """Blocks held for declared lanes that have no arena yet."""
        return sum(
            blocks_for(cost.floor_bytes, block_bytes=self.block_bytes)
            for model, cost in self.lanes().items()
            if model != exclude and model not in resident
        )

    # -- measured reserve inputs -------------------------------------------
    def begin_transient_measurement(self) -> bool:
        """Claim the one-time rebase of this device's transient measurement.

        True exactly once per pool.
        `max_memory_allocated` is a *process* lifetime peak, so the first sample after decode starts still carries model load + arena construction -- one-off allocations the high-water then never lets go of.
        Caller resets the device peak on the True call and samples nothing from it.

        This is the half missing when resetting the torch peak alone was tried and changed nothing: the mark had already latched.
        """
        with self._lock:
            if self._transient_measurement_started:
                return False
            self._transient_measurement_started = True
            return True

    def observe_transient(self, transient_bytes: int) -> int:
        """Record how far above steady allocation one operation reached.

        High-water: the reserve covers the worst step the run will take, not the current one.

        In memory only. The defect this replaces wasn't the measurement -- it was a high-water *persisted across processes*, which never decayed,
        so a host's usable width fell monotonically over its lifetime and the only cure was wiping the DB.
        This one dies with the process.
        """
        with self._lock:
            self._transient_high_water = max(
                self._transient_high_water, max(0, int(transient_bytes))
            )
            return self._transient_high_water

    def begin_width_measurement(self, rows: int) -> bool:
        """Claim a rebase because the resident row count just changed.

        True -> caller resets the device peak and samples NOTHING from this call.
        Without it every measurement after the first is the peak of the widest arena ever opened, so a narrow width reports what a wide one once reached
        -- and from the counters alone that looks identical to the cost being flat.

        Pairs with `begin_transient_measurement`: that rebases once per process, this once per width.
        """
        with self._lock:
            if self._transient_width == int(rows):
                return False
            self._transient_width = int(rows)
            return True

    def observe_transient_at_width(self, transient_bytes: int, rows: int) -> int:
        """Record the transient at the row count it was reached with.

        Replaces a per-row ratio, and that's the point: a ratio gets fitted at whatever width is open and spent at whatever comes next (a reserve is needed before the next width exists),
        and it can't be right at both ends because the transient has a fixed term AND a length term. What the ratio cost: docs/PROJECT_MEMORY.md.

        So: store the reading against its own row count instead of fitting coefficients on a host that can't re-measure per width.
        Price is always measured where it's spent; a width nobody has run falls back to the bootstrap, never to an extrapolation.

        High-water *within* each row count, same reason as the absolute one.
        """
        if rows < 1:
            return 0
        with self._lock:
            width = int(rows)
            observed = max(0, int(transient_bytes))
            current = self._transient_by_rows.get(width, 0)
            self._transient_by_rows[width] = max(current, observed)
            return self._transient_by_rows[width]

    def transient_bytes_for(self, rows: int) -> int:
        """What a decode step at this width has been seen to reach.

        The measurement for exactly this row count, else the bootstrap -- which is every width until a step runs at it, including the one an arena is about to open at.

        No interpolation, no extrapolation. Both were available and both are how this class of defect keeps arriving: a number derived for one width, charged at another.
        """
        if rows < 1:
            return 0
        with self._lock:
            measured = self._transient_by_rows.get(int(rows))
            if measured:
                return measured
        return int(rows) * _bootstrap_transient_per_row_bytes

    def note_width_decision(self, rows: int) -> bool:
        """Record that a width was just chosen against the price for `rows`. True if measured.

        Separate from `transient_bytes_for` on purpose: the price is read constantly (every state observation asks) while a width is chosen a handful of times a run,
        and only those few reads bind anything. Its count is what says `width_outcomes` fired at all.

        A `width_decisions_bootstrapped` counter stood beside it and is deleted: it was provably equal to `width_decisions` (8832 of 8832 over 32 runs),
        because a width is always chosen before anything has decoded at that row count. One quantity, two names -- and the name that could never move is the one that was read as "the bootstrap is harmless".
        """
        with self._lock:
            self._width_decisions += 1
            return int(rows) in self._transient_by_rows

    def extrapolated_transient_bytes(self, rows: int) -> int | None:
        """What the measurements say a width should cost, None if none do.

        The counterfactual `transient_bytes_for` can't give: a width decision prices a row count nothing has decoded at yet, so its real answer is always the bootstrap.
        Anchored on a measured total + moved by the measured slope -- never a total / rows (R6).

        Only meaningful beside the bootstrap. The pair says how wrong the bootstrap is; `note_width_outcome` says whether that mattered.
        """
        with self._lock:
            points = sorted(self._transient_by_rows.items())
        if not points:
            return None
        anchor_rows, anchor_bytes = points[-1]
        slope = self.marginal_transient_per_row_bytes()
        return max(0, anchor_bytes + slope * (int(rows) - anchor_rows))

    def note_width_outcome(self, bootstrap_blocks: int, measured_blocks: int) -> None:
        """Record whether pricing by the bootstrap changed the width chosen.

        What `note_width_decision` can't answer -- that counts decisions made against the bootstrap, and the answer is *all of them* by construction (a width is chosen before anything decoded at it).
        Only re-running the same division against a measurement-derived price says whether it cost anything, and that's arithmetic, so it's free.
        """
        with self._lock:
            self._width_outcomes += 1
            if int(bootstrap_blocks) != int(measured_blocks):
                self._width_outcomes_diverged += 1
                self._width_outcome_worst_blocks = max(
                    self._width_outcome_worst_blocks,
                    abs(int(measured_blocks) - int(bootstrap_blocks)),
                )

    def bootstrap_pricing_report(self) -> dict[str, object]:
        """Whether the bootstrap ever bound a width, and whether that cost anything.

        `width_decisions` is the responsiveness witness -- read it FIRST. It once read 0 across 47 runs and that was recorded as "harmless"
        when it was really "the probe never fired": the caller reached a different pool instance for the same card (registry keyed by device string, caller re-derived it from a tensor).
        `test_the_registry_key_is_a_string_so_one_producer_must_own_it`.

        `width_outcomes*` is the answer: re-run the same division against a measurement-derived price and see whether the width would have differed.
        Measured 2026-08-29 over 32 runs: 4415 outcomes, **0 diverged** -- the bootstrap has never changed a width on this host.
        Kept anyway, because the 64 MiB constant is one host, one model, one backend; a card where it is wrong has only this to say so.

        Three counters stood here and are deleted: `bootstrap_vs_measured_mib` and `bootstrap_error_percent_worst` (predicted/measured pairs never formed,
        so both were permanently 0 and neither reached a run record) and `width_decisions_bootstrapped` (provably equal to `width_decisions`).
        """
        with self._lock:
            return {
                "width_decisions": self._width_decisions,
                "width_outcomes": self._width_outcomes,
                "width_outcomes_diverged": self._width_outcomes_diverged,
                "width_outcome_worst_blocks": self._width_outcome_worst_blocks,
            }

    def measured_widths(self) -> dict[int, int]:
        """Every row count a transient has actually been measured at."""
        with self._lock:
            return dict(self._transient_by_rows)

    def marginal_transient_per_row_bytes(self) -> int:
        """What ONE MORE ROW adds to a decode step's peak.

        Not `transient_bytes_for`, and collapsing the two is how this defect keeps arriving.
        That = "what does a step at this width reach". This = "how many more rows will these bytes fund", which is marginal by nature and what `_kv_fundable_rows` asks.

        Slope between two measured widths, NOT a total / rows: the total has a fixed term,
        so dividing overstates the marginal most at narrow widths -- exactly where a lane is deciding whether it can grow.
        Both figures: docs/PROJECT_MEMORY.md.

        Bootstrap until two widths have readings; one point has no slope.
        """
        with self._lock:
            points = sorted(self._transient_by_rows.items())
        if len(points) < 2:
            return _bootstrap_transient_per_row_bytes
        (low_rows, low_bytes), (high_rows, high_bytes) = points[0], points[-1]
        span = high_rows - low_rows
        if span < 1 or high_bytes <= low_bytes:
            return _bootstrap_transient_per_row_bytes
        return (high_bytes - low_bytes) // span

    @property
    def resident_rows(self) -> int:
        """Rows this device was last seen holding open, across every lane.

        Recorded by `view` -- the scheduler is the only thing that knows which sessions exist and it already tells the pool every snapshot.
        Kept for the one caller pricing a reserve *outside* a snapshot: `_declare_kv_pool`, which knows its own width and nothing about other lanes.
        """
        with self._lock:
            return self._resident_rows

    def observe_fragmentation(self, fragmentation_bytes: int) -> int:
        """Record free bytes trapped inside segments, unusable for an arena."""
        with self._lock:
            self._fragmentation_high_water = max(
                self._fragmentation_high_water, max(0, int(fragmentation_bytes))
            )
            return self._fragmentation_high_water

    @property
    def measured_transient_bytes(self) -> int:
        with self._lock:
            return self._transient_high_water

    @property
    def measured_fragmentation_bytes(self) -> int:
        with self._lock:
            return self._fragmentation_high_water

    # -- views ------------------------------------------------------------
    def view(
        self,
        *,
        free_bytes: int | None,
        total_bytes: int | None,
        reclaimable_bytes: int = 0,
        resident_lanes: frozenset[str] = frozenset(),
        resident_rows: int = 0,
    ) -> PoolView:
        """Read the pool against one live device snapshot.

        Device-wide, not caller-specific: what one caller may add back to its own budget is `PoolView.admits`'s business,
        so a scheduler snapshot reads the device once and every run compares against the same reading.

        KV pool supply is NOT here, or anywhere in this file.
        Those bytes left `mem_get_info` free when the pool was allocated and serve only arenas of that pool's geometry, so they were a caller quantity credited back into a budget whose row cost had already charged them.
        KV is its own resource with its own bound now; see `_kv_row_capacity`.

        `resident_rows` prices the reserve and comes from the caller for the same reason `resident_lanes` does: which sessions exist is the scheduler's fact, not the pool's.
        `total_bytes` is still taken because `unbounded` is decided from it, not because anything is a fraction of it.
        """
        with self._lock:
            self._resident_rows = max(0, int(resident_rows))
        if free_bytes is None or total_bytes is None:
            return PoolView(
                device=self.device,
                block_bytes=self.block_bytes,
                unbounded=True,
            )
        allocatable = max(
            0,
            int(free_bytes)
            - reserve_bytes(
                resident_rows=resident_rows,
                transient_bytes=self.transient_bytes_for(resident_rows),
                fragmentation_bytes=self.measured_fragmentation_bytes,
            ),
        )
        return PoolView(
            device=self.device,
            block_bytes=self.block_bytes,
            free_blocks=allocatable // self.block_bytes,
            reclaimable_blocks=blocks_for(
                reclaimable_bytes, block_bytes=self.block_bytes
            ),
            floor_blocks=self.floor_blocks(resident=resident_lanes),
        )


_POOLS: dict[str, DeviceBlockPool] = {}
_POOLS_LOCK = threading.Lock()


def device_block_pool(device: str) -> DeviceBlockPool:
    """The one pool for a physical device, shared by every worker lane.

    Process-scoped, not worker-scoped, and that's load-bearing: two stem lanes on one GPU are separate workers, and a per-worker pool would let each promise the same blocks.
    Live free memory can't catch that -- a reservation is precisely the memory nobody has taken *yet*.

    The key is a string and this module can't canonicalise it: resolving `cuda` to an ordinal needs torch, and staying torch-free is what lets the pool be reasoned about and tested without a device.
    So the invariant is enforced instead of assumed -- see `_reject_ambiguous_cuda_key`.
    """
    with _POOLS_LOCK:
        pool = _POOLS.get(device)
        if pool is None:
            _reject_ambiguous_cuda_key(device)
            pool = DeviceBlockPool(device)
            _POOLS[device] = pool
        return pool


def _reject_ambiguous_cuda_key(device: str) -> None:
    """Refuse a bare `cuda` key beside an indexed one, or the reverse.

    `cuda` and `cuda:0` are the same card and different dict keys, so mixing them gives one device TWO pools -- and the split is silent, because each is internally consistent.
    It happened: the scheduler said `cuda` while `_declare_kv_pool` re-derived `str(weight.device)`,
    and the width declaration then priced its reserve against a pool that had never received a measurement (docs/PROJECT_MEMORY.md).

    Whether two ordinals are two cards is knowable from the strings; whether `cuda` *is* `cuda:0` is not, without torch.
    So refuse the ambiguity instead of resolving it, which also names the real defect: two producers of device identity, where there must be one.
    """
    if not device.startswith("cuda"):
        return
    bare = device == "cuda"
    conflicting = [
        existing
        for existing in _POOLS
        if existing.startswith("cuda") and (existing == "cuda") is not bare
    ]
    if conflicting:
        raise ValueError(
            "device block pools mix bare and indexed CUDA keys, so one card "
            f"would get two pools: {device!r} against {sorted(conflicting)!r}. "
            "Thread one device string down from the scheduler rather than "
            "re-deriving it from a tensor."
        )


def reset_device_block_pools() -> None:
    """Drop every pool. For tests and process teardown only."""
    with _POOLS_LOCK:
        _POOLS.clear()
