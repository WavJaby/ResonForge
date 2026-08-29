"""Serialized, priority-ordered access to lazily loaded models."""

from __future__ import annotations

import contextlib
import gc
import itertools
import logging
import math
import os
import queue
import sys
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, wait
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Generic, Protocol, TypeVar

import torch

from resonforge.scheduler.device.block_pool import (
    BLOCK_BYTES,
    ArenaCost,
    DeviceBlockPool,
    PoolView,
    blocks_for,
    demand_width,
    device_block_pool,
    reserve_bytes,
    width_that_fits,
)
from resonforge.scheduler.liveness.scheduler import (
    ActionResult,
    ActionResultStatus,
    SchedulerAction,
    SchedulerActionKind,
    SchedulerBundleState,
    SchedulerExternalWaitPhase,
    SchedulerExternalWaitState,
    SchedulerInvariantError,
    SchedulerReclaimAdmissionState,
    SchedulerRunState,
    SchedulerState,
    enabled_actions,
    external_waits_cover_state,
    validate_scheduler_state,
)
from resonforge.scheduler.model_types import (
    ArenaCostProbe,
    ArenaDeclaration,
    ArenaLaneDeclaration,
    BatchSubmission,
    ClaimedBatchResult,
    DependencyClaimToken,
    GenerationJobType,
    KVPoolDeclaration,
    ModelIdentity,
    ModelKey,
    ModelScope,
    ModelTaskTiming,
    PreparedWorkItem,
    TransitiveDependencyClaim,
)
from resonforge.scheduler.observer import SchedulerObserver

_ModelT = TypeVar("_ModelT")
_ResultT = TypeVar("_ResultT")

_LOGGER = logging.getLogger(__name__)
# How often a submitter re-tests the arena-declaration contradiction while waiting.
# Not a deadline -- the wait stays unbounded until resolution finishes, this only bounds how long a real contradiction stays unseen.
_DECLARATION_POLL_SECONDS = 1.0
# `RESONFORGE_EXPERIMENT_SPILL` is gone with the allocation that justified it: freeing an arena slot used to mean *allocating* a second GPU copy of the paused row.
# Since A3 preemption frees the same slot -- pages go back to the supply, the row is rebuilt bit-identical by re-running what wrote its KV.
# Allocates nothing => nothing left to trade.


@dataclass(frozen=True)
class SchedulerWatchdogConfig:
    """Last-resort bound for an event-driven scheduler making no progress."""

    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("scheduler watchdog timeout must be finite and positive")



# `SchedulerRunState` fields that come from a live `mem_get_info` read, not from scheduler-owned state.
# Held out of the version signature: a background process moving free VRAM must not make the scheduler believe one of its own actions happened.
_MEASURED_RUN_FIELDS = frozenset(
    {
        "admission_safe",
        "arena_admission_safe",
        "pool_available_blocks",
    }
)


def _decided_run_signature(run: SchedulerRunState) -> tuple[object, ...]:
    """Everything about one run that only a scheduler action can change."""
    return tuple(
        getattr(run, field.name)
        for field in fields(run)
        if field.name not in _MEASURED_RUN_FIELDS
    )


def _measured_run_signature(run: SchedulerRunState) -> tuple[object, ...]:
    """The live-reading half, identified by run so ordering cannot alias it."""
    return (
        run.run_id,
        *(getattr(run, name) for name in sorted(_MEASURED_RUN_FIELDS)),
    )


class SchedulerLivenessError(RuntimeError):
    """Fail-closed scheduler stop with a durable diagnostic payload."""

    def __init__(self, message: str, report: dict[str, object]) -> None:
        super().__init__(message)
        self.report = report


class DeviceCannotHoldOneRow(SchedulerInvariantError):
    """A lane's WHOLE page supply funds zero rows: total below its own row floor with nothing lent.

    Terminal by arithmetic, not by load: no release can raise free above total, so `_pages_admit`'s
    assumption ("a lane refused here always has a running row whose completion clears the refusal")
    is false and every later dispatch grinds to `enabled_actions == [fail]` with nothing resident --
    deadlock capture C, 2026-08-28. Named here rather than in the pool because the pool reports
    and the banker enforces (`row_floor_blocks` docstring). Delete when supply declaration is gated
    to never open below the floor.
    """


def resident_handle_key(handle: object) -> tuple[int, int, int]:
    """Identify a resident KV row by value, not by object identity.

    The same logical row comes back as a different object across a preemption, restore or bundle rebuild,
    so anything asking "is this the handle I already accounted for" compares these keys. One definition, scheduler and consumers alike.
    """

    return (int(handle.session_id), int(handle.slot), int(handle.generation))


def _kv_row_capacity(device: str, lane: str) -> int:
    """Rows this lane's KV supply can still fund, or -1 when it has no pool.

    The KV half of admission. It used to be a *byte credit* added back into the device budget against bytes `ArenaCost.row_bytes` had already charged
    -- one number in two places, nearly cancelling, so `release-admit` stopped being reachable because a session that reserved nothing fit wherever it was asked.

    Two resources, two conservation laws: device bytes are the block pool's, KV pages the page pool's, and this is that law in the unit admission works in.
    -1 = no pool declared, i.e. a lane whose arenas are private and bounded by the byte budget like anything else.

    Imported here, not at module scope -- this scheduler is generic over the model it drives, and page geometry is the model's business.
    """
    try:
        from muscriptor.modules.paged_kv import (
            declared_pool_bytes,
            declared_pool_free_bytes,
            declared_pool_row_capacity,
            lane_row_bytes,
        )
    except ImportError:
        return -1
    rows = declared_pool_row_capacity(device, lane)
    # 0 with pages lent is ordinary exhaustion: rows end, pages return, the refusal clears.
    # 0 with the supply WHOLE can never clear -- raise it by name instead of reporting a
    # refusal `_pages_admit` will wait on forever (deadlock capture C).
    if rows == 0 and declared_pool_free_bytes(device, lane) == declared_pool_bytes(device, lane):
        raise DeviceCannotHoldOneRow(
            f"lane {lane!r} on {device}: whole KV supply of "
            f"{declared_pool_bytes(device, lane)} bytes cannot fund one row of "
            f"{lane_row_bytes(device, lane)} bytes"
        )
    return rows


def _kv_fundable_rows(
    device: str,
    lane: str,
    available_blocks: int,
    transient_per_row: int = 0,
) -> int:
    """Rows this lane's KV **could** be funded for, not what its pool holds.

    Different questions, and conflating them made the first declaration final (R12).
    A pool is sized from the width the arena opened at; reading that back as the affordable width means the opening width can never be exceeded, so no resize is selected and the pool is never re-declared
    -- a fixed point where the device could fund thousands of rows and the page pool answered 1.

    A width CHOICE is bounded by the device, because every growth path re-declares the supply (`resize_width(on_released=...)` is that moment).
    The current pool bounds an arena BUILD that doesn't re-declare, which `declared_pool_row_capacity` still answers.

    This lane's own pool is added back -- a re-declaration returns those bytes before taking new ones. Other lanes' supplies are not: nothing here may spend them.
    """
    try:
        from muscriptor.modules.paged_kv import declared_pool_bytes
    except ImportError:
        return -1
    row_bytes = _kv_lane_row_bytes(device, lane)
    if row_bytes < 1:
        return -1
    spare = max(0, available_blocks) * BLOCK_BYTES + declared_pool_bytes(device, lane)
    # The transient is charged here too, because this is the bound that
    # actually binds: pages are the scarce resource and a row's decode peak is
    # paid out of the same device.
    return spare // (row_bytes + max(0, transient_per_row))


def _kv_lane_row_bytes(device: str, lane: str) -> int:
    """Bytes one row of this lane's KV needs, or 0 when the lane is unpriced."""
    try:
        from muscriptor.modules.paged_kv import lane_row_bytes
    except ImportError:
        return 0
    return lane_row_bytes(device, lane)


def _resident_rows(device: str, runs: dict) -> int:
    """Rows a device is holding open right now, across every lane on it.

    One definition, two readers that must never disagree: the reserve is looked up by this row count and the reading that fills that entry is recorded against the same one.
    Two spellings would let a price be measured at one width and spent at another -- the shape of every accounting defect in this file.

    Declared width, not active rows: the transient scales with the per-row static state a decode step touches, which exists as soon as the arena does (`observe_transient_at_width`).
    """
    return sum(
        candidate.width
        for candidate in runs.values()
        if candidate.key.device == device and candidate.session is not None
    )


def _kv_pool_shortfall_totals(device: str) -> tuple[int, int]:
    """Running (events, blocks short) for every pool declared on this device.

    Absolute totals, caller records deltas.
    `scheduler_observations` are summed across jobs when metadata is written, so sampling a monotone counter directly multiplies it by however many jobs looked.
    """
    try:
        from muscriptor.modules.paged_kv import canonical_device, pool_shortfalls
    except ImportError:
        return (0, 0)
    # Both sides canonical. A pool is keyed by `canonical_device` ("cuda" -> "cuda:0"); the scheduler's `key.device` is whatever the CLI wrote.
    # Comparing them raw matched nothing on this host and reported it as "no shortfalls" -- the exact failure a counter exists to rule out.
    wanted = canonical_device(device)
    events = blocks = 0
    for key, entry in pool_shortfalls().items():
        if key[0] != wanted:
            continue
        events += entry.events
        blocks += entry.blocks_short
    return (events, blocks)


def _kv_pool_tail_totals(device: str) -> tuple[int, int]:
    """One instant's `(total bytes, releasable tail bytes)` on this device.

    Bytes, not blocks: two lanes on one device are two pools with different page sizes, so blocks summed across them can't be converted afterwards -- and A2e's threshold is in bytes.

    Instantaneous, unlike `_kv_pool_shortfall_totals`: the caller sums each reading and counts samples, so the mean survives `scheduler_observations` being summed across jobs. A minimum would not.
    """
    try:
        from muscriptor.modules.paged_kv import (
            canonical_device,
            pool_tail_occupancy,
        )
    except ImportError:
        return (0, 0)
    wanted = canonical_device(device)
    total = tail = 0
    for key, (pool_total, pool_tail, page_bytes) in pool_tail_occupancy().items():
        if key[0] != wanted:
            continue
        total += pool_total * page_bytes
        tail += pool_tail * page_bytes
    return (total, tail)


def _device_is_cuda(device: str) -> bool:
    return torch.device(device).type == "cuda"


def _is_resident_generation_control(item: object) -> bool:
    """Identify resume/discard controls without requiring a session factory."""
    return getattr(item, "resident_handle", None) is not None and getattr(
        item, "action", None
    ) in {"resume", "discard"}


@dataclass
class _PendingPrefill(Generic[_ModelT]):
    """Condition-complete rows waiting for their independent KV-prefill turn."""

    selected: tuple[tuple[_BatchTask[_ModelT, Any], PreparedWorkItem], ...]
    conditioned: object


@dataclass
class _ProducerDecisionWait:
    """External ownership between checkpoint publication and one control."""

    handle_key: tuple[int, int, int]
    handle: object
    # The task that produced this checkpoint. The key names a *row*, so when two
    # waits collide on one key the key cannot say which two decisions collided;
    # this can, and it is the first question every such failure asks.
    sequence: int
    run_identity: int
    pipeline_session_id: str | None
    observations: Counter[str]
    created_at: float
    phase: SchedulerExternalWaitPhase
    phase_started_at: float
    deadline_at: float
    completion_source: str = "checkpoint_result_delivery"
    wake_source: str = "model_worker_queue"
    future: Future[object] | None = None
    delivery_future: Future[object] | None = None
    binding_token: int | None = None
    future_failure: str | None = None


@dataclass
class _TransitiveClaimOwner:
    """Measured maximum downstream claim attached to one parent handle."""

    claim: TransitiveDependencyClaim
    device: str
    pipeline_session_id: str | None
    run_identity: int = 0
    created_at: float = 0.0
    phase: SchedulerExternalWaitPhase = SchedulerExternalWaitPhase.AWAITING_BIND
    phase_started_at: float = 0.0
    deadline_at: float = float("inf")
    observations: Counter[str] = field(default_factory=Counter)
    handle_key: tuple[int, int, int] | None = None
    future: Future[object] | None = None
    delivery_future: Future[object] | None = None
    binding_token: int | None = None
    future_failure: str | None = None


@dataclass
class _DependencyBundleOwner:
    """Atomic recovery bundle from claim consumption through tuple publication."""

    bundle_id: int
    claim_token: DependencyClaimToken
    parent_handle_key: tuple[int, int, int] | None
    parent_wait_owner: _ProducerDecisionWait | None
    run_identity: int
    pipeline_session_id: str | None
    observations: Counter[str]
    claim: TransitiveDependencyClaim
    future: Future[tuple[object, ...]]
    submissions: tuple[BatchSubmission, ...]
    results: list[object | None]
    resident_handles: list[object | None]
    wait_owners: list[tuple[object, object, object] | None]
    completed: list[bool]
    member_sequences: tuple[int, ...] = ()
    published: bool = False

    @property
    def device(self) -> str:
        """The one device this bundle draws from.

        Read off the claim, not the submissions: a terminal path releases the
        submissions first and then still has to release the bundle's pool
        reservation.
        """
        return self.claim.members[0].key.device


@dataclass(frozen=True)
class _WorkerLane:
    """One serialized execution lane, independent of cached model identity."""

    scope: ModelScope
    device: str
    stem: str | None

    @classmethod
    def from_key(cls, key: ModelKey) -> _WorkerLane:
        return cls(scope=key.scope, device=key.device, stem=key.stem)


@dataclass
class _CachedModel(Generic[_ModelT]):
    """One weight instance and the runtime profile currently bound to it."""

    key: ModelKey
    model: _ModelT
    load_growth_bytes: int = 0
    weight_storage_bytes: int = 0


def _resident_model_bytes(cached: _CachedModel[Any] | None) -> int:
    """Return what a loaded model keeps, not what loading it briefly cost.

    Load growth is the largest of the allocated, reserved, and peak deltas
    around the load, so it carries the staging copy that loading frees again —
    measured locally at roughly twice the weights. Budgets subtract this for
    the whole run, so they must use the storage the model actually holds.
    """
    if cached is None:
        return 0
    return cached.weight_storage_bytes or cached.load_growth_bytes


def _model_weight_storage_bytes(model: object) -> int:
    """Return a conservative exact total for unique parameter/buffer storage."""
    module = getattr(model, "_model", model)
    parameters = getattr(module, "parameters", None)
    buffers = getattr(module, "buffers", None)
    if not callable(parameters) or not callable(buffers):
        return 0
    try:
        parameter_iter = iter(parameters())
        buffer_iter = iter(buffers())
    except TypeError:
        return 0
    storages: dict[tuple[str, int], int] = {}
    for tensor in itertools.chain(parameter_iter, buffer_iter):
        if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
            continue
        try:
            storage = tensor.untyped_storage()
            storage_bytes = int(storage.nbytes())
            storage_key = (str(tensor.device), int(storage.data_ptr()))
        except (RuntimeError, TypeError, ValueError):
            continue
        if storage_bytes > 0:
            storages[storage_key] = max(storages.get(storage_key, 0), storage_bytes)
    return sum(storages.values())


@dataclass
class _Task(Generic[_ModelT, _ResultT]):
    sequence: int
    key: ModelKey
    priority: int
    job_type: GenerationJobType
    submitted_at: float
    operation: Callable[[_ModelT], _ResultT]
    future: Future[_ResultT]
    pipeline_session_id: str | None = None
    record_timing: bool = True
    trace_model_load: bool = False


@dataclass
class _BatchTask(Generic[_ModelT, _ResultT]):
    sequence: int
    key: ModelKey
    priority: int
    job_type: GenerationJobType
    submitted_at: float
    compatibility_key: object
    item: object
    max_batch_size: int
    future: Future[_ResultT]
    condition_compatibility_key: object | None = None
    arena_cost: ArenaCostProbe | None = None
    kv_pool: KVPoolDeclaration | None = None
    minimum_width: int = 1
    maximum_width: int | None = None
    pipeline_session_id: str | None = None
    batch_dimension: int | None = None
    continuous_factory: Callable[..., object] | None = None
    prepare: (
        Callable[[_ModelT, tuple[object, ...]], tuple[PreparedWorkItem, ...]] | None
    ) = None
    continuous_replacement: bool = True
    ready_order: str = ""
    dependency_claim: TransitiveDependencyClaim | None = None
    dependency_bundle_id: int | None = None
    dependency_bundle_index: int | None = None
    inherited_claim_token: DependencyClaimToken | None = None


@dataclass
class _ResumableSession(Protocol):
    """Model-specific session driven one bounded quantum at a time."""

    active_count: int
    occupied_count: int
    available_count: int
    session_id: int
    remaining_decode_token_budget: int
    next_completion_token_budget: int | None

    def admit(self, item: object) -> bool: ...

    def admit_many(self, items: tuple[object, ...]) -> bool: ...

    def run_quantum(
        self,
        max_steps: int,
        completed: Callable[[object, object], None],
        checkpointed: Callable[[object, object], None] | None = None,
    ) -> object: ...

    def resume(self, handle: object) -> object: ...

    def discard(self, handle: object) -> object: ...

    def can_resume(self, handle: object) -> bool: ...

    def preempt_for_admission(self) -> bool: ...

    def can_preempt_for_admission(self) -> bool: ...

    def is_displaced(self, handle: object) -> bool: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class _WidthBound:
    """The widest arena each of the two conservation laws allows on its own."""

    bytes_width: int
    kv_width: int

    @property
    def width(self) -> int:
        return min(self.bytes_width, self.kv_width)


@dataclass
class _RunTelemetry:
    """What the execution path accumulates and the reporting path reads. Never a decision input.

    Split out of `_PersistentRun` on measurement, not taste: of its 63 fields these 27 were read by exactly two method clusters -- execute writes them, observability reads them --
    and by nothing that chooses an action. Everything left on the run participates in a decision.

    Reset AS A WHOLE (`run.stats = _RunTelemetry()`) when a task publishes its stats. It was 27 hand-written clears, which is one forgotten line away from a counter
    that silently never resets and reports a run's totals as the process's.

    Rates rather than totals, and drained rather than read, are the callers' business -- see `ModelTaskTiming`. What this owns is that the set has one name and one lifetime.
    """

    physical_steps: int = 0
    #: Device tokens computed past a row's own stop -- EOS, temporal-floor break, max_gen_len -- inside a quantum. Checkpoint overrun is NOT waste (retained as `pending_tokens`).
    #: Load-bearing for S5 (host-async decode): speculation waste lands here, and the q8 overshoot reading was unconfirmed for lack of exactly this counter.
    wasted_token_rows: int = 0
    active_steps_by_width: dict[int, int] = field(default_factory=dict)
    prefill_batches_by_width: dict[int, int] = field(default_factory=dict)
    condition_batches_by_width: dict[int, int] = field(default_factory=dict)
    first_token_batches_by_width: dict[int, int] = field(default_factory=dict)
    packed_prefill_batches_by_width: dict[int, int] = field(default_factory=dict)
    quantum_steps_by_size: dict[int, int] = field(default_factory=dict)
    scheduler_decision_cpu_seconds: float = 0.0
    scheduler_decision_phase_seconds: dict[str, float] = field(default_factory=dict)
    scheduler_action_wall_seconds: dict[str, float] = field(default_factory=dict)
    scheduler_boundary_gap_seconds: float = 0.0
    scheduler_boundary_gap_count: int = 0
    phase_gpu_ms: dict[str, float] = field(default_factory=dict)
    cuda_graph_captures: int = 0
    cuda_graph_replays: int = 0
    cuda_graph_refusals: int = 0
    cuda_graph_evictions: int = 0
    cuda_graph_variants: dict[str, dict[str, int]] = field(default_factory=dict)
    cuda_graph_cache_entries_peak: int = 0
    cuda_graph_cache_static_bytes_peak: int = 0
    cuda_graph_cache_pool_bytes_peak: int = 0
    #: Gauges, not counters: the prefill graph cache outlives a session, so these are assigned from its running totals rather than accumulated here.
    prefill_graph: dict[str, int] = field(default_factory=dict)
    prefill_graph_refusals: dict[str, int] = field(default_factory=dict)
    scheduler_gauges: dict[str, int] = field(default_factory=dict)
    scheduler_state_histograms: dict[str, Counter[int]] = field(default_factory=dict)
    hot_replacements: int = 0
    borrowed_admissions: int = 0


@dataclass
class _PersistentRun(Generic[_ModelT]):
    """Worker-owned session plus unresolved ready and active task state."""

    key: ModelKey
    priority: int
    compatibility_key: object
    capacity: int
    width: int
    factory: Callable[[_ModelT, tuple[object, ...], int, int], _ResumableSession]
    allow_replacement: bool
    # How this lane prices a row, and the floor it keeps. `None` means the
    # lane is unpriced: no device bounds it and the pool never refuses it.
    arena_cost_probe: ArenaCostProbe | None = None
    # Declares the KV supply this lane's arenas draw from, once the width is
    # resolved. `None` leaves arenas private -- exactly today's behaviour.
    kv_pool_declaration: KVPoolDeclaration | None = None
    minimum_width: int = 1
    maximum_width: int | None = None
    # Widest this lane had work for during its last batch. P4's input: `demand_width` asks for this, not instantaneous ready width, which is lumpy (5.26 mean vs peak 13 on one lane) and would resize on every dip.
    #
    # What a growth attempt was actually funded at when it came back narrower than asked; 0 = no such attempt stands.
    # Caps later targets so the scheduler doesn't re-select a resize the supply already refused.
    # Cleared the moment a resize does move the width -- the device is shared, so a refusal is a reading, not a property of the lane.
    width_funded_ceiling: int = 0
    # Reset at each RESIZE to the queue standing then, and nowhere else.
    # Resetting when the lane goes owner-free sounds equivalent (both are phase boundaries) and isn't: owner-free is when the lane just finished everything, so it reads the trough,
    # and the peak this exists to remember is gone before anything can act on it. Measured (R12).
    demand_high_water: int = 0
    # resolved once at capacity preparation, by asking the loaded model
    arena_cost: ArenaCost | None = None
    # blocks this run's arena occupies once resident
    arena_blocks: int = 0
    # Backend's own ceiling, kept apart from the live width.
    # Solving against `width` instead makes narrowing permanent -- a run that once opened narrow could never widen again, the residual ratchet rebuilt out of a different variable.
    width_ceiling: int = 0
    # set once the lane's row cost is measured. Replaces `capacity_source`, which named which of three prediction paths produced a width; there is one path now and it is measurement.
    capacity_resolved: bool = False
    purposes: set[str] = field(default_factory=set)
    pending: list[_BatchTask[_ModelT, Any]] = field(default_factory=list)
    pending_controls: list[_BatchTask[_ModelT, Any]] = field(default_factory=list)
    in_flight: list[_BatchTask[_ModelT, Any]] = field(default_factory=list)
    prefill_pending: list[_PendingPrefill[_ModelT]] = field(default_factory=list)
    active_by_item: dict[int, tuple[_BatchTask[_ModelT, Any], PreparedWorkItem]] = (
        field(default_factory=dict)
    )
    # (handle, own checkpoint credit bytes, blocks its claim's members owe).
    # The third element is what makes the whole-claim guarantee survive the
    # parent parking: see `_move_task_credit_to_handle`.
    resident_handles_by_key: dict[tuple[int, int, int], object] = field(
        default_factory=dict
    )
    session: _ResumableSession | None = None
    age: int = 0
    stats: _RunTelemetry = field(default_factory=_RunTelemetry)
    scheduler_observations: Counter[str] = field(default_factory=Counter)
    started_at: dict[int, float] = field(default_factory=dict)
    model_loaded: bool = False
    quantum_count: int = 0
    capacity_resident_bytes: int = 0
    capacity_graph_static_bytes: int = 0
    capacity_total_bytes: int = 0
    measured_row_bytes: int = 0

    @property
    def logical_available_count(self) -> int:
        if self.session is None:
            return self.width
        committed_prefill = sum(
            len(batch.selected) for batch in self.prefill_pending
        )
        return max(
            0,
            min(
                self.session.available_count,
                self.width - self.session.occupied_count,
            )
            - committed_prefill,
        )


@dataclass(frozen=True)
class _SchedulerObservation(Generic[_ModelT]):
    state: SchedulerState
    runs_by_id: dict[int, _PersistentRun[_ModelT]]
    legal_actions: tuple[SchedulerAction, ...]
    snapshot_seconds: float
    snapshot_phase_seconds: dict[str, float]
    validation_seconds: float
    legality_seconds: float


class ModelWorkerPool(Generic[_ModelT]):
    """One worker that serializes tasks across a device-aware model cache.

    Models are loaded in the worker thread on first use. A failed load is not
    cached. Lower numeric priorities run first; submission order breaks ties.
    The active task is never preempted.
    """

    def __init__(
        self,
        loader: Callable[[ModelKey], _ModelT],
        *,
        name: str = "model-worker",
        stop_event: threading.Event | None = None,
        observer: SchedulerObserver | None = None,
        execution_lock: threading.RLock | None = None,
        watchdog_config: SchedulerWatchdogConfig | None = None,
        device: str | None = None,
        device_terminal: Callable[[str, BaseException], None] | None = None,
    ) -> None:
        self._loader = loader
        self._stop_event = stop_event
        self._queue: queue.PriorityQueue[tuple[float, int, object]] = (
            queue.PriorityQueue()
        )
        self._sequence = itertools.count()
        self._models: dict[ModelIdentity, _CachedModel[_ModelT]] = {}
        # Last observed KV-pool shortfall totals per device, so a monotone
        # counter is reported as a delta rather than re-summed each sample.
        self._kv_shortfall_baseline: dict[str, tuple[int, int]] = {}
        self._persistent_runs: dict[
            tuple[ModelKey, object, int, int, bool],
            _PersistentRun[_ModelT],
        ] = {}
        self._observer = observer or SchedulerObserver()
        self._execution_lock = execution_lock or threading.RLock()
        self._watchdog_config = watchdog_config or SchedulerWatchdogConfig()
        self._device = device
        self._device_terminal = device_terminal
        self._admission_burst_turns = 0
        # Devices this worker has touched, so terminal paths can clear their
        # promises without reaching into the process-wide registry.
        self._pool_devices: set[str] = set()
        self._capacity_memory_info: tuple[int, int] | None = None
        self._watchdog_started_at: float | None = None
        # Intents from threads that are not the scheduler worker. They are
        # applied at exactly one point in the turn (`_drain_intents`), which is
        # what lets a decided-version difference across an action have only one
        # possible cause (the single-writer rule).
        self._intent_lock = threading.Lock()
        self._pending_intents: list[tuple[str, Callable[[], None]]] = []
        self._applied_intents = 0
        self._last_intent_name: str | None = None
        self._state_version = 0
        self._measurement_epoch = 0
        self._progress_epoch = 0
        self._last_state_signature: object = ()
        self._last_measured_signature: object = ()
        self._last_snapshot_phase_seconds: dict[str, float] = {}
        self._rejected_actions: set[
            tuple[int, int, int, SchedulerActionKind]
        ] = set()
        self._recent_transitions: list[dict[str, object]] = []
        self._last_decode_quantum_finished_at: float | None = None
        # S5c host-async decode (docs/mixed-batch/PLAN.md): the decode action
        # launches a replay and consumes it on the NEXT decode dispatch, so the
        # scheduler's own work between the two runs beside the device instead
        # of after it. Off by default until the A/B lands; the toggle is
        # temporary and is removed with the measurement either way.
        self._async_decode = os.environ.get("RESONFORGE_ASYNC_DECODE", "") == "1"
        self._liveness_error: SchedulerLivenessError | None = None
        self._requested_terminal_error: BaseException | None = None
        self._producer_waits: dict[tuple[int, int, int], _ProducerDecisionWait] = {}
        self._transitive_claims: dict[int, _TransitiveClaimOwner] = {}
        self._handle_claim_tokens: dict[
            tuple[int, int, int], DependencyClaimToken
        ] = {}
        self._dependency_bundles: dict[int, _DependencyBundleOwner] = {}
        self._dependency_bundle_sequence = itertools.count()
        self._producer_wait_tombstones: dict[tuple[int, int, int], int | None] = {}
        self._claim_tombstones: dict[int, int | None] = {}
        self._producer_binding_sequence = itertools.count()
        self._producer_callback_failure: BaseException | None = None
        self._pending: dict[
            int,
            _Task[_ModelT, Any] | _BatchTask[_ModelT, Any],
        ] = {}
        self._state_lock = threading.Lock()
        self._closed = False
        self._abort_active_on_close = False
        self._stop = object()
        self._wake = object()
        self._worker_stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        self._stop_watcher: threading.Thread | None = None
        if stop_event is not None:
            self._stop_watcher = threading.Thread(
                target=self._watch_stop_event,
                name=f"{name}-stop",
                daemon=True,
            )
            self._stop_watcher.start()

    def submit(
        self,
        key: ModelKey,
        operation: Callable[[_ModelT], _ResultT],
        *,
        priority: int = 10,
        job_type: GenerationJobType = "primary",
        record_timing: bool = True,
        trace_model_load: bool = False,
        pipeline_session_id: str | None = None,
    ) -> Future[_ResultT]:
        """Queue one operation and return its standard-library Future."""
        future: Future[_ResultT] = Future()
        with self._state_lock:
            if self._liveness_error is not None:
                raise self._liveness_error
            if self._closed or (
                self._stop_event is not None and self._stop_event.is_set()
            ):
                raise RuntimeError("model worker pool is closed")
            sequence = next(self._sequence)
            task = _Task(
                sequence=sequence,
                key=key,
                priority=priority,
                job_type=job_type,
                submitted_at=time.perf_counter(),
                operation=operation,
                future=future,
                pipeline_session_id=pipeline_session_id,
                record_timing=record_timing,
                trace_model_load=trace_model_load,
            )
            self._pending[task.sequence] = task
            self._queue.put((priority, sequence, task))
        return future

    def run(
        self,
        key: ModelKey,
        operation: Callable[[_ModelT], _ResultT],
        *,
        priority: int = 10,
    ) -> _ResultT:
        """Run one operation, queueing unless called recursively on this lane."""
        if threading.current_thread() is self._thread:
            return self._run_inline(key, operation, priority=priority)
        return self.submit(key, operation, priority=priority).result()

    def submit_batch(
        self,
        key: ModelKey,
        compatibility_key: object,
        item: object,
        *,
        condition_compatibility_key: object | None = None,
        max_batch_size: int,
        arena_cost: ArenaCostProbe | None = None,
        kv_pool: KVPoolDeclaration | None = None,
        minimum_width: int = 1,
        maximum_width: int | None = None,
        priority: int = 10,
        job_type: GenerationJobType = "primary",
        batch_dimension: int | None = None,
        continuous_factory: Callable[..., object] | None = None,
        prepare: Callable[[_ModelT, tuple[object, ...]], tuple[PreparedWorkItem, ...]]
        | None = None,
        continuous_replacement: bool = True,
        ready_order: str | None = None,
        dependency_claim: TransitiveDependencyClaim | None = None,
        pipeline_session_id: str | None = None,
    ) -> Future[_ResultT]:
        """Queue one row and coalesce adjacent compatible rows on this lane."""
        submission = BatchSubmission(
            key=key,
            compatibility_key=compatibility_key,
            item=item,
            condition_compatibility_key=condition_compatibility_key,
            max_batch_size=max_batch_size,
            arena_cost=arena_cost,
            kv_pool=kv_pool,
            minimum_width=minimum_width,
            maximum_width=maximum_width,
            priority=priority,
            job_type=job_type,
            batch_dimension=batch_dimension,
            continuous_factory=continuous_factory,
            prepare=prepare,
            continuous_replacement=continuous_replacement,
            ready_order=ready_order,
            dependency_claim=dependency_claim,
        )
        return self.submit_prepared(
            submission, pipeline_session_id=pipeline_session_id
        )

    def submit_prepared(
        self,
        submission: BatchSubmission,
        *,
        pipeline_session_id: str | None = None,
    ) -> Future[_ResultT]:
        """Queue a submission that already exists, without restating it.

        Callers holding a `BatchSubmission` used to re-submit by unpacking it argument by argument, so the field list was something two places kept in step by hand. They didn't:
        `kv_pool` was added to the submission and dropped by the re-submission, so the KV pool was wired end to end and never declared -- a full pipeline run measured nothing and looked fine.
        No list here to fall out of step.
        """
        self._validate_batch_submission(submission)
        with self._state_lock:
            self._ensure_submission_open_locked()
            task = self._new_batch_task_locked(
                submission,
                pipeline_session_id=pipeline_session_id,
            )
            self._pending[task.sequence] = task
            self._queue.put((task.priority + 0.5, task.sequence, task))
        return task.future

    @staticmethod
    def _validate_batch_submission(submission: BatchSubmission) -> None:
        # **0 means no forced ceiling**, which is the ordinary case since the
        # two width constants were deleted: what bounds a lane is its own
        # demand and what the device can give it. A positive value is an
        # operator's `--batch-size`, which the pool refuses rather than narrows.
        if submission.max_batch_size < 0:
            raise ValueError("max_batch_size cannot be negative")
        if submission.batch_dimension is not None and submission.batch_dimension < 0:
            raise ValueError("batch_dimension cannot be negative")
        if submission.minimum_width < 1:
            raise ValueError("minimum width must be positive")
        if submission.continuous_factory is None and not _is_resident_generation_control(
            submission.item
        ):
            raise ValueError("generation work requires a persistent session factory")

    def _ensure_submission_open_locked(self) -> None:
        if self._liveness_error is not None:
            raise self._liveness_error
        if self._closed or (
            self._stop_event is not None and self._stop_event.is_set()
        ):
            raise RuntimeError("model worker pool is closed")

    def _new_batch_task_locked(
        self,
        submission: BatchSubmission,
        *,
        pipeline_session_id: str | None,
        dependency_bundle_id: int | None = None,
        dependency_bundle_index: int | None = None,
    ) -> _BatchTask[_ModelT, Any]:
        sequence = next(self._sequence)
        return _BatchTask(
            sequence=sequence,
            key=submission.key,
            priority=submission.priority,
            job_type=submission.job_type,
            submitted_at=time.perf_counter(),
            compatibility_key=submission.compatibility_key,
            condition_compatibility_key=submission.condition_compatibility_key,
            item=submission.item,
            max_batch_size=submission.max_batch_size,
            future=Future(),
            arena_cost=submission.arena_cost,
            kv_pool=submission.kv_pool,
            minimum_width=submission.minimum_width,
            maximum_width=submission.maximum_width,
            pipeline_session_id=pipeline_session_id,
            batch_dimension=submission.batch_dimension,
            continuous_factory=submission.continuous_factory,
            prepare=submission.prepare,
            continuous_replacement=submission.continuous_replacement,
            ready_order=(
                f"{sequence:020d}"
                if submission.ready_order is None
                else submission.ready_order
            ),
            dependency_claim=submission.dependency_claim,
            dependency_bundle_id=dependency_bundle_id,
            dependency_bundle_index=dependency_bundle_index,
        )

    def submit_batch_group(
        self,
        claim_token: DependencyClaimToken,
        parent_resident_handle: object | None,
        members: tuple[BatchSubmission, ...],
        *,
        pipeline_session_id: str | None,
    ) -> Future[tuple[object, ...]]:
        """Atomically consume one parent claim and reserve every bundle member."""
        submission_started_at = time.perf_counter()
        if len(members) < 2:
            raise ValueError("dependency bundle requires multiple members")
        for member in members:
            self._validate_batch_submission(member)
            if member.dependency_claim is not None:
                raise ValueError("dependency bundle members cannot declare parent claims")
            if not bool(getattr(member.item, "requires_dependency_credit", False)):
                raise ValueError("dependency bundle members require dependency credit")
        lanes = {_WorkerLane.from_key(member.key) for member in members}
        if len(lanes) != 1:
            raise ValueError("dependency bundle crossed worker lanes")
        parent_handle_key = (
            None
            if parent_resident_handle is None
            else self._resident_handle_key(parent_resident_handle)
        )
        bundle_future: Future[tuple[object, ...]] = Future()
        # Argument validation above stays on the calling thread. Everything
        # below reads and writes decided state, so the worker applies it --
        # the single-writer rule. The caller already waits on
        # `bundle_future`, which is where a violation now surfaces.
        self._enqueue_intent(
            "submit_batch_group",
            lambda: self._apply_submit_batch_group(
                claim_token,
                parent_handle_key,
                members,
                bundle_future,
                submission_started_at,
                pipeline_session_id=pipeline_session_id,
            ),
        )
        return bundle_future

    def _apply_submit_batch_group(
        self,
        claim_token: DependencyClaimToken,
        parent_handle_key: tuple[int, int, int] | None,
        members: tuple[BatchSubmission, ...],
        bundle_future: Future[tuple[object, ...]],
        submission_started_at: float,
        *,
        pipeline_session_id: str | None,
    ) -> None:
        try:
            self._apply_submit_batch_group_locked(
                claim_token,
                parent_handle_key,
                members,
                bundle_future,
                submission_started_at,
                pipeline_session_id=pipeline_session_id,
            )
        except BaseException as error:
            self._request_producer_terminal(error)
            if not bundle_future.done():
                bundle_future.set_exception(error)

    def _apply_submit_batch_group_locked(
        self,
        claim_token: DependencyClaimToken,
        parent_handle_key: tuple[int, int, int] | None,
        members: tuple[BatchSubmission, ...],
        bundle_future: Future[tuple[object, ...]],
        submission_started_at: float,
        *,
        pipeline_session_id: str | None,
    ) -> None:
        with self._state_lock:
            self._ensure_submission_open_locked()
            owner = self._transitive_claims.get(claim_token.value)
            if owner is None:
                raise SchedulerInvariantError(
                    "dependency bundle has no matching parent claim"
                )
            if owner.pipeline_session_id != pipeline_session_id:
                raise SchedulerInvariantError(
                    "dependency bundle crossed pipeline ownership"
                )
            if (
                owner.future is None
                or not owner.future.done()
                or owner.future_failure is not None
                or owner.delivery_future is None
                or not owner.delivery_future.done()
                or owner.phase is not SchedulerExternalWaitPhase.DELIVERY_READY
            ):
                raise SchedulerInvariantError(
                    "dependency bundle parent decision is not complete"
                )
            if owner.handle_key != parent_handle_key:
                raise SchedulerInvariantError(
                    "dependency bundle parent handle does not match its claim"
                )
            declared = tuple(member.key for member in owner.claim.members)
            submitted = tuple(member.key for member in members)
            if submitted != declared:
                raise SchedulerInvariantError(
                    "dependency bundle does not match the parent claim"
                )
            parent_wait = None
            if parent_handle_key is not None:
                parent_wait = self._producer_waits.get(parent_handle_key)
                if parent_wait is None:
                    raise SchedulerInvariantError(
                        "dependency bundle parent handle has no producer wait"
                    )
                if parent_wait.pipeline_session_id != pipeline_session_id:
                    raise SchedulerInvariantError(
                        "dependency bundle parent wait crossed pipeline ownership"
                    )
                if (
                    parent_wait.future is not owner.future
                    or parent_wait.future_failure is not None
                    or parent_wait.future is None
                    or not parent_wait.future.done()
                    or parent_wait.delivery_future is None
                    or not parent_wait.delivery_future.done()
                    or parent_wait.phase
                    is not SchedulerExternalWaitPhase.DELIVERY_READY
                    or owner.delivery_future is None
                    or not owner.delivery_future.done()
                ):
                    raise SchedulerInvariantError(
                        "dependency bundle parent wait lacks the completed decision"
                    )
            bundle_id = next(self._dependency_bundle_sequence)
            self._dependency_bundles[bundle_id] = _DependencyBundleOwner(
                bundle_id=bundle_id,
                claim_token=claim_token,
                parent_handle_key=parent_handle_key,
                parent_wait_owner=parent_wait,
                run_identity=owner.run_identity,
                pipeline_session_id=pipeline_session_id,
                observations=owner.observations,
                claim=owner.claim,
                future=bundle_future,
                submissions=members,
                results=[None] * len(members),
                resident_handles=[None] * len(members),
                wait_owners=[None] * len(members),
                completed=[False] * len(members),
            )
            del self._transitive_claims[claim_token.value]
            if parent_handle_key is not None:
                del self._producer_waits[parent_handle_key]
                self._handle_claim_tokens.pop(parent_handle_key, None)
                owner.observations["dependency_bundle_parent_wait_consumed"] += 1
            owner.observations["dependency_bundle_admitted"] += 1
            # No wake: this already runs on the worker `_enqueue_intent` woke.
        bundle_future.add_done_callback(
            lambda completed, bundle_id=bundle_id: self._bundle_future_completed(
                bundle_id, completed
            )
        )
        admitted_at = time.perf_counter()
        device = members[0].key.device
        self._observer.record_span(
            "bundle_transaction_admit",
            submission_started_at,
            admitted_at,
            resource="control",
            lane=f"scheduler {device}",
            target_device=device,
            model=members[0].key.model,
            pipeline_session_id=pipeline_session_id,
            bundle_id=bundle_id,
            claim_token=claim_token.value,
            member_count=len(members),
            parent_resident=parent_handle_key is not None,
        )

    def _bundle_future_completed(
        self,
        bundle_id: int,
        future: Future[tuple[object, ...]],
    ) -> None:
        if future.cancelled():
            self._request_producer_terminal(
                RuntimeError(f"dependency bundle {bundle_id} was cancelled")
            )

    def _admit_dependency_bundle(self, bundle_id: int) -> None:
        """Convert one queued claim into all concrete member reservations."""
        with self._state_lock:
            bundle = self._dependency_bundles.get(bundle_id)
            if bundle is None or bundle.member_sequences:
                raise SchedulerInvariantError("bundle admission has no queued owner")
            tasks = tuple(
                self._new_batch_task_locked(
                    submission,
                    pipeline_session_id=bundle.pipeline_session_id,
                    dependency_bundle_id=bundle_id,
                    dependency_bundle_index=index,
                )
                for index, submission in enumerate(bundle.submissions)
            )
            bundle.member_sequences = tuple(task.sequence for task in tasks)
            for task in tasks:
                self._pending[task.sequence] = task
        for task in tasks:
            self._enqueue_persistent(task)
        bundle.observations["dependency_bundle_members_reserved"] += len(tasks)

    def bind_producer_decision_waits(
        self,
        handles: tuple[object, ...],
        future: Future[object],
        *,
        claim_tokens: tuple[DependencyClaimToken, ...] = (),
        pipeline_session_id: str | None,
    ) -> Future[object]:
        """Bind one CPU decision Future to all handle and claim owners."""
        return self._bind_producer_decision_waits(
            handles,
            future,
            claim_tokens=claim_tokens,
            pipeline_session_id=pipeline_session_id,
            transfer=False,
        )

    def transfer_producer_decision_waits(
        self,
        handles: tuple[object, ...],
        future: Future[object],
        *,
        claim_tokens: tuple[DependencyClaimToken, ...] = (),
        pipeline_session_id: str | None,
    ) -> Future[object]:
        """Atomically move completed CPU ownership to its next bounded step."""
        return self._bind_producer_decision_waits(
            handles,
            future,
            claim_tokens=claim_tokens,
            pipeline_session_id=pipeline_session_id,
            transfer=True,
        )

    def _bind_producer_decision_waits(
        self,
        handles: tuple[object, ...],
        future: Future[object],
        *,
        claim_tokens: tuple[DependencyClaimToken, ...],
        pipeline_session_id: str | None,
        transfer: bool,
    ) -> Future[object]:
        handle_keys = tuple(self._resident_handle_key(handle) for handle in handles)
        claim_ids = tuple(token.value for token in claim_tokens)
        if not handle_keys and not claim_ids:
            raise ValueError("producer decision binding requires an owner")
        if (
            len(set(handle_keys)) != len(handle_keys)
            or len(set(claim_ids)) != len(claim_ids)
        ):
            raise ValueError("producer decision binding requires unique owners")
        callback_token = next(self._producer_binding_sequence)
        delivery_future: Future[object] = Future()
        # Everything above is an argument check and stays here. Everything below
        # reads and writes decided state, so the worker applies it -- the
        # single-writer rule. The caller gets `delivery_future`
        # immediately and waits on that, exactly as before.
        self._enqueue_intent(
            "bind_producer_decision_waits",
            lambda: self._apply_bind_producer_decision_waits(
                handle_keys,
                claim_ids,
                future,
                delivery_future,
                callback_token,
                pipeline_session_id=pipeline_session_id,
                transfer=transfer,
            ),
        )
        return delivery_future

    def _apply_bind_producer_decision_waits(
        self,
        handle_keys: tuple[tuple[int, int, int], ...],
        claim_ids: tuple[int, ...],
        future: Future[object],
        delivery_future: Future[object],
        callback_token: int,
        *,
        pipeline_session_id: str | None,
        transfer: bool,
    ) -> None:
        try:
            with self._state_lock:
                waits = []
                for handle_key in handle_keys:
                    wait = self._producer_waits.get(handle_key)
                    if wait is None:
                        raise SchedulerInvariantError(
                            "producer decision Future has no checkpoint wait owner"
                        )
                    if wait.pipeline_session_id != pipeline_session_id:
                        raise SchedulerInvariantError(
                            "producer decision Future crossed pipeline ownership"
                        )
                    if not transfer and wait.future is not None:
                        raise SchedulerInvariantError(
                            "checkpoint wait already has a producer Future"
                        )
                    if transfer and (
                        wait.future is None
                        or not wait.future.done()
                        or wait.future_failure is not None
                        or wait.delivery_future is None
                        or not wait.delivery_future.done()
                        or wait.phase is not SchedulerExternalWaitPhase.DELIVERY_READY
                    ):
                        raise SchedulerInvariantError(
                            "checkpoint wait transfer lacks a completed CPU step"
                        )
                    waits.append(wait)
                claims = []
                for claim_id in claim_ids:
                    claim = self._transitive_claims.get(claim_id)
                    if claim is None:
                        raise SchedulerInvariantError(
                            "producer decision Future has no dependency claim owner"
                        )
                    if claim.pipeline_session_id != pipeline_session_id:
                        raise SchedulerInvariantError(
                            "dependency claim crossed pipeline ownership"
                        )
                    if not transfer and claim.future is not None:
                        raise SchedulerInvariantError(
                            "dependency claim already has a producer Future"
                        )
                    if transfer and (
                        claim.future is None
                        or not claim.future.done()
                        or claim.future_failure is not None
                        or claim.delivery_future is None
                        or not claim.delivery_future.done()
                        or claim.phase
                        is not SchedulerExternalWaitPhase.DELIVERY_READY
                    ):
                        raise SchedulerInvariantError(
                            "dependency claim transfer lacks a completed CPU step"
                        )
                    claims.append(claim)
                for wait in waits:
                    wait.future = future
                    wait.delivery_future = delivery_future
                    wait.binding_token = callback_token
                    wait.phase = SchedulerExternalWaitPhase.PREPARATION_PENDING
                    wait.phase_started_at = time.perf_counter()
                    wait.deadline_at = (
                        wait.phase_started_at + self._watchdog_config.timeout_seconds
                    )
                    wait.observations[
                        "producer_decision_wait_transferred"
                        if transfer
                        else "producer_decision_wait_bound"
                    ] += 1
                for claim in claims:
                    claim.future = future
                    claim.delivery_future = delivery_future
                    claim.binding_token = callback_token
                    claim.phase = SchedulerExternalWaitPhase.PREPARATION_PENDING
                    claim.phase_started_at = time.perf_counter()
                    claim.deadline_at = (
                        claim.phase_started_at + self._watchdog_config.timeout_seconds
                    )
                    claim.observations[
                        "dependency_claim_transferred"
                        if transfer
                        else "dependency_claim_bound"
                    ] += 1
        except BaseException as error:
            self._request_producer_terminal(error)
            # An intent has no caller stack to unwind into, so the violation
            # reaches the waiter the same way every other producer failure does:
            # through the future it is already blocked on. `_request_producer_
            # terminal` still stops the device.
            if not delivery_future.done():
                delivery_future.set_exception(error)
            return
        future.add_done_callback(
            lambda completed: self._producer_future_completed(
                handle_keys,
                callback_token,
                completed,
                claim_ids,
            )
        )

    def _request_producer_terminal(self, error: BaseException) -> None:
        with self._state_lock:
            if self._producer_callback_failure is None:
                self._producer_callback_failure = error
            sequence = next(self._sequence)
            self._queue.put((float("-inf"), sequence, self._wake))

    def _producer_future_completed(
        self,
        handle_keys: tuple[tuple[int, int, int], ...],
        callback_token: int,
        future: Future[object],
        claim_ids: tuple[int, ...] = (),
    ) -> None:
        failure: str | None = None
        result: object | None = None
        if future.cancelled():
            failure = "producer decision Future was cancelled"
        else:
            try:
                error = future.exception()
            except BaseException as callback_error:
                error = callback_error
            if error is not None:
                failure = f"{type(error).__name__}: {error}"
            else:
                result = future.result()
        # Inspecting the future is a read and stays on the callback thread.
        # Everything below moves decided state, so the worker applies it --
        # the single-writer rule. The delivery future is resolved
        # inside the intent, after the mutation, which is what keeps a consumer
        # that waits on it from observing the state from before.
        self._enqueue_intent(
            "producer_future_completed",
            lambda: self._apply_producer_future_completed(
                handle_keys,
                callback_token,
                claim_ids,
                failure,
                result,
            ),
        )

    def _apply_producer_future_completed(
        self,
        handle_keys: tuple[tuple[int, int, int], ...],
        callback_token: int,
        claim_ids: tuple[int, ...],
        failure: str | None,
        result: object | None,
    ) -> None:
        with self._state_lock:
            unknown = []
            delivery_futures: set[Future[object]] = set()
            for handle_key in handle_keys:
                wait = self._producer_waits.get(handle_key)
                if wait is not None and wait.binding_token == callback_token:
                    wait.future_failure = failure
                    if failure is None:
                        wait.phase = SchedulerExternalWaitPhase.DELIVERY_READY
                        wait.phase_started_at = time.perf_counter()
                        wait.deadline_at = (
                            wait.phase_started_at
                            + self._watchdog_config.timeout_seconds
                        )
                    if wait.delivery_future is not None:
                        delivery_futures.add(wait.delivery_future)
                    wait.observations["producer_decision_wait_wakes"] += 1
                    if failure is not None:
                        wait.observations[
                            "producer_decision_wait_future_failures"
                        ] += 1
                    continue
                if self._producer_wait_tombstones.get(handle_key) == callback_token:
                    continue
                unknown.append(handle_key)
            for claim_id in claim_ids:
                claim = self._transitive_claims.get(claim_id)
                if claim is not None and claim.binding_token == callback_token:
                    claim.future_failure = failure
                    if failure is None:
                        claim.phase = SchedulerExternalWaitPhase.DELIVERY_READY
                        claim.phase_started_at = time.perf_counter()
                        claim.deadline_at = (
                            claim.phase_started_at
                            + self._watchdog_config.timeout_seconds
                        )
                    if claim.delivery_future is not None:
                        delivery_futures.add(claim.delivery_future)
                    claim.observations["dependency_claim_wakes"] += 1
                    continue
                if self._claim_tombstones.get(claim_id) == callback_token:
                    continue
                unknown.append((claim_id, -1, -1))
            if unknown and self._liveness_error is None:
                self._producer_callback_failure = SchedulerInvariantError(
                    f"stale producer callback has no owner: {unknown}"
                )
            # No wake needed: this body already runs on the worker that
            # `_enqueue_intent` woke.
        for delivery_future in delivery_futures:
            if delivery_future.done() or failure is not None:
                continue
            delivery_future.set_result(result)

    def _register_producer_decision_wait(
        self,
        run: _PersistentRun[_ModelT],
        task: _BatchTask[_ModelT, Any],
        handle: object,
    ) -> None:
        wait = self._new_producer_decision_wait(run, task, handle)
        with self._state_lock:
            self._insert_producer_decision_wait_locked(run, wait)

    def _new_producer_decision_wait(
        self,
        run: _PersistentRun[_ModelT],
        task: _BatchTask[_ModelT, Any],
        handle: object,
    ) -> _ProducerDecisionWait:
        handle_key = self._resident_handle_key(handle)
        created_at = time.perf_counter()
        return _ProducerDecisionWait(
            handle_key=handle_key,
            handle=handle,
            sequence=task.sequence,
            run_identity=id(run),
            pipeline_session_id=task.pipeline_session_id,
            observations=run.scheduler_observations,
            created_at=created_at,
            phase=SchedulerExternalWaitPhase.AWAITING_BIND,
            phase_started_at=created_at,
            deadline_at=created_at + self._watchdog_config.timeout_seconds,
        )

    def _insert_producer_decision_wait_locked(
        self,
        run: _PersistentRun[_ModelT],
        wait: _ProducerDecisionWait,
    ) -> None:
        existing = self._producer_waits.get(wait.handle_key)
        if existing is not None:
            # Name both decisions. The key is a row, and a row-keyed message
            # cannot distinguish "the same checkpoint arrived twice" from "two
            # different tasks checkpointed the same row" -- which are opposite
            # bugs with opposite fixes.
            raise SchedulerInvariantError(
                "checkpoint handle already has a producer decision wait: "
                f"handle={wait.handle_key} "
                f"held_by_sequence={existing.sequence} "
                f"held_phase={existing.phase.value} "
                f"held_age_seconds={time.perf_counter() - existing.created_at:.6f} "
                f"rejected_sequence={wait.sequence}"
            )
        self._producer_waits[wait.handle_key] = wait
        run.scheduler_observations["producer_decision_wait_registered"] += 1
        run.scheduler_observations["producer_decision_waits_peak"] = max(
            run.scheduler_observations["producer_decision_waits_peak"],
            sum(
                candidate.run_identity == id(run)
                for candidate in self._producer_waits.values()
            ),
        )

    def _record_dependency_bundle_completion(
        self,
        run: _PersistentRun[_ModelT],
        task: _BatchTask[_ModelT, Any],
        result: object,
        resident_handle: object | None,
    ) -> None:
        bundle_id = task.dependency_bundle_id
        index = task.dependency_bundle_index
        if bundle_id is None or index is None:
            raise SchedulerInvariantError("bundle completion lacks member identity")
        publish: tuple[_DependencyBundleOwner, tuple[object, ...]] | None = None
        completed_at = time.perf_counter()
        with self._state_lock:
            bundle = self._dependency_bundles.get(bundle_id)
            if bundle is None or bundle.published:
                raise SchedulerInvariantError("bundle completion has no live owner")
            if bundle.member_sequences[index] != task.sequence or bundle.completed[index]:
                raise SchedulerInvariantError("bundle member completion is stale")
            bundle.results[index] = result
            bundle.resident_handles[index] = resident_handle
            bundle.completed[index] = True
            if resident_handle is not None:
                bundle.wait_owners[index] = (run, task, resident_handle)
            if all(bundle.completed):
                if bundle.parent_wait_owner is not None:
                    parent_wait = bundle.parent_wait_owner
                    if parent_wait.handle_key in self._producer_waits:
                        raise SchedulerInvariantError(
                            "bundle parent wait was recreated before publication"
                        )
                    parent_run = next(
                        (
                            candidate
                            for candidate in self._persistent_runs.values()
                            if id(candidate) == parent_wait.run_identity
                        ),
                        None,
                    )
                    if parent_run is None:
                        raise SchedulerInvariantError(
                            "bundle parent wait lost its persistent run"
                        )
                    self._insert_producer_decision_wait_locked(
                        parent_run,
                        parent_wait,
                    )
                for wait_owner in bundle.wait_owners:
                    if wait_owner is None:
                        continue
                    owner_run, owner_task, handle = wait_owner
                    wait = self._new_producer_decision_wait(
                        owner_run,
                        owner_task,
                        handle,
                    )
                    self._insert_producer_decision_wait_locked(owner_run, wait)
                bundle.published = True
                bundle.observations["dependency_bundle_published"] += 1
                if bundle.parent_wait_owner is not None:
                    bundle.observations[
                        "dependency_bundle_parent_wait_published"
                    ] += 1
                publish = (bundle, tuple(bundle.results))  # type: ignore[arg-type]
        self._observer.record_span(
            "bundle_member_complete",
            completed_at,
            time.perf_counter(),
            resource="control",
            lane=f"scheduler {run.key.device}",
            target_device=run.key.device,
            model=run.key.model,
            pipeline_session_id=task.pipeline_session_id,
            bundle_id=bundle_id,
            member_index=index,
            resident=resident_handle is not None,
        )
        if publish is None:
            return
        bundle, results = publish
        publication_started_at = time.perf_counter()
        if not bundle.future.done():
            bundle.future.set_result(results)
        with self._state_lock:
            if self._dependency_bundles.get(bundle.bundle_id) is bundle:
                del self._dependency_bundles[bundle.bundle_id]
        self._observer.record_span(
            "bundle_transaction_publish",
            publication_started_at,
            time.perf_counter(),
            resource="control",
            lane=f"scheduler {run.key.device}",
            target_device=run.key.device,
            model=run.key.model,
            pipeline_session_id=bundle.pipeline_session_id,
            bundle_id=bundle.bundle_id,
            member_count=len(bundle.submissions),
            parent_resident=bundle.parent_handle_key is not None,
        )

    def _consume_producer_decision_wait(
        self,
        handle: object,
        *,
        pipeline_session_id: str | None,
    ) -> tuple[_ProducerDecisionWait, DependencyClaimToken | None]:
        handle_key = self._resident_handle_key(handle)
        with self._state_lock:
            wait = self._producer_waits.get(handle_key)
            if wait is None:
                raise SchedulerInvariantError(
                    "resident generation control has no producer decision wait"
                )
            if wait.pipeline_session_id != pipeline_session_id:
                raise SchedulerInvariantError(
                    "resident generation control crossed pipeline ownership"
                )
            if wait.future is None:
                raise SchedulerInvariantError(
                    "resident generation control consumed an unbound wait"
                )
            if not wait.future.done() or wait.future_failure is not None:
                raise SchedulerInvariantError(
                    "resident generation control consumed an unresolved CPU decision"
                )
            if (
                wait.delivery_future is None
                or not wait.delivery_future.done()
                or wait.phase is not SchedulerExternalWaitPhase.DELIVERY_READY
            ):
                raise SchedulerInvariantError(
                    "resident generation control consumed an undelivered CPU decision"
                )
            del self._producer_waits[handle_key]
            self._producer_wait_tombstones[handle_key] = wait.binding_token
            while len(self._producer_wait_tombstones) > 256:
                self._producer_wait_tombstones.pop(next(iter(self._producer_wait_tombstones)))
            wait.observations["producer_decision_wait_consumed"] += 1
            claim_token = self._handle_claim_tokens.get(handle_key)
            if claim_token is not None:
                claim = self._transitive_claims.get(claim_token.value)
                if claim is None or claim.handle_key != handle_key:
                    raise SchedulerInvariantError(
                        "resident control found a stale claim-handle mapping"
                    )
        return wait, claim_token

    def _begin_inherited_claim_control(
        self,
        task: _BatchTask[_ModelT, Any],
        handle_key: tuple[int, int, int],
        *,
        terminal: bool,
    ) -> None:
        token = task.inherited_claim_token
        if token is None:
            return
        with self._state_lock:
            owner = self._transitive_claims.get(token.value)
            if owner is None or owner.handle_key != handle_key:
                raise SchedulerInvariantError(
                    "resident control cannot inherit a stale dependency claim"
                )
            if self._handle_claim_tokens.get(handle_key) != token:
                raise SchedulerInvariantError(
                    "resident control claim lineage index is inconsistent"
                )
            del self._handle_claim_tokens[handle_key]
            owner.handle_key = None
            owner.observations["dependency_claim_control_started"] += 1

    def _complete_inherited_claim_control(
        self,
        task: _BatchTask[_ModelT, Any],
        resident_handle: object | None,
    ) -> None:
        token = task.inherited_claim_token
        if token is None:
            return
        new_key = (
            None
            if resident_handle is None
            else self._resident_handle_key(resident_handle)
        )
        with self._state_lock:
            owner = self._transitive_claims.get(token.value)
            if owner is None or owner.handle_key is not None:
                raise SchedulerInvariantError(
                    "resident control completed without active claim lineage"
                )
            if new_key is not None and new_key in self._handle_claim_tokens:
                raise SchedulerInvariantError(
                    "resident control produced a duplicate claim-handle mapping"
                )
            owner.handle_key = new_key
            if new_key is not None:
                self._handle_claim_tokens[new_key] = token
                if task.dependency_bundle_id is not None:
                    # Evidence, not a verdict. `submit_batch_group` forbids a bundle member *declaring* a claim; nothing forbids it INHERITING one,
                    # and this is where the inherited claim lands on a handle its own bundle is about to record in `resident_handles`.
                    # Two owners, one handle -- what `resident handle belongs to multiple dependency bundles` reports one quantum later, from a place that can't say how the state was reached.
                    # Count, don't raise: whether to extend the prohibition or transfer ownership depends on how often recursive recovery happens, and raising here destroys that measurement.
                    owner.observations["bundle_member_inherited_claim"] += 1
            owner.observations[
                "dependency_claim_control_checkpointed"
                if new_key is not None
                else "dependency_claim_control_terminal"
            ] += 1

    def release_dependency_claim(
        self,
        token: DependencyClaimToken,
        *,
        pipeline_session_id: str | None,
    ) -> None:
        """Release one completed parent decision that requested no bundle."""
        # Worker-applied like every other external mutation. There is no future
        # to carry a violation here, so it reaches the caller the only way it
        # can: `_request_producer_terminal` stops the device, and the caller
        # learns about it through whatever future it is already waiting on.
        self._enqueue_intent(
            "release_dependency_claim",
            lambda: self._apply_release_dependency_claim(
                token,
                pipeline_session_id=pipeline_session_id,
            ),
        )

    def _apply_release_dependency_claim(
        self,
        token: DependencyClaimToken,
        *,
        pipeline_session_id: str | None,
    ) -> None:
        try:
            self._apply_release_dependency_claim_locked(
                token,
                pipeline_session_id=pipeline_session_id,
            )
        except BaseException as error:
            self._request_producer_terminal(error)

    def _apply_release_dependency_claim_locked(
        self,
        token: DependencyClaimToken,
        *,
        pipeline_session_id: str | None,
    ) -> None:
        with self._state_lock:
            claim = self._transitive_claims.get(token.value)
            if claim is None:
                raise SchedulerInvariantError("dependency claim has no live owner")
            if claim.pipeline_session_id != pipeline_session_id:
                raise SchedulerInvariantError(
                    "dependency claim release crossed pipeline ownership"
                )
            if (
                claim.future is None
                or not claim.future.done()
                or claim.future_failure is not None
                or claim.delivery_future is None
                or not claim.delivery_future.done()
                or claim.phase is not SchedulerExternalWaitPhase.DELIVERY_READY
            ):
                raise SchedulerInvariantError(
                    "dependency claim released before its CPU decision"
                )
            del self._transitive_claims[token.value]
            if claim.handle_key is not None:
                self._handle_claim_tokens.pop(claim.handle_key, None)
            self._claim_tombstones[token.value] = claim.binding_token
            claim.observations["dependency_claim_released"] += 1

    def _external_waits_snapshot(self) -> list[dict[str, object]]:
        now = time.perf_counter()
        with self._state_lock:
            waits = tuple(self._producer_waits.values())
            claims = tuple(self._transitive_claims.items())
        return [
            {
                "kind": "producer_decision_wait",
                "completion_source": wait.completion_source,
                "wake_source": wait.wake_source,
                "created_at_monotonic": wait.created_at,
                "age_seconds": max(0.0, now - wait.created_at),
                "phase": wait.phase.value,
                "phase_started_at_monotonic": wait.phase_started_at,
                "deadline_at_monotonic": wait.deadline_at,
                "deadline_remaining_seconds": wait.deadline_at - now,
                "pipeline_session_id": wait.pipeline_session_id,
                "sequence": wait.sequence,
                "handle": {
                    "session_id": wait.handle_key[0],
                    "slot": wait.handle_key[1],
                    "generation": wait.handle_key[2],
                },
                "future_state": (
                    "unbound"
                    if wait.future is None
                    else "cancelled"
                    if wait.future.cancelled()
                    else "done"
                    if wait.future.done()
                    else "running"
                    if wait.future.running()
                    else "pending"
                ),
                "future_failure": wait.future_failure,
            }
            for wait in waits
        ] + [
            {
                "kind": "dependency_claim_wait",
                "completion_source": "producer_decision_future",
                "wake_source": "model_worker_queue",
                "created_at_monotonic": claim.created_at,
                "age_seconds": max(0.0, now - claim.created_at),
                "phase": claim.phase.value,
                "phase_started_at_monotonic": claim.phase_started_at,
                "deadline_at_monotonic": claim.deadline_at,
                "deadline_remaining_seconds": claim.deadline_at - now,
                "pipeline_session_id": claim.pipeline_session_id,
                "claim_token": claim_id,
                "parent_handle": claim.handle_key,
                "future_state": (
                    "unbound"
                    if claim.future is None
                    else "cancelled"
                    if claim.future.cancelled()
                    else "done"
                    if claim.future.done()
                    else "running"
                    if claim.future.running()
                    else "pending"
                ),
                "future_failure": claim.future_failure,
            }
            for claim_id, claim in claims
        ]

    def _dependency_bundles_snapshot(self) -> list[dict[str, object]]:
        with self._state_lock:
            bundles = tuple(self._dependency_bundles.values())
        return [
            {
                "kind": "dependency_bundle",
                "bundle_id": bundle.bundle_id,
                "claim_token": bundle.claim_token.value,
                "pipeline_session_id": bundle.pipeline_session_id,
                "parent_handle": None
                if bundle.parent_handle_key is None
                else {
                    "session_id": bundle.parent_handle_key[0],
                    "slot": bundle.parent_handle_key[1],
                    "generation": bundle.parent_handle_key[2],
                },
                "member_sequences": list(bundle.member_sequences),
                "completed": list(bundle.completed),
                "resident_handles": [
                    None
                    if handle is None
                    else {
                        "session_id": self._resident_handle_key(handle)[0],
                        "slot": self._resident_handle_key(handle)[1],
                        "generation": self._resident_handle_key(handle)[2],
                    }
                    for handle in bundle.resident_handles
                ],
                "published": bundle.published,
                "future_state": (
                    "cancelled"
                    if bundle.future.cancelled()
                    else "done"
                    if bundle.future.done()
                    else "running"
                ),
            }
            for bundle in bundles
        ]

    def _producer_wait_terminal_error(self) -> BaseException | None:
        with self._state_lock:
            callback_failure = self._producer_callback_failure
            waits = tuple(self._producer_waits.values())
            claims = tuple(self._transitive_claims.items())
        if callback_failure is not None:
            return callback_failure
        failed = next((wait for wait in waits if wait.future_failure), None)
        if failed is not None:
            failed.observations["producer_decision_wait_terminals"] += 1
            return RuntimeError(
                "producer decision failed for handle "
                f"{failed.handle_key}: {failed.future_failure}"
            )
        failed_claim = next(
            ((claim_id, claim) for claim_id, claim in claims if claim.future_failure),
            None,
        )
        if failed_claim is not None:
            claim_id, claim = failed_claim
            claim.observations["dependency_claim_terminals"] += 1
            return RuntimeError(
                f"producer decision failed for claim {claim_id}: "
                f"{claim.future_failure}"
            )
        return None

    @property
    def timings(self) -> tuple[ModelTaskTiming, ...]:
        """Return a stable snapshot of completed task timings."""
        return self._observer.timings

    def cancel_pending(self) -> int:
        """Cancel work that has not started, leaving the active task alone."""
        with self._state_lock:
            return self._cancel_pending_locked()

    def request_device_terminal(self, error: BaseException) -> None:
        """Wake this lane and fail it through the normal terminal path."""
        with self._state_lock:
            if self._liveness_error is not None or self._requested_terminal_error is not None:
                return
            self._requested_terminal_error = error
            sequence = next(self._sequence)
            self._queue.put((float("-inf"), sequence, self._wake))

    def close(
        self, *, cancel_pending: bool = False, wait: bool = True
    ) -> None:
        """Reject new work, optionally cancel queued work, and stop safely."""
        if wait and threading.current_thread() is self._thread:
            raise RuntimeError("model worker cannot close itself")
        with self._state_lock:
            self._begin_close_locked(cancel_pending=cancel_pending)
        if wait:
            self._thread.join()

    def __enter__(self) -> ModelWorkerPool[_ModelT]:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _run(self) -> None:
        try:
            self._run_tasks()
        except BaseException as error:
            if not isinstance(error, SchedulerLivenessError):
                try:
                    error = self._terminate_liveness(
                        error,
                        classification=self._classify_action_failure(error),
                    )
                except BaseException as reporting_failure:  # noqa: BLE001
                    # Belt to `_terminate_liveness`'s braces. Whatever it costs
                    # to *explain* a failure, every waiter still has to be told
                    # there was one -- this block aborting is what turned a
                    # dead worker into a process that hung until it was killed.
                    _LOGGER.exception("liveness reporting failed")
                    error.__cause__ = reporting_failure
            with self._state_lock:
                self._closed = True
                pending = tuple(self._pending.values())
                self._pending.clear()
            for task in pending:
                if not task.future.done():
                    task.future.set_exception(error)
        finally:
            self._worker_stopped.set()

    def _enqueue_intent(self, name: str, apply: Callable[[], None]) -> None:
        """Hand one external mutation to the scheduler worker.

        The caller never waits for it: everything an external caller needs back
        -- a `Future`, usually -- is built in the calling thread before the
        intent is queued. That is what keeps this from becoming a wait the
        scheduler cannot see, which is the failure R7 already cost this project.
        """
        with self._intent_lock:
            self._pending_intents.append((name, apply))
        # Never separate these. An intent that lands while the worker is parked
        # on `self._queue.get(timeout=...)` and does not wake it is a stall that
        # looks exactly like the hang this scheduler exists to prevent.
        self._queue.put((float("-inf"), next(self._sequence), self._wake))

    def _drain_intents(self) -> None:
        """Apply every queued external mutation. The only place they land."""
        while True:
            with self._intent_lock:
                if not self._pending_intents:
                    return
                batch = tuple(self._pending_intents)
                self._pending_intents.clear()
            for name, apply in batch:
                apply()
                self._applied_intents += 1
                self._last_intent_name = name

    def _run_tasks(self) -> None:
        while True:
            self._drain_intents()
            if self._liveness_error is not None:
                raise self._liveness_error
            if self._requested_terminal_error is not None:
                raise self._terminate_liveness(
                    self._requested_terminal_error,
                    classification="device-terminal",
                )
            producer_error = self._producer_wait_terminal_error()
            if producer_error is not None:
                raise self._terminate_liveness(
                    producer_error,
                    classification="external-contract",
                )
            if self._stop_event is not None and self._stop_event.is_set():
                with self._state_lock:
                    self._begin_close_locked(cancel_pending=True)
            observation = self._observe_scheduler_state()
            if (
                observation.state.unresolved
                and not observation.legal_actions
                and not external_waits_cover_state(observation.state)
            ):
                raise self._terminate_liveness(
                    SchedulerInvariantError(
                        "actionless unresolved state has no exact external owner"
                    ),
                    classification=self._classify_liveness(observation.state),
                )
            try:
                runnable = bool(observation.legal_actions)
                unresolved = observation.state.unresolved
                if runnable:
                    _priority, _sequence, item = self._queue.get_nowait()
                elif unresolved:
                    if self._watchdog_started_at is None:
                        self._watchdog_started_at = time.perf_counter()
                    remaining = max(
                        0.0,
                        self._watchdog_config.timeout_seconds
                        - (time.perf_counter() - self._watchdog_started_at),
                    )
                    _priority, _sequence, item = self._queue.get(timeout=remaining)
                else:
                    self._clear_blocked_state()
                    _priority, _sequence, item = self._queue.get()
            except queue.Empty:
                if observation.legal_actions:
                    self._clear_blocked_state()
                    with self._execution_lock:
                        self._accept_action_result(
                            self._drive_persistent_quantum(observation)
                        )
                elif observation.state.unresolved:
                    self._report_blocked_state()
                continue
            try:
                if item is self._stop:
                    if self._abort_active_on_close:
                        error = RuntimeError("model worker stopped")
                        self._fail_dependency_bundles(error)
                        for run in self._persistent_runs.values():
                            self._fail_persistent_run(run, error)
                        return
                    if observation.legal_actions:
                        self._queue.put((float("inf"), _sequence, item))
                        with self._execution_lock:
                            self._accept_action_result(
                                self._drive_persistent_quantum(observation)
                            )
                        continue
                    if observation.state.unresolved:
                        error = RuntimeError(
                            "model worker closed with unresolved scheduler ownership"
                        )
                        self._fail_dependency_bundles(error)
                        for run in self._persistent_runs.values():
                            self._fail_persistent_run(run, error)
                    return
                if item is self._wake:
                    continue
                task = item
                assert isinstance(task, (_Task, _BatchTask))
                if isinstance(task, _BatchTask):
                    self._enqueue_persistent(task)
                    continue
                with self._state_lock:
                    if self._stop_event is not None and self._stop_event.is_set():
                        self._begin_close_locked(cancel_pending=True)
                    self._pending.pop(task.sequence, None)
                    should_run = task.future.set_running_or_notify_cancel()
                if not should_run:
                    continue
                started_at = time.perf_counter()
                succeeded = False
                model_loaded = False
                try:
                    with self._execution_lock:
                        model_started = time.perf_counter()
                        model, model_loaded = self._model_for(task.key)
                        model_finished = time.perf_counter()
                        if task.trace_model_load:
                            self._record_model_resource_span(
                                task.key,
                                "model_load" if model_loaded else "model_activate",
                                model_started,
                                model_finished,
                                pipeline_session_id=task.pipeline_session_id,
                            )
                        result = task.operation(model)
                except BaseException as error:
                    task.future.set_exception(error)
                else:
                    succeeded = True
                    task.future.set_result(result)
                finally:
                    finished_at = time.perf_counter()
                    timing = ModelTaskTiming(
                        sequence=task.sequence,
                        key=task.key,
                        priority=task.priority,
                        job_type=task.job_type,
                        queue_wait_seconds=started_at - task.submitted_at,
                        run_seconds=finished_at - started_at,
                        succeeded=succeeded,
                        model_loaded=model_loaded,
                        pipeline_session_id=task.pipeline_session_id,
                    )
                    if task.record_timing:
                        with self._state_lock:
                            self._observer.task_completed(timing)
            finally:
                self._queue.task_done()

    def _has_persistent_work(self) -> bool:
        return bool(self._observe_scheduler_state().legal_actions)

    def _observe_scheduler_state(self) -> _SchedulerObservation[_ModelT]:
        started = time.perf_counter()
        self._validate_claim_lineage()
        state, runs_by_id = self._scheduler_state()
        snapshot_phase_seconds = dict(self._last_snapshot_phase_seconds)
        snapshot_finished = time.perf_counter()
        validate_scheduler_state(state)
        validation_finished = time.perf_counter()
        legal_actions = self._available_actions(state, validate_state=False)
        legality_finished = time.perf_counter()
        return _SchedulerObservation(
            state=state,
            runs_by_id=runs_by_id,
            legal_actions=legal_actions,
            snapshot_seconds=snapshot_finished - started,
            snapshot_phase_seconds=snapshot_phase_seconds,
            validation_seconds=validation_finished - snapshot_finished,
            legality_seconds=legality_finished - validation_finished,
        )

    def _available_actions(
        self,
        state: SchedulerState,
        *,
        validate_state: bool = True,
    ) -> tuple[SchedulerAction, ...]:
        if validate_state:
            validate_scheduler_state(state)
        return tuple(
            action
            for action in enabled_actions(state)
            if (
                state.version,
                state.measurement_epoch,
                action.run_id,
                action.kind,
            )
            not in self._rejected_actions
        )

    @staticmethod
    def _accumulate_seconds(
        target: dict[str, float],
        name: str,
        seconds: float,
    ) -> None:
        target[name] = target.get(name, 0.0) + seconds

    def _accept_action_result(self, result: ActionResult | None) -> None:
        if result is None:
            return
        snapshot_started = time.perf_counter()
        self._validate_claim_lineage()
        post_state, runs = self._scheduler_state()
        post_snapshot_phase_seconds = dict(self._last_snapshot_phase_seconds)
        snapshot_finished = time.perf_counter()
        validate_scheduler_state(post_state)
        validation_finished = time.perf_counter()
        run = runs.get(result.action.run_id)
        if run is not None:
            self._accumulate_seconds(
                run.stats.scheduler_decision_phase_seconds,
                "post_snapshot",
                snapshot_finished - snapshot_started,
            )
            for name, seconds in post_snapshot_phase_seconds.items():
                self._accumulate_seconds(
                    run.stats.scheduler_decision_phase_seconds,
                    f"post_snapshot_{name}",
                    seconds,
                )
            self._accumulate_seconds(
                run.stats.scheduler_decision_phase_seconds,
                "post_validation",
                validation_finished - snapshot_finished,
            )
        self._recent_transitions.append(
            {
                "status": result.status.value,
                "action": asdict(result.action),
                "observed_version": result.observed_version,
                "post_version": post_state.version,
                "reason": result.reason,
            }
        )
        del self._recent_transitions[:-64]
        if result.status in {
            ActionResultStatus.APPLIED,
            ActionResultStatus.TERMINAL,
        }:
            if post_state.version <= result.observed_version:
                raise RuntimeError(
                    f"{result.status.value} action did not change scheduler state"
                )
            self._progress_epoch += 1
            self._rejected_actions.clear()
            self._clear_blocked_state()
            return
        if result.status is ActionResultStatus.REJECTED:
            if post_state.version != result.observed_version:
                # True only while decided state is single-writer. If this fires
                # again, something mutates it outside the worker's drain -- the
                # last intent name is the first place to look, not this call
                # site (the single-writer rule).
                raise RuntimeError(
                    "changed state cannot reject a stale action "
                    f"(observed={result.observed_version} "
                    f"post={post_state.version} "
                    f"applied_intents={self._applied_intents} "
                    f"last_intent={self._last_intent_name})"
                )
            # The epoch comes from `post_state`, not from the result. A
            # rejection means the decided state did not move, so the world that
            # made the action illegal is the one this observation just read --
            # and when the next reading differs the epoch moves and the memo
            # entry stops matching, which is precisely when the action deserves
            # another try.
            self._rejected_actions.add(
                (
                    result.observed_version,
                    post_state.measurement_epoch,
                    result.action.run_id,
                    result.action.kind,
                )
            )
            return
        if result.status is ActionResultStatus.STALE:
            if post_state.version <= result.observed_version:
                raise RuntimeError("stale action did not observe a newer state")
            return
        if result.status is ActionResultStatus.WAITING_EXTERNAL:
            return
        raise RuntimeError(f"unsupported action result: {result.status}")

    def _has_unresolved_persistent_state(self) -> bool:
        return any(
            run.pending
            or run.pending_controls
            or run.in_flight
            or run.prefill_pending
            or run.active_by_item
            or (run.session is not None and bool(run.session.occupied_count))
            for run in self._persistent_runs.values()
        )

    def _clear_blocked_state(self) -> None:
        self._watchdog_started_at = None

    def _report_blocked_state(self) -> None:
        now = time.perf_counter()
        started_at = self._watchdog_started_at
        if started_at is None:
            raise SchedulerInvariantError("watchdog fired without an armed deadline")
        blocked_seconds = now - started_at
        state, _runs = self._scheduler_state()
        error = self._terminate_liveness(
            RuntimeError(
                "event-driven scheduler made no progress for "
                f"{blocked_seconds:.3f}s"
            ),
            classification=self._classify_liveness(state),
        )
        raise error

    def _classify_liveness(self, state: SchedulerState) -> str:
        legal = enabled_actions(state)
        if legal and not self._available_actions(state):
            return "runtime-adapter"
        if any(
            run.pending
            and (not run.admission_safe or not run.arena_admission_safe)
            for run in state.runs
        ):
            return "external-contract"
        return "algorithm-invariant"

    @staticmethod
    def _classify_action_failure(error: BaseException) -> str:
        if isinstance(error, SchedulerInvariantError):
            return "algorithm-invariant"
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return "external-contract"
        return "runtime-adapter"

    def _terminate_liveness(
        self,
        cause: BaseException,
        *,
        classification: str,
    ) -> SchedulerLivenessError:
        """Stop the device and publish the failure. The report is OPTIONAL.

        Every gatherer below reads the state the failure may have broken, and `_scheduler_state` *validates* -- so a violated invariant raises from inside the reporter with the error it was called to report.
        When that happened the `except` in `_run` aborted before completing a single pending future, and a run that should have failed in seconds hung until killed, ten stem workers parked on futures nobody would resolve.
        Cost two incidents: once hiding a real exception behind a pytest timeout, once wedging a two-song run.

        So: failing is mandatory, explaining is best-effort. A gatherer that raises contributes its message to `evidence_errors` and nothing else; publication runs either way.
        """
        if self._liveness_error is not None:
            return self._liveness_error
        now = time.perf_counter()
        evidence_errors: dict[str, str] = {}

        def gather(name: str, produce: Callable[[], Any], fallback: Any) -> Any:
            try:
                return produce()
            except BaseException as failure:  # noqa: BLE001 - reporting only
                evidence_errors[name] = f"{type(failure).__name__}: {failure}"
                return fallback

        observed = gather("scheduler_state", self._scheduler_state, None)
        state, runs_by_id = (
            observed if observed is not None else (None, dict(self._persistent_runs))
        )
        actions = (
            () if state is None else gather("enabled_actions", lambda: enabled_actions(state), ())
        )
        external_waits = gather("external_waits", self._external_waits_snapshot, [])
        dependency_bundles = gather(
            "dependency_bundles", self._dependency_bundles_snapshot, []
        )
        pipeline_session_ids = sorted(
            {
                session_id
                for run in runs_by_id.values()
                for session_id in gather(
                    "pipeline_session_ids",
                    lambda run=run: self._run_pipeline_session_ids(run),
                    (),
                )
            }
            | {
                str(wait["pipeline_session_id"])
                for wait in external_waits
                if wait.get("pipeline_session_id") is not None
            }
            | {
                str(bundle["pipeline_session_id"])
                for bundle in dependency_bundles
                if bundle.get("pipeline_session_id") is not None
            }
        )
        frames = sys._current_frames()
        report: dict[str, object] = {
            "schema_version": 2,
            "classification": classification,
            "message": f"{type(cause).__name__}: {cause}",
            "state_version": None if state is None else state.version,
            "progress_epoch": None if state is None else state.progress_epoch,
            "pipeline_session_ids": pipeline_session_ids,
            "state": None if state is None else asdict(state),
            "enabled_actions": [asdict(action) for action in actions],
            "rejected_actions": [
                {
                    "state_version": version,
                    "measurement_epoch": epoch,
                    "run_id": run_id,
                    "kind": kind.value,
                }
                for version, epoch, run_id, kind in sorted(
                    self._rejected_actions,
                    key=lambda value: (value[0], value[1], value[2], value[3].value),
                )
            ],
            "external_waits": external_waits,
            "watchdog": {
                "timeout_seconds": self._watchdog_config.timeout_seconds,
                "started_at_monotonic": self._watchdog_started_at,
                "age_seconds": (
                    None
                    if self._watchdog_started_at is None
                    else max(0.0, now - self._watchdog_started_at)
                ),
            },
            "dependency_bundles": dependency_bundles,
            "ownership": gather("ownership", self._persistent_state_snapshot, []),
            "recent_transitions": list(self._recent_transitions),
            # Every counter the run accumulated, keyed by run. A failed run
            # writes no `transcription_timing.scheduler` to its metadata, so
            # until now a failure discarded exactly the evidence collected to
            # explain it -- and "the counter is absent" reads identically to
            # "the thing never happened". It cost one wrong diagnosis already.
            "observations": gather(
                "observations",
                lambda: {
                    self._session_label(run): dict(run.scheduler_observations)
                    for run in self._persistent_runs.values()
                    if run.scheduler_observations
                },
                {},
            ),
            "device_memory": self._capacity_memory_info,
            "thread_stacks": {
                thread.name: traceback.format_list(
                    traceback.extract_stack(frames[thread.ident])
                )
                for thread in threading.enumerate()
                if thread.ident in frames
            },
        }
        # A device-side assert reports asynchronously, so `cause` carries the
        # only stack that names the offending kernel. Stringifying it into the
        # message discarded that: every consumer then saw a traceback ending at
        # its own `wait_for_first`, which is why the recovery-path assert stayed
        # unlocated. Keep the traceback in the durable report, and chain the
        # exception so it also survives into the stem's stderr log.
        report["cause_traceback"] = traceback.format_exception(
            type(cause), cause, cause.__traceback__
        )
        # Named, not silent. A degraded report is evidence about the failure --
        # "the state could not even be read" is itself a finding.
        report["evidence_errors"] = evidence_errors
        error = SchedulerLivenessError(
            "scheduler liveness contract failed; affected device stopped: "
            f"{type(cause).__name__}: {cause}",
            report,
        )
        error.__cause__ = cause
        self._liveness_error = error
        gather("fail_bundles", lambda: self._fail_dependency_bundles(error), None)
        for run in list(runs_by_id.values()):
            # Unresolved is a *filter*, and a filter that cannot be evaluated
            # must not decide "leave it running": failing a settled run is
            # harmless, leaving an unsettled one is the hang.
            unresolved = gather(
                "run_unresolved",
                lambda run=run: self._scheduler_run_state(run, run_id=0).unresolved,
                True,
            )
            if unresolved:
                gather(
                    "fail_run",
                    lambda run=run: self._fail_persistent_run(run, error),
                    None,
                )
        with self._state_lock:
            pending = tuple(self._pending.values())
            self._pending.clear()
        for task in pending:
            if not task.future.done():
                task.future.set_exception(error)
        if self._observer.trace_spans:
            self._observer.record_span(
                "scheduler_liveness_failure",
                now,
                time.perf_counter(),
                resource="control",
                lane=f"scheduler {next(iter(runs_by_id.values())).key.device}",
                pipeline_session_id=(
                    pipeline_session_ids[0]
                    if len(pipeline_session_ids) == 1
                    else None
                ),
                pipeline_session_ids=pipeline_session_ids,
                terminal=True,
                classification=classification,
                state_version=None if state is None else state.version,
                progress_epoch=None if state is None else state.progress_epoch,
            )
        if self._device is not None and self._device_terminal is not None:
            self._device_terminal(self._device, error)
        return error

    def _fail_dependency_bundles(self, error: BaseException) -> None:
        with self._state_lock:
            bundles = tuple(self._dependency_bundles.values())
            claims = tuple(self._transitive_claims.items())
            self._dependency_bundles.clear()
            self._transitive_claims.clear()
            self._handle_claim_tokens.clear()
        for bundle in bundles:
            if not bundle.future.done():
                bundle.future.set_exception(error)
        for _claim_id, claim in claims:
            if claim.delivery_future is not None and not claim.delivery_future.done():
                claim.delivery_future.set_exception(error)
            if claim.future is not None and not claim.future.done():
                claim.future.cancel()

    def _persistent_state_snapshot(self) -> list[dict[str, object]]:
        snapshots: list[dict[str, object]] = []
        for run in self._persistent_runs.values():
            session = run.session
            controls = []
            for task in run.pending_controls:
                handle = getattr(task.item, "resident_handle", None)
                can_resume = getattr(session, "can_resume", None)
                controls.append(
                    {
                        "action": getattr(task.item, "action", None),
                        "inherited_claim_token": (
                            None
                            if task.inherited_claim_token is None
                            else task.inherited_claim_token.value
                        ),
                        "slot": getattr(handle, "slot", None),
                        "generation": getattr(handle, "generation", None),
                        "can_resume": bool(
                            session is not None
                            and (not callable(can_resume) or can_resume(handle))
                        ),
                    }
                )
            slots = []
            for index, slot in enumerate(
                () if session is None else getattr(session, "slots", ())
            ):
                row = getattr(slot, "row", None)
                request = None if row is None else getattr(row, "request", None)
                slots.append(
                    {
                        "slot": index,
                        "paused": bool(getattr(slot, "paused", False)),
                        "tracked": id(request) in run.active_by_item,
                    }
                )
            snapshots.append(
                {
                    "model": run.key.model,
                    "job_types": sorted(self._run_job_types(run)),
                    "kv_capacity": run.capacity,
                    "session_id": (None if session is None else session.session_id),
                    "pending": len(run.pending),
                    "prefill": sum(
                        len(batch.selected) for batch in run.prefill_pending
                    ),
                    "in_flight": len(run.in_flight),
                    "tracked": len(run.active_by_item),
                    "active": 0 if session is None else session.active_count,
                    "occupied": 0 if session is None else session.occupied_count,
                    "displaced": (
                        0 if session is None else int(getattr(session, "displaced_count", 0))
                    ),
                    "available": run.width
                    if session is None
                    else run.logical_available_count,
                    "physical_available": (
                        run.width if session is None else session.available_count
                    ),
                    "controls": controls,
                    "slots": slots,
                }
            )
        return snapshots

    def _enqueue_persistent(self, task: _BatchTask[_ModelT, Any]) -> None:
        handle = getattr(task.item, "resident_handle", None)
        action = getattr(task.item, "action", None)
        if handle is not None and action in {"resume", "discard"}:
            try:
                self._enqueue_generation_control(task, handle, action)
            except BaseException as error:
                with self._state_lock:
                    self._pending.pop(task.sequence, None)
                if not task.future.done():
                    task.future.set_exception(error)
                if isinstance(error, SchedulerInvariantError):
                    raise
            return
        capacity = int(getattr(task.item, "initial_kv_capacity_bucket", 2048))
        # Length is **not** in this key any more. It split one model's work into
        # separate sessions by how long each request estimated it would run,
        # and those sessions then competed for one device instead of sharing
        # one batch. The split existed because rows sharing an arena shared its
        # capacity; A3 removed that, and nothing had removed the split.
        key = (
            task.key,
            task.compatibility_key,
            task.continuous_replacement,
        )
        run = self._persistent_runs.get(key)
        if run is None:
            assert task.continuous_factory is not None
            run = _PersistentRun(
                key=task.key,
                priority=task.priority,
                compatibility_key=task.compatibility_key,
                capacity=capacity,
                # A forced ceiling, or none. With none, the width this lane
                # opens at is `_resolve_arena_width` -- its demand, bounded by
                # the device -- and this number only bounds the first admission
                # cohort, which the session then grows past.
                width=task.max_batch_size or task.minimum_width,
                width_ceiling=task.max_batch_size,
                arena_cost_probe=task.arena_cost,
                kv_pool_declaration=task.kv_pool,
                minimum_width=task.minimum_width,
                maximum_width=task.maximum_width,
                factory=task.continuous_factory,
                allow_replacement=task.continuous_replacement,
                purposes={self._task_purpose(task)},
            )
            self._persistent_runs[key] = run
        else:
            if task.arena_cost is not run.arena_cost_probe:
                raise RuntimeError("persistent arena cost source changed")
            if task.minimum_width != run.minimum_width:
                raise RuntimeError("persistent lane minimum width changed")
            if task.maximum_width != run.maximum_width:
                raise RuntimeError("persistent lane maximum width changed")
            run.priority = min(run.priority, task.priority)
            run.purposes.add(self._task_purpose(task))
        run.pending.append(task)

    @staticmethod
    def _task_purpose(task: _BatchTask[_ModelT, Any]) -> str:
        if str(getattr(task.item, "candidate_name", "")) == "bootstrap":
            return "bootstrap"
        return task.job_type

    @staticmethod
    def _session_label(run: _PersistentRun[_ModelT]) -> str:
        order = {"bootstrap": 0, "primary": 1, "recovery": 2, "control": 3}
        purposes = "/".join(
            sorted(run.purposes, key=lambda value: (order.get(value, 9), value))
        )
        return f"{run.key.model} {purposes}".rstrip()

    def _enqueue_generation_control(
        self,
        task: _BatchTask[_ModelT, Any],
        handle: object,
        action: str,
    ) -> None:
        run = next(
            (
                candidate
                for candidate in self._persistent_runs.values()
                if candidate.session is not None
                and candidate.session.session_id == getattr(handle, "session_id", None)
            ),
            None,
        )
        if run is None or run.session is None:
            raise ValueError("resident generation session is unavailable")
        handle_key = self._resident_handle_key(handle)
        if handle_key not in run.resident_handles_by_key:
            raise SchedulerInvariantError(
                "resident generation handle is not scheduler-owned"
            )
        if any(
            self._resident_handle_key(candidate.item.resident_handle) == handle_key
            for candidate in run.pending_controls
        ):
            raise SchedulerInvariantError(
                "resident generation handle already has a control intent"
            )
        _wait, inherited_claim = self._consume_producer_decision_wait(
            handle,
            pipeline_session_id=task.pipeline_session_id,
        )
        task.inherited_claim_token = inherited_claim
        run.pending_controls.append(task)

    def _execute_generation_control(
        self,
        run: _PersistentRun[_ModelT],
        task: _BatchTask[_ModelT, Any],
        handle: object,
        action: str,
    ) -> None:
        assert run.session is not None
        is_displaced = getattr(run.session, "is_displaced", None)
        was_displaced = bool(callable(is_displaced) and is_displaced(handle))
        handle_key = self._resident_handle_key(handle)
        operation_started = time.perf_counter()
        slots_before = self._waterfall_slots(run)
        with self._state_lock:
            if not task.future.set_running_or_notify_cancel():
                self._pending.pop(task.sequence, None)
                raise SchedulerInvariantError(
                    "resident control was cancelled after consuming its wait"
                )
            self._pending.pop(task.sequence, None)
        run.started_at[task.sequence] = time.perf_counter()
        self._begin_inherited_claim_control(
            task,
            handle_key,
            terminal=action == "discard",
        )
        if action == "discard":
            from muscriptor.generation_batch import GenerationControlResult

            run.session.discard(handle)
            run.resident_handles_by_key.pop(handle_key, None)
            self._complete_inherited_claim_control(task, None)
            self._refresh_capacity_measurement(run)
            if was_displaced:
                run.scheduler_observations["displaced_row_discards"] += 1
            task.future.set_result(GenerationControlResult(discarded=True))
            self._record_persistent_timing(
                run,
                task,
                False,
                result=GenerationControlResult(discarded=True),
            )
            self._record_waterfall_event(
                run,
                "discard",
                operation_started,
                slots_before,
            )
            return
        request = run.session.resume(handle)
        run.resident_handles_by_key.pop(handle_key, None)
        self._refresh_capacity_measurement(run)
        if was_displaced:
            run.scheduler_observations["displaced_row_restores"] += 1
        run.active_by_item[id(request)] = (
            task,
            PreparedWorkItem(request),
        )
        self._record_waterfall_event(
            run,
            "resume",
            operation_started,
            slots_before,
        )

    def _collect_preemption_telemetry(self, run: _PersistentRun[_ModelT]) -> None:
        """Move what preemption cost this run's session into its observations.

        Called once per quantum and once more wherever the session is about to be let go, so nothing is attributed to a run that didn't incur it and nothing is dropped because the last quantum was the one that preempted.
        Not bracketed around a single action on purpose: the admission path preempts too, and bracketing decode alone is what reported a run's 66 preemptions as none.
        """
        drain = getattr(run.session, "drain_preemption_telemetry", None)
        if not callable(drain):
            return
        for name, value in dict(drain()).items():
            if value:
                run.scheduler_observations[name] += int(value)

    def _drive_persistent_quantum(
        self,
        observation: _SchedulerObservation[_ModelT] | None = None,
    ) -> ActionResult | None:
        if observation is None:
            observation = self._observe_scheduler_state()
        state = observation.state
        runs_by_id = observation.runs_by_id
        legal_actions = observation.legal_actions
        legality_finished = time.perf_counter()
        preferred_by_id = self._preferred_actions(legal_actions)
        if not preferred_by_id:
            return
        runnable = [runs_by_id[run_id] for run_id in preferred_by_id]
        force_decode = self._admission_burst_turns >= 2 and any(
            candidate.session is not None and candidate.session.active_count
            for candidate in runnable
        )
        run = min(
            runnable,
            key=lambda candidate: self._persistent_run_score(
                candidate,
                action=preferred_by_id[
                    next(
                        run_id
                        for run_id, mapped in runs_by_id.items()
                        if mapped is candidate
                    )
                ],
                force_decode=force_decode,
            ),
        )
        selected_id = next(
            run_id for run_id, candidate in runs_by_id.items() if candidate is run
        )
        selected_action = preferred_by_id[selected_id]
        self._collect_preemption_telemetry(run)
        policy_finished = time.perf_counter()
        self._accumulate_seconds(
            run.stats.scheduler_decision_phase_seconds,
            "snapshot",
            observation.snapshot_seconds,
        )
        for name, seconds in observation.snapshot_phase_seconds.items():
            self._accumulate_seconds(
                run.stats.scheduler_decision_phase_seconds,
                f"snapshot_{name}",
                seconds,
            )
        self._accumulate_seconds(
            run.stats.scheduler_decision_phase_seconds,
            "invariant_validation",
            observation.validation_seconds,
        )
        self._accumulate_seconds(
            run.stats.scheduler_decision_phase_seconds,
            "legality",
            observation.legality_seconds,
        )
        self._accumulate_seconds(
            run.stats.scheduler_decision_phase_seconds,
            "policy",
            policy_finished - legality_finished,
        )
        action = selected_action.kind.value
        operation_started = time.perf_counter()
        slots_before = self._waterfall_slots(run)
        waterfall_action = (
            "session_init"
            if action == "condition_prefill" and run.session is None
            else action
        )
        model_span: tuple[str, float, float] | None = None
        session_first_decode_span: tuple[float, float] | None = None
        if action == "decode":
            self._admission_burst_turns = 0
        else:
            self._admission_burst_turns += 1
        telemetry_started = time.perf_counter()
        self._record_scheduler_snapshot(run, runnable, action=selected_action.kind)
        for candidate in runnable:
            if candidate is not run:
                candidate.age += 1
        run.age = 0
        telemetry_finished = time.perf_counter()
        self._accumulate_seconds(
            run.stats.scheduler_decision_phase_seconds,
            "telemetry",
            telemetry_finished - telemetry_started,
        )
        run.stats.scheduler_decision_cpu_seconds += (
            observation.snapshot_seconds
            + observation.validation_seconds
            + observation.legality_seconds
            + policy_finished
            - legality_finished
            + telemetry_finished
            - telemetry_started
        )
        completed: list[tuple[object, object]] = []
        action_adapter_started: float | None = None
        # Every span below is recorded from one `finally`, which most actions
        # reach without ever dispatching a quantum. None means "this span is
        # not a decode dispatch", and the four fields are simply absent.
        dispatch_readiness: dict[str, int] | None = None
        try:
            # S5c: every non-decode action may mutate slots, pages or arenas,
            # so every in-flight quantum on this device is consumed first --
            # decode consumes its own run's inline. A no-op while nothing is
            # in flight, which is always the case with async decode off.
            if action != "decode":
                for candidate in runs_by_id.values():
                    self._drain_in_flight_quantum(candidate)
            if action == "release_admit":
                action_adapter_started = time.perf_counter()
                loaded, model_started, model_finished = (
                    self._execute_reclaim_admission(
                        selected_action,
                        runs_by_id,
                    )
                )
                if loaded or model_finished - model_started >= 0.05:
                    model_span = (
                        "model_load" if loaded else "model_activate",
                        model_started,
                        model_finished,
                    )
                self._accumulate_seconds(
                    run.stats.scheduler_decision_phase_seconds,
                    "model_activation",
                    model_finished - model_started,
                )
                return ActionResult(
                    ActionResultStatus.APPLIED,
                    selected_action,
                    state.version,
                )
            model_started = time.perf_counter()
            model, loaded = self._model_for(run.key)
            model_finished = time.perf_counter()
            model_elapsed = model_finished - model_started
            if loaded or model_elapsed >= 0.05:
                model_span = (
                    "model_load" if loaded else "model_activate",
                    model_started,
                    model_finished,
                )
            operation_started = model_finished
            self._accumulate_seconds(
                run.stats.scheduler_decision_phase_seconds,
                "model_activation",
                model_elapsed,
            )
            post_activation_state, _post_activation_runs = self._scheduler_state()
            if (
                action != "capacity_prepare"
                and post_activation_state.version != state.version
            ):
                return ActionResult(
                    ActionResultStatus.STALE,
                    selected_action,
                    state.version,
                    reason="model activation changed the scheduler snapshot",
                )
            action_adapter_started = operation_started
            run.model_loaded = run.model_loaded or loaded
            if action == "capacity_prepare":
                self._prepare_run_capacity(run, model)
                return ActionResult(
                    ActionResultStatus.APPLIED,
                    selected_action,
                    state.version,
                )
            if action == "resize":
                if not self._resize_run_width(run, model):
                    return ActionResult(
                        ActionResultStatus.STALE,
                        selected_action,
                        state.version,
                        reason="the demand this resize was selected for changed",
                    )
                return ActionResult(
                    ActionResultStatus.APPLIED,
                    selected_action,
                    state.version,
                )
            if action == "bundle_admit":
                self._admit_dependency_bundle(selected_action.participants[0])
                return ActionResult(
                    ActionResultStatus.APPLIED,
                    selected_action,
                    state.version,
                )
            if action == "cancel":
                if not self._cancel_pending_action(run):
                    return ActionResult(
                        ActionResultStatus.REJECTED,
                        selected_action,
                        state.version,
                        reason="cancelled task was no longer pending",
                    )
                return ActionResult(
                    ActionResultStatus.APPLIED,
                    selected_action,
                    state.version,
                )
            if action in {"discard", "restore"}:
                task = run.pending_controls.pop(0)
                self._execute_generation_control(
                    run,
                    task,
                    getattr(task.item, "resident_handle", None),
                    action,
                )
                return ActionResult(
                    ActionResultStatus.APPLIED,
                    selected_action,
                    state.version,
                )
            if action == "condition_batch":
                stale_reason = self._execute_condition_batch(
                    selected_action,
                    runs_by_id,
                    model,
                )
                if stale_reason is not None:
                    return ActionResult(
                        ActionResultStatus.STALE,
                        selected_action,
                        state.version,
                        reason=stale_reason,
                    )
                return ActionResult(
                    ActionResultStatus.APPLIED,
                    selected_action,
                    state.version,
                )
            if action == "fail":
                failure = self._terminate_liveness(
                    RuntimeError(
                        "scheduler has no resource-feasible transition for device"
                    ),
                    classification="external-contract",
                )
                return ActionResult(
                    ActionResultStatus.TERMINAL,
                    selected_action,
                    state.version,
                    reason=f"{type(failure).__name__}: {failure}",
                )
            if action in {"preempt_restore", "preempt_condition"}:
                assert run.session is not None
                self._refresh_capacity_measurement(run)
                if action == "preempt_condition" and not self._preempt_condition_resources_fit(
                    run,
                    refresh_device=True,
                ):
                    # Same classification question as the credit check below:
                    # `refresh_device=True` reads the device live, so this can
                    # flip with the scheduler snapshot unmoved, and that is a
                    # rejection at the current version rather than a stale one.
                    fresh_state, _fresh_runs = self._scheduler_state()
                    return ActionResult(
                        ActionResultStatus.STALE
                        if fresh_state.version != state.version
                        else ActionResultStatus.REJECTED,
                        selected_action,
                        state.version,
                        reason="transient GPU budget changed before preemption",
                    )
                preempt = getattr(run.session, "preempt_for_admission", None)
                if not callable(preempt) or not preempt():
                    raise RuntimeError(
                        "the preempt macro was selected but found no victim"
                    )
                # No counter here. `preemptions`, drained from the session,
                # counts this event, and `selected_action_preempt_condition_
                # observations` counts the action that causes it -- three
                # names for one thing was two too many.
                self._refresh_capacity_measurement(run)
                if action == "preempt_restore":
                    task = run.pending_controls.pop(0)
                    self._execute_generation_control(
                        run,
                        task,
                        getattr(task.item, "resident_handle", None),
                        "restore",
                    )
                else:
                    pending_before = len(run.pending)
                    deferred_prefill = self._fill_persistent_run(run, model)
                    if len(run.pending) >= pending_before:
                        raise RuntimeError(
                            "the preempt macro consumed no pending row"
                        )
                    if deferred_prefill:
                        self._admit_conditioned_batch(run)
                return ActionResult(
                    ActionResultStatus.APPLIED,
                    selected_action,
                    state.version,
                )
            if run.prefill_pending:
                self._admit_conditioned_batch(run)
                return ActionResult(
                    ActionResultStatus.APPLIED,
                    selected_action,
                    state.version,
                )
            if action == "condition_prefill":
                if run.session is None:
                    fresh_state, _fresh_runs = self._scheduler_state()
                    if fresh_state.version != state.version:
                        return ActionResult(
                            ActionResultStatus.STALE,
                            selected_action,
                            state.version,
                            reason=(
                                "arena capacity changed before nonresident admission"
                            ),
                        )
                    if not fresh_state.runs[selected_id].arena_admission_safe:
                        raise SchedulerInvariantError(
                            "legal nonresident admission lacks a current arena proof"
                        )
                pending_before = len(run.pending)
                deferred_prefill = self._fill_persistent_run(run, model)
                if len(run.pending) >= pending_before:
                    return ActionResult(
                        ActionResultStatus.REJECTED,
                        selected_action,
                        state.version,
                        reason="condition/prefill transaction admitted no row",
                    )
                if deferred_prefill:
                    self._admit_conditioned_batch(run)
                return ActionResult(
                    ActionResultStatus.APPLIED,
                    selected_action,
                    state.version,
                )
            session = run.session
            if session is None or not session.active_count:
                return ActionResult(
                    ActionResultStatus.REJECTED,
                    selected_action,
                    state.version,
                    reason="selected action produced no runnable session row",
                )
            graph_before = {
                name: int(value)
                for name, value in dict(
                    getattr(session, "cuda_graph_telemetry", lambda: {})()
                ).items()
            }
            variants_before = dict(
                getattr(session, "cuda_graph_variant_telemetry", lambda: {})()
            )
            drain_prefills = getattr(
                session,
                "drain_prefill_batches_by_width",
                None,
            )
            if drain_prefills is not None:
                for width, count in dict(drain_prefills()).items():
                    run.stats.prefill_batches_by_width[int(width)] = (
                        run.stats.prefill_batches_by_width.get(int(width), 0) + int(count)
                    )
            for method_name, target in (
                (
                    "drain_condition_batches_by_width",
                    run.stats.condition_batches_by_width,
                ),
                (
                    "drain_first_token_batches_by_width",
                    run.stats.first_token_batches_by_width,
                ),
                (
                    "drain_packed_prefill_batches_by_width",
                    run.stats.packed_prefill_batches_by_width,
                ),
            ):
                drain_batches = getattr(session, method_name, None)
                if drain_batches is not None:
                    for width, count in dict(drain_batches()).items():
                        target[int(width)] = target.get(int(width), 0) + int(count)
            self._drain_phase_gpu_ms(run, session)
            # Keep one hot graph-quantum family. q8 was tried and reverted: it
            # delivers every mechanical thing it promises -- far fewer
            # dispatches, less idle between actions, higher kernel residency --
            # and buys **zero** wall, because removing device idle cannot help a
            # workload that is not waiting on the device. A wider quantum also
            # overshoots, stepping rows a narrower boundary would have stopped;
            # `wasted_token_rows` reaches metadata since 2026-08-30 (S5a), so a
            # rerun can now confirm the overshoot half of that reading.

            quantum_steps = 4
            run.stats.quantum_steps_by_size[quantum_steps] = (
                run.stats.quantum_steps_by_size.get(quantum_steps, 0) + 1
            )
            decode_started = time.perf_counter()
            # Sampled here and nowhere else: one line later the quantum has
            # started retiring rows, and the span this rides on is recorded
            # when the action ends. See `_dispatch_readiness`.
            dispatch_readiness = self._dispatch_readiness(run)
            if self._last_decode_quantum_finished_at is not None:
                run.stats.scheduler_boundary_gap_seconds += max(
                    0.0,
                    decode_started - self._last_decode_quantum_finished_at,
                )
                run.stats.scheduler_boundary_gap_count += 1
            if self._async_decode and getattr(
                session, "split_decode_supported", False
            ):
                # S5c: consume the previous launch's tokens (a full sync), then
                # launch the next replay and return while it runs -- the
                # scheduler's decision work and the other lane's dispatches
                # execute beside it. The first dispatch of a burst launches
                # only; the last is consumed by a later dispatch or by
                # `_drain_in_flight_quantum` on the next slot-mutating action.
                stats = (
                    session.consume_quantum(
                        lambda item, result: completed.append((item, result)),
                        checkpointed=lambda item, result: completed.append(
                            (item, result)
                        ),
                    )
                    if session.quantum_in_flight
                    else None
                )
                launched = session.launch_quantum(quantum_steps)
                if stats is None and not launched:
                    # Nothing consumed and nothing to launch, yet DECODE was
                    # legal: `active_count` counts a preempted row no slot
                    # holds, and its restore can be refused (never take the
                    # last page). Sync mode live-spins here legally -- an
                    # empty quantum retries the restore and `quantum_count`
                    # moves the version. Reproduce exactly that; skipping it
                    # left an APPLIED action that changed nothing, which the
                    # progress contract kills (found on the first 16-song
                    # async arm, 2026-08-30).
                    stats = session.run_quantum(
                        quantum_steps,
                        lambda item, result: completed.append((item, result)),
                        checkpointed=lambda item, result: completed.append(
                            (item, result)
                        ),
                    )
            else:
                stats = session.run_quantum(
                    quantum_steps,
                    lambda item, result: completed.append((item, result)),
                    checkpointed=lambda item, result: completed.append(
                        (item, result)
                    ),
                )
            self._last_decode_quantum_finished_at = time.perf_counter()
            if waterfall_action == "session_init":
                session_first_decode_span = (
                    decode_started,
                    time.perf_counter(),
                )
            graph_after = {
                name: int(value)
                for name, value in dict(
                    getattr(session, "cuda_graph_telemetry", lambda: {})()
                ).items()
            }
            variants_after = dict(
                getattr(session, "cuda_graph_variant_telemetry", lambda: {})()
            )
            cache_after = dict(
                getattr(session, "cuda_graph_cache_telemetry", lambda: {})()
            )

            def graph_delta(name: str) -> int:
                return graph_after.get(name, 0) - graph_before.get(name, 0)

            # Assigned, not accumulated: the prefill cache belongs to the model
            # and its totals already span every job this worker has run.
            run.stats.prefill_graph = {
                name: int(value)
                for name, value in dict(
                    getattr(session, "prefill_graph_telemetry", lambda: {})()
                ).items()
            }
            run.stats.prefill_graph_refusals = {
                name: int(value)
                for name, value in dict(
                    getattr(session, "prefill_graph_refusals", lambda: {})()
                ).items()
            }
            run.stats.cuda_graph_captures += graph_delta("captures")
            run.stats.cuda_graph_replays += graph_delta("replays")
            run.stats.cuda_graph_refusals += graph_delta("graph_refusals")
            run.stats.cuda_graph_evictions += graph_delta("evictions")
            for key in variants_before.keys() | variants_after.keys():
                before = variants_before.get(key, {})
                after = variants_after.get(key, {})
                delta = {
                    name: int(after.get(name, 0)) - int(before.get(name, 0))
                    for name in after.keys() | before.keys()
                }
                delta = {name: value for name, value in delta.items() if value}
                if not delta:
                    continue
                target = run.stats.cuda_graph_variants.setdefault(key, {})
                for name, value in delta.items():
                    target[name] = target.get(name, 0) + value
            run.stats.cuda_graph_cache_entries_peak = max(
                run.stats.cuda_graph_cache_entries_peak,
                int(cache_after.get("entries", 0)),
            )
            run.stats.cuda_graph_cache_static_bytes_peak = max(
                run.stats.cuda_graph_cache_static_bytes_peak,
                int(cache_after.get("static_bytes", 0)),
            )
            run.stats.cuda_graph_cache_pool_bytes_peak = max(
                run.stats.cuda_graph_cache_pool_bytes_peak,
                int(cache_after.get("pool_bytes", 0)),
            )
            self._drain_phase_gpu_ms(run, session)
        except BaseException as error:
            failure = (
                error
                if isinstance(error, SchedulerLivenessError)
                else self._terminate_liveness(
                    error,
                    classification=self._classify_action_failure(error),
                )
            )
            return ActionResult(
                ActionResultStatus.TERMINAL,
                selected_action,
                state.version,
                reason=f"{type(failure).__name__}: {failure}",
            )
        finally:
            if action_adapter_started is not None:
                self._accumulate_seconds(
                    run.stats.scheduler_action_wall_seconds,
                    action,
                    time.perf_counter() - action_adapter_started,
                )
            if model_span is not None:
                model_action, model_started, model_finished = model_span
                self._record_model_resource_span(
                    run.key,
                    model_action,
                    model_started,
                    model_finished,
                    pipeline_session_id=(
                        next(iter(self._run_pipeline_session_ids(run)), None)
                    ),
                )
            if session_first_decode_span is not None and self._observer.trace_spans:
                first_decode_started, first_decode_finished = session_first_decode_span
                self._observer.record_span(
                    "session_first_decode",
                    first_decode_started,
                    first_decode_finished,
                    resource="gpu",
                    lane=f"{self._session_label(run)} init",
                    model=run.key.model,
                    session_id=run.session.session_id,
                    session_label=self._session_label(run),
                )
            self._record_waterfall_event(
                run,
                waterfall_action,
                operation_started,
                slots_before,
                dispatch_readiness=dispatch_readiness,
            )
        if stats is not None:
            self._accumulate_quantum_stats(run, stats)
        self._publish_quantum_completions(run, completed)
        return ActionResult(
            ActionResultStatus.APPLIED,
            selected_action,
            state.version,
        )

    def _accumulate_quantum_stats(
        self,
        run: _PersistentRun[_ModelT],
        stats: object,
    ) -> None:
        """Fold one consumed quantum's session stats into the run's telemetry."""
        run.quantum_count += 1
        run.stats.physical_steps += int(getattr(stats, "physical_steps", 0))
        run.stats.wasted_token_rows += int(getattr(stats, "wasted_token_rows", 0))
        for width, steps in dict(getattr(stats, "active_steps_by_width", {})).items():
            run.stats.active_steps_by_width[int(width)] = run.stats.active_steps_by_width.get(
                int(width), 0
            ) + int(steps)

    def _publish_quantum_completions(
        self,
        run: _PersistentRun[_ModelT],
        completed: list[tuple[object, object]],
    ) -> None:
        """Publish a consumed quantum's completions and checkpoints.

        Shared by the decode action and `_drain_in_flight_quantum`, so a
        completion drained ahead of a slot-mutating action follows exactly the
        path a synchronous quantum's would."""
        for index, (item, result) in enumerate(completed):
            resident_handle = getattr(result, "resident_handle", None)
            if resident_handle is not None:
                checkpoint_started = time.perf_counter()
                self._record_waterfall_event(
                    run,
                    "checkpoint",
                    checkpoint_started,
                    self._waterfall_slots(run),
                )
            task, prepared = run.active_by_item.pop(id(item))
            claim_token: DependencyClaimToken | None = None
            if resident_handle is not None:
                handle_key = self._resident_handle_key(resident_handle)
                run.resident_handles_by_key[handle_key] = resident_handle
                if task.dependency_bundle_id is None:
                    self._register_producer_decision_wait(run, task, resident_handle)
            self._complete_inherited_claim_control(task, resident_handle)
            if task.dependency_claim is not None:
                claim_token = DependencyClaimToken(task.sequence)
                created_at = time.perf_counter()
                owner = _TransitiveClaimOwner(
                    claim=task.dependency_claim,
                    device=run.key.device,
                    pipeline_session_id=task.pipeline_session_id,
                    run_identity=id(run),
                    created_at=created_at,
                    phase=SchedulerExternalWaitPhase.AWAITING_BIND,
                    phase_started_at=created_at,
                    deadline_at=(
                        created_at + self._watchdog_config.timeout_seconds
                    ),
                    observations=run.scheduler_observations,
                    handle_key=(
                        None
                        if resident_handle is None
                        else self._resident_handle_key(resident_handle)
                    ),
                )
                with self._state_lock:
                    self._transitive_claims[claim_token.value] = owner
                    if owner.handle_key is not None:
                        self._handle_claim_tokens[owner.handle_key] = claim_token
            completed_result = (
                result if prepared.complete is None else prepared.complete(result)
            )
            if claim_token is not None:
                completed_result = ClaimedBatchResult(completed_result, claim_token)
            if task.dependency_bundle_id is not None:
                self._record_dependency_bundle_completion(
                    run,
                    task,
                    completed_result,
                    resident_handle,
                )
            if not task.future.done():
                task.future.set_result(completed_result)
            self._record_persistent_timing(
                run,
                task,
                index == 0,
                result=completed_result,
                source_item=prepared.session_item,
            )

    def _drain_in_flight_quantum(self, run: _PersistentRun[_ModelT]) -> None:
        """Consume a run's in-flight replay before anything mutates its state.

        Publishing goes through `_publish_quantum_completions`, so a drained
        quantum is indistinguishable from a synchronous one downstream. No
        launch here: drain sites are about to change slots, pages or arenas,
        and the next decode dispatch relaunches on the new state."""
        session = run.session
        if session is None or not getattr(session, "quantum_in_flight", False):
            return
        completed: list[tuple[object, object]] = []
        drain_started = time.perf_counter()
        slots_before = self._waterfall_slots(run)
        stats = session.consume_quantum(
            lambda item, result: completed.append((item, result)),
            checkpointed=lambda item, result: completed.append((item, result)),
        )
        self._last_decode_quantum_finished_at = time.perf_counter()
        self._accumulate_quantum_stats(run, stats)
        self._record_waterfall_event(run, "decode", drain_started, slots_before)
        self._publish_quantum_completions(run, completed)

    def _execute_condition_batch(
        self,
        action: SchedulerAction,
        runs_by_id: dict[int, _PersistentRun[_ModelT]],
        model: _ModelT,
    ) -> str | None:
        """Encode one row per emitted run and route views to owner prefills."""
        if len(action.participants) < 2:
            raise RuntimeError("global condition batch requires multiple runs")
        cohort = [runs_by_id[run_id] for run_id in action.participants]
        if any(run.key != cohort[0].key for run in cohort[1:]):
            raise RuntimeError("global condition batch crossed model identity")
        if any(not run.pending for run in cohort):
            return "condition cohort changed before execution"
        selected = [(run, self._ordered_pending(run)[0]) for run in cohort]
        condition_key = selected[0][1].condition_compatibility_key
        if condition_key is None or any(
            task.condition_compatibility_key != condition_key
            or run.session is None
            or run.logical_available_count < 1
            or run.prefill_pending
            for run, task in selected
        ):
            return "condition cohort changed before execution"
        if any(task.future.cancelled() for _run, task in selected):
            return "condition cohort was cancelled before execution"


        prepared_by_run: list[
            tuple[_PersistentRun[_ModelT], _BatchTask[_ModelT, Any], PreparedWorkItem]
        ] = []
        for run, task in selected:
            pending_index = run.pending.index(task)
            run.pending.pop(pending_index)
            with self._state_lock:
                if not task.future.set_running_or_notify_cancel():
                    raise RuntimeError(
                        "condition cohort cancellation raced committed admission"
                    )
                self._pending.pop(task.sequence, None)
            run.started_at[task.sequence] = time.perf_counter()
            run.in_flight.append(task)
            prepared_items = (
                (PreparedWorkItem(task.item),)
                if task.prepare is None
                else task.prepare(model, (task.item,))
            )
            if len(prepared_items) != 1:
                raise RuntimeError("condition preparation returned the wrong row count")
            prepared_by_run.append((run, task, prepared_items[0]))

        session = cohort[0].session
        assert session is not None
        prepare_cohort = getattr(session, "prepare_condition_cohort", None)
        if not callable(prepare_cohort):
            raise RuntimeError("persistent session lacks global condition batching")
        encoded = prepare_cohort(
            tuple(prepared.session_item for _run, _task, prepared in prepared_by_run)
        )
        split_cohort = getattr(session, "split_condition_cohort", None)
        if not callable(split_cohort):
            raise RuntimeError("persistent session lacks global condition batching")
        conditioned = split_cohort(
            encoded,
            tuple(1 for _entry in prepared_by_run),
        )
        for (run, task, prepared), condition in zip(
            prepared_by_run,
            conditioned,
            strict=True,
        ):
            run.in_flight.remove(task)
            run.prefill_pending.append(
                _PendingPrefill(
                    selected=((task, prepared),),
                    conditioned=condition,
                )
            )
        cohort[0].scheduler_observations["global_condition_batches"] += 1
        cohort[0].scheduler_observations["global_condition_rows"] += len(cohort)
        return None

    def _cancel_pending_action(self, run: _PersistentRun[_ModelT]) -> bool:
        for tasks in (run.pending_controls, run.pending):
            for index, task in enumerate(tasks):
                if not task.future.cancelled():
                    continue
                if task.inherited_claim_token is not None:
                    raise SchedulerInvariantError(
                        "cancelled resident control retained dependency claim lineage"
                    )
                tasks.pop(index)
                with self._state_lock:
                    self._pending.pop(task.sequence, None)
                return True
        return False

    @staticmethod
    def _waterfall_slots(run: _PersistentRun[_ModelT]) -> list[dict[str, object]]:
        if run.session is None:
            return []
        slots: list[dict[str, object]] = []
        for index, slot in enumerate(getattr(run.session, "slots", ())):
            row = getattr(slot, "row", None)
            if row is None:
                continue
            request = getattr(row, "request", None)
            task_entry = run.active_by_item.get(id(request))
            task = None if task_entry is None else task_entry[0]
            slots.append(
                {
                    "session_id": run.session.session_id,
                    "slot": index,
                    "sequence": None if task is None else task.sequence,
                    "job_type": None if task is None else task.job_type,
                    "pipeline_session_id": (
                        None if task is None else task.pipeline_session_id
                    ),
                    "paused": bool(getattr(slot, "paused", False)),
                }
            )
        return slots

    def _record_waterfall_event(
        self,
        run: _PersistentRun[_ModelT],
        action: str,
        started_at: float,
        slots_before: list[dict[str, object]],
        *,
        finished_at: float | None = None,
        dispatch_readiness: dict[str, int] | None = None,
    ) -> None:
        if not self._observer.trace_spans:
            return
        session = run.session
        if session is None:
            return
        finished_at = time.perf_counter() if finished_at is None else finished_at
        slots_after = self._waterfall_slots(run)
        owners = sorted(
            {
                str(slot["pipeline_session_id"])
                for slot in (*slots_before, *slots_after)
                if slot.get("pipeline_session_id")
            }
        )
        condition_ready, prefill_ready, decode_ready = self._run_phase_depths(run)
        self._observer.record(
            {
                "start_seconds": self._observer.relative_time(started_at),
                "end_seconds": self._observer.relative_time(finished_at),
                "action": action,
                "session_id": session.session_id,
                "session_label": self._session_label(run),
                "pipeline_session_id": owners[0] if len(owners) == 1 else None,
                "pipeline_session_ids": owners,
                "model": run.key.model,
                "target_device": run.key.device,
                "active_before": sum(not bool(slot["paused"]) for slot in slots_before),
                "active_after": session.active_count,
                "occupied_before": len(slots_before),
                "occupied_after": session.occupied_count,
                "capacity": run.width,
                "resident_bytes": run.capacity_resident_bytes,
                "displaced_count": int(getattr(session, "displaced_count", 0)),
                "graph_static_bytes": run.capacity_graph_static_bytes,
                "owned_total_bytes": run.capacity_total_bytes,
                "kv_capacity": run.capacity,
                "pending_jobs": len(run.pending),
                "prefill_jobs": sum(
                    len(batch.selected) for batch in run.prefill_pending
                ),
                "condition_ready_width": condition_ready,
                "prefill_ready_width": prefill_ready,
                "decode_ready_width": decode_ready,
                "compatible_ready_width": (
                    condition_ready + prefill_ready + decode_ready
                ),
                "slots_before": slots_before,
                "slots_after": slots_after,
                **(dispatch_readiness or {}),
            }
        )

    def _record_model_resource_span(
        self,
        key: ModelKey,
        action: str,
        started_at: float,
        finished_at: float,
        *,
        pipeline_session_id: str | None = None,
    ) -> None:
        self._observer.record_span(
            action,
            started_at,
            finished_at,
            resource="gpu",
            lane=f"{key.device} model {key.model}",
            model=key.model,
            pipeline_session_id=pipeline_session_id,
        )

    @staticmethod
    def _run_pipeline_session_ids(run: _PersistentRun[_ModelT]) -> set[str]:
        return {
            str(task.pipeline_session_id)
            for task in ModelWorkerPool._persistent_run_tasks(run)
            if getattr(task, "pipeline_session_id", None) is not None
        }

    @staticmethod
    def _persistent_run_tasks(
        run: _PersistentRun[_ModelT],
    ) -> tuple[_BatchTask[_ModelT, Any], ...]:
        return (
            *run.pending,
            *run.pending_controls,
            *run.in_flight,
            *(task for batch in run.prefill_pending for task, _item in batch.selected),
            *(task for task, _item in run.active_by_item.values()),
        )

    @staticmethod
    def _run_job_types(run: _PersistentRun[_ModelT]) -> set[GenerationJobType]:
        return {
            getattr(task, "job_type", "primary")
            for task in ModelWorkerPool._persistent_run_tasks(run)
        }

    @staticmethod
    def _task_schedule_key(task: _BatchTask[_ModelT, Any]) -> tuple[object, ...]:
        return (
            getattr(task, "priority", 10),
            0 if getattr(task, "dependency_bundle_id", None) is not None else 1,
            getattr(task, "sequence", 0),
            getattr(task, "ready_order", ""),
        )

    @classmethod
    def _ordered_pending(
        cls,
        run: _PersistentRun[_ModelT],
    ) -> list[_BatchTask[_ModelT, Any]]:
        return sorted(run.pending, key=cls._task_schedule_key)

    @classmethod
    def _run_priority(cls, run: _PersistentRun[_ModelT]) -> int:
        tasks = cls._persistent_run_tasks(run)
        return min(
            (getattr(task, "priority", run.priority) for task in tasks),
            default=run.priority,
        )

    @staticmethod
    def _phase_depths(run: _PersistentRun[_ModelT]) -> tuple[int, int, int]:
        """The three depths, with no side effect. **The one definition.**

        Split out from `_run_phase_depths` so a second reader can sample them
        without also moving `demand_high_water`, which P4 sizes an arena from:
        a telemetry sample that changes an allocation is two variables, not one.
        """
        return (
            len(run.pending),
            sum(len(batch.selected) for batch in run.prefill_pending),
            run.session.active_count if run.session is not None else 0,
        )

    @staticmethod
    def _run_phase_depths(run: _PersistentRun[_ModelT]) -> tuple[int, int, int]:
        condition_ready, prefill_ready, decode_ready = (
            ModelWorkerPool._phase_depths(run)
        )
        # Recorded here because this is the one place the three depths are
        # resolved together, and their sum is the demand P4 sizes against. A
        # second definition elsewhere is how "ready width" would come to mean
        # two things.
        run.demand_high_water = max(
            run.demand_high_water, condition_ready + prefill_ready + decode_ready
        )
        return condition_ready, prefill_ready, decode_ready

    def _dispatch_readiness(self, run: _PersistentRun[_ModelT]) -> dict[str, int]:
        """Width and readiness at ONE instant, sampled at quantum dispatch.

        Scheduler-fill S0. The sibling `*_ready_width` fields on the same span are sampled when the action *ends*,
        so "decode ran narrow while rows were ready" is a comparison across two sampling points until something carries both.

        What the four numbers separate, which no pair of them can alone:
        * `dispatch_decode_width` IS `active_count`, and so is `decode_ready` -- same quantity, never independent evidence of rows left behind.
          Decode runs every installed unpaused row; it doesn't choose a narrower width.
        * so rows left behind are the ones NOT installed: `dispatch_condition_ready` + `dispatch_prefill_ready`.
        * `dispatch_free_slots` says whether they could have been. No free slot = a capacity story; slots free = a scheduling one, and only the second is that set's premise.

        ! `dispatch_prefill_ready` is zero by construction, and NOT for the reason this said until 2026-08-29.
        The credit went to the `PREFILL` barrier; that action has zero spans in every run measured.
        Real cause: `_fill_persistent_run`'s callers drain the queue inside their own action, before any snapshot, so `run.prefill` is never true when `enabled_actions` runs.
        Keep the field -- non-zero would mean a prefill survived an action -- but never read the zero as evidence about demand.

        `dispatch_producer_waits` is the lookahead and the four above can't substitute for it.
        They count rows that exist NOW; this counts rows that do not exist yet and are CERTAIN TO, because a checkpoint bound a future that will resolve and `undecided_handle_keys`'
        invariant says that future owns a real resident row.
        That is the "not ready but certain to become ready" condition the skip design is gated on, and the only quantity that can justify *waiting* rather than merely preferring
        -- preferring measured zero (`RESONFORGE_MIN_WIDTH`), because a preference can't conjure an alternative when nothing else is runnable.

        `dispatch_producer_waits_unbound` is registered-but-not-bound, reported apart on purpose: those are NOT certain, and a skip gated on them is gated on a hope.
        """
        session = run.session
        condition_ready, prefill_ready, decode_ready = (
            ModelWorkerPool._phase_depths(run)
        )
        identity = id(run)
        bound = unbound = 0
        for producer_wait in self._producer_waits.values():
            if producer_wait.run_identity != identity:
                continue
            if producer_wait.future is None:
                unbound += 1
            else:
                bound += 1
        return {
            "dispatch_decode_width": decode_ready,
            "dispatch_condition_ready": condition_ready,
            "dispatch_prefill_ready": prefill_ready,
            "dispatch_free_slots": (
                0 if session is None else max(0, run.width - session.occupied_count)
            ),
            "dispatch_producer_waits": bound,
            "dispatch_producer_waits_unbound": unbound,
        }

    def _scheduler_run_state(
        self,
        run: _PersistentRun[_ModelT],
        *,
        run_id: int,
        pool: PoolView | None = None,
        bundle_owned_handle_keys: frozenset[tuple[int, int, int]] = frozenset(),
        producer_wait_snapshot: tuple[
            tuple[tuple[int, int, int], bool], ...
        ]
        | None = None,
        phase_seconds: dict[str, float] | None = None,
    ) -> SchedulerRunState:
        started_at = time.perf_counter()
        session = run.session
        control_can_resume = False
        if session is not None and run.pending_controls:
            handle = getattr(
                run.pending_controls[0].item,
                "resident_handle",
                None,
            )
            can_resume = getattr(session, "can_resume", None)
            control_can_resume = not callable(can_resume) or bool(can_resume(handle))
        # Rows this lane's page supply can still fund, read once: it decides
        # both whether preemption is *needed* and what legality is told.
        # `None` is "unpriced", never "exhausted".
        page_rows = self._run_page_rows(run)
        page_blocked = page_rows is not None and page_rows <= 0
        needs_preempt = bool(
            session is not None
            and (
                (run.pending_controls and not control_can_resume)
                or (
                    run.pending
                    # Blocked on EITHER supply. This asked only `available_count == 0`, which is slot scarcity -- the pre-paging meaning of "blocked".
                    # A lane out of *pages* has free slots it can't use, so the trigger never fired and the parked row kept its whole-length reservation indefinitely.
                    # That is the hold-and-wait `preempt_for_admission` exists to remove, sitting unreachable behind this condition.
                    and (session.available_count == 0 or page_blocked)
                    and session.occupied_count > session.active_count
                )
            )
        )
        control_finished_at = time.perf_counter()
        control_keys = [
            self._resident_handle_key(task.item.resident_handle)
            for task in run.pending_controls
        ]
        if producer_wait_snapshot is None:
            state_lock = getattr(self, "_state_lock", None)
            if state_lock is None:
                producer_wait_snapshot = ()
            else:
                with state_lock:
                    producer_wait_snapshot = tuple(
                        (wait.handle_key, wait.future is None)
                        for wait in self._producer_waits.values()
                        if wait.run_identity == id(run)
                    )
        producer_wait_keys = {
            handle_key for handle_key, _unbound in producer_wait_snapshot
        }
        undecided_handle_keys = set(run.resident_handles_by_key) - set(control_keys)
        run_bundle_owned_keys = (
            set(run.resident_handles_by_key) & bundle_owned_handle_keys
        )
        if not producer_wait_keys <= undecided_handle_keys:
            raise SchedulerInvariantError(
                "producer decision wait does not own an undecided resident handle"
            )
        if not run_bundle_owned_keys <= undecided_handle_keys:
            raise SchedulerInvariantError(
                "dependency bundle does not own an undecided resident handle"
            )
        if producer_wait_keys & run_bundle_owned_keys:
            raise SchedulerInvariantError(
                "resident handle has both producer and dependency bundle owners"
            )
        ownership_finished_at = time.perf_counter()
        active = 0 if session is None else session.active_count
        occupied = 0 if session is None else session.occupied_count
        if pool is None:
            pool = self._pool(run.key.device).view(
                free_bytes=None, total_bytes=None
            )
        # What this run may still claim. Its own floor is added back, because
        # that is not memory it has to compete for -- the pool is holding it on
        # its behalf.
        pool_available_blocks = (
            None if pool.unbounded else pool.available_blocks(**self._own_credits(run))
        )
        # One comparison, and it is the same one everywhere -- which became
        # true when `_own_credits` replaced eleven hand-assembled argument
        # lists that had disagreed about which terms applied. The blocks
        # admission would newly promise used to be the left-hand side; nothing
        # promises blocks any more, so what is left to ask is whether this
        # run's declared floors are still covered by the device.
        admission_safe = pool_available_blocks is None or pool_available_blocks >= 0
        arena_admission_safe = self._nonresident_arena_admission_safe(run, pool)
        admission_finished_at = time.perf_counter()
        preempt_finished_at = time.perf_counter()
        condition_group = None
        if run.session is not None and run.pending:
            condition_task = self._ordered_pending(run)[0]
            condition_key = getattr(
                condition_task,
                "condition_compatibility_key",
                None,
            )
            if condition_key is not None:
                condition_group = repr(
                    (run.key, condition_key)
                )
        # What this lane should hold, against what it does. Computed here so
        # legality sees it as a fact of the snapshot rather than a decision:
        # whether the change is *allowed* is `enabled_actions`' business, and
        # whether it is worth making is the policy layer's.
        resize_target: int | None = None
        resize_delta = 0
        if session is not None and run.arena_cost is not None and not pool.unbounded:
            wanted = self._run_demand_width(run, pool)
            if wanted != run.width:
                resize_target = wanted
                resize_delta = self._arena_blocks(run, wanted) - run.arena_blocks
        state = SchedulerRunState(
            run_id=run_id,
            physical_width=run.width,
            session_exists=session is not None,
            pending=len(run.pending),
            controls=len(run.pending_controls),
            prefill=sum(len(batch.selected) for batch in run.prefill_pending),
            in_flight=len(run.in_flight),
            tracked=len(run.active_by_item),
            active=active,
            occupied=occupied,
            physical_available=(
                run.width if session is None else session.available_count
            ),
            logical_available=run.logical_available_count,
            paused=occupied - active,
            displaced=(
                0 if session is None else int(getattr(session, "displaced_count", 0))
            ),
            resident_handles=len(run.resident_handles_by_key),
            control_intents=len(control_keys),
            unique_control_intents=len(set(control_keys)),
            remaining_decode_tokens=(
                0
                if session is None
                else int(getattr(session, "remaining_decode_token_budget", 0))
            ),
            completed_quanta=run.quantum_count,
            decode_launch_in_flight=bool(
                getattr(session, "quantum_in_flight", False)
            ),
            cancelled_pending=sum(
                bool(
                    getattr(task, "future", None) is not None
                    and task.future.cancelled()
                )
                for task in (*run.pending_controls, *run.pending)
            ),
            control_action=(
                None
                if not run.pending_controls
                else str(getattr(run.pending_controls[0].item, "action", ""))
            ),
            control_can_resume=control_can_resume,
            preempt_allowed=bool(
                needs_preempt
                and session is not None
                and getattr(session, "can_preempt_for_admission", lambda: False)()
            ),
            admission_safe=admission_safe,
            capacity_ready=self._run_capacity_ready(run),
            arena_admission_safe=arena_admission_safe,
            pool_available_blocks=pool_available_blocks,
            page_rows_available=page_rows,
            producer_waits=len(producer_wait_snapshot),
            bundle_owned_handles=len(run_bundle_owned_keys),
            unbound_producer_waits=sum(
                unbound for _handle_key, unbound in producer_wait_snapshot
            ),
            condition_group=condition_group,
            condition_batch_limit=run.width if condition_group is not None else 0,
            resize_target_width=resize_target,
            resize_block_delta=resize_delta,
        )
        finished_at = time.perf_counter()
        if phase_seconds is not None:
            phases = {
                "control": control_finished_at - started_at,
                "wait_ownership": ownership_finished_at - control_finished_at,
                "admission": admission_finished_at - ownership_finished_at,
                "preempt": preempt_finished_at - admission_finished_at,
                "state_build": finished_at - preempt_finished_at,
            }
            for name, seconds in phases.items():
                phase_seconds[name] = phase_seconds.get(name, 0.0) + seconds
        return state

    def _pool(self, device: str) -> DeviceBlockPool:
        """The process-wide block pool for one device."""
        self._pool_devices.add(device)
        return device_block_pool(device)

    def _device_pool_views(
        self,
        runs: dict[int, _PersistentRun[_ModelT]],
    ) -> dict[str, PoolView]:
        """One pool reading per device, taken once for a whole snapshot.

        Reclaimable blocks are deliberately 0 here. A run may not count on another run's arena being closed unless it takes the `RELEASE_ADMIT` action, which is proved separately and atomically;
        folding reclaimable space into the ordinary admission test lets two runs both believe the same donor is theirs.
        """
        views: dict[str, PoolView] = {}
        for run in runs.values():
            device = run.key.device
            if device in views:
                continue
            views[device] = self._pool_view(device, runs)
        return views

    def _observe_pool_reading(
        self,
        device: str,
        runs: dict[int, _PersistentRun[_ModelT]],
        *,
        free_bytes: int,
        total_bytes: int,
    ) -> None:
        """Record what one device reading looked like. No decision is made here.

        Split out of `_pool_view`, which was 98 lines of which ~70 were this -- a function named for building a view, mostly recording.

        Exactly once is the intent: these are *device* readings, and `scheduler_observations` are summed across jobs, so recording one reading against every run on the device multiplies it by however many were open.
        Histograms not scalars for the same reason -- a summed max is not a max. Bucketed so the distribution stays small.
        """
        pool = self._pool(device)
        subject = next(
            (
                candidate
                for candidate in runs.values()
                if candidate.key.device == device
            ),
            None,
        )
        if subject is None:
            return
        transient_now = pool.measured_transient_bytes
        fragmentation_now = pool.measured_fragmentation_bytes
        rows_now = _resident_rows(device, runs)
        transient_at_width = pool.transient_bytes_for(rows_now)
        # Same arguments the view divides with, so the recorded reserve is the one admission compared against, not a second arithmetic that can drift.
        # `pool_transient_mib` stays beside it: no longer an input, but the two diverging is what would say the per-row price stopped tracking the whole.
        reserve_now = reserve_bytes(
            resident_rows=rows_now,
            transient_bytes=transient_at_width,
            fragmentation_bytes=fragmentation_now,
        )
        # Bucket per metric, not one width for all. A single 64 MiB bucket once covered every quantity here including the per-row price, which is itself ~one bucket wide
        # -- so that counter had exactly one usable value and reported it whatever the price really was.
        # A probe that can't resolve its own subject reports "nothing changed" and "nothing was measured" identically.
        for name, value, bucket_mib in (
            ("pool_reserve_mib", reserve_now, 64),
            ("pool_transient_mib", transient_now, 64),
            ("pool_transient_at_width_mib", transient_at_width, 64),
            ("pool_fragmentation_mib", fragmentation_now, 16),
            ("pool_free_mib", free_bytes, 64),
        ):
            quantum = bucket_mib * 1024**2
            self._observe_scheduler_histogram(
                subject, name, value // quantum * bucket_mib
            )
        self._observe_scheduler_histogram(subject, "pool_resident_rows", rows_now)
        if rows_now not in pool.measured_widths():
            # Priced by the bootstrap because nothing has decoded at this width
            # yet. Expected while an arena is opening and a defect if it
            # persists -- a width that never gets measured never gets its real
            # price, and the bootstrap is the number known to be wrong at both
            # ends.
            subject.scheduler_observations["pool_transient_unmeasured_width"] += 1
        # The counter above counts *readings* at an unmeasured width, and a reading binds nothing -- a width is chosen a handful of times a run, the price is read tens of thousands.
        # `pool_width_decisions` counts the DECISIONS, and is the responsiveness witness: it read 0 for weeks because they landed on a second pool instance for the same card.
        # Set, not accumulated: the pool owns the running totals and these observations are summed across jobs.
        report = pool.bootstrap_pricing_report()
        subject.scheduler_observations["pool_width_decisions"] = int(
            report["width_decisions"]
        )
        # `diverged` is the answer the decision counter can only pose: declarations where pricing from measurements would have chosen a DIFFERENT width.
        # diverged==0 with outcomes>0 is the real "harmless" reading.
        subject.scheduler_observations["pool_width_outcomes"] = int(
            report["width_outcomes"]
        )
        subject.scheduler_observations["pool_width_outcomes_diverged"] = int(
            report["width_outcomes_diverged"]
        )
        subject.scheduler_observations["pool_width_outcome_worst_blocks"] = int(
            report["width_outcome_worst_blocks"]
        )
        # How often a row couldn't be given a page + by how much the supply fell short -- the measurement that would justify declaring a larger one.
        # Delta against this worker's own baseline: these observations are summed across jobs and the source is a running total.
        events, blocks_short = _kv_pool_shortfall_totals(device)
        seen_events, seen_blocks = self._kv_shortfall_baseline.get(device, (0, 0))
        if events != seen_events or blocks_short != seen_blocks:
            self._kv_shortfall_baseline[device] = (events, blocks_short)
            subject.scheduler_observations["kv_pool_shortfall_events"] += (
                events - seen_events
            )
            subject.scheduler_observations["kv_pool_shortfall_blocks"] += (
                blocks_short - seen_blocks
            )
        # A2e's ceiling, sampled: how much of the supply sits above the highest
        # lent page and could therefore be handed back to the device while rows
        # are still running. Three sums rather than a min, because these are
        # added across jobs; `tail_free / total` over the samples is the mean
        # releasable fraction and `samples` makes it readable.
        pool_total, pool_tail = _kv_pool_tail_totals(device)
        if pool_total:
            subject.scheduler_observations["kv_pool_tail_samples"] += 1
            subject.scheduler_observations["kv_pool_tail_total_bytes"] += pool_total
            subject.scheduler_observations["kv_pool_tail_free_bytes"] += pool_tail
            # The mean above says what is *usually* releasable; A2e needs the
            # tail free at the one instant separation asks, which a mean cannot
            # bound. The histogram carries the low end, in the same 64 MiB
            # buckets the reserve readings use, and needs no threshold to be
            # chosen in advance.
            self._observe_scheduler_histogram(
                subject, "kv_pool_tail_free_mib", pool_tail // (64 * 1024**2) * 64
            )

    def _pool_view(
        self,
        device: str,
        runs: dict[int, _PersistentRun[_ModelT]],
        *,
        reclaimable_bytes: int = 0,
    ) -> PoolView:
        """One immutable device reading, and nothing else."""
        pool = self._pool(device)
        priced = _device_is_cuda(device) and any(
            candidate.key.device == device
            and (
                candidate.arena_cost_probe is not None
                or candidate.arena_cost is not None
            )
            for candidate in runs.values()
        )
        if not priced:
            return pool.view(free_bytes=None, total_bytes=None)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(device))
        except (RuntimeError, ValueError):
            for candidate in runs.values():
                if candidate.key.device == device:
                    candidate.scheduler_observations[
                        "pool_memory_query_failures"
                    ] += 1
            return pool.view(free_bytes=0, total_bytes=0)
        self._capacity_memory_info = (int(free_bytes), int(total_bytes))
        self._observe_device_reserve_inputs(device, runs)
        self._observe_pool_reading(
            device, runs, free_bytes=int(free_bytes), total_bytes=int(total_bytes)
        )
        return pool.view(
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
            reclaimable_bytes=reclaimable_bytes,
            resident_lanes=frozenset(
                candidate.key.model
                for candidate in runs.values()
                if candidate.key.device == device and candidate.session is not None
            ),
            resident_rows=_resident_rows(device, runs),
        )

    def _observe_device_reserve_inputs(
        self,
        device: str,
        runs: dict[int, _PersistentRun[_ModelT]],
    ) -> None:
        """Measure the two things the pool cannot price, under steady work.

        Neither is inferred from something correlated with it:

        * ``max_memory_allocated - memory_allocated`` is how far above its
          steady allocation this process has gone while executing -- the
          headroom an operation needs above its arena.
        * ``inactive_split_bytes`` is free space trapped inside allocator
          segments, which the allocator counts as free and an arena cannot
          use.

        *When* it's sampled matters: right after an arena closes the steady allocation drops while the peak does not, so the difference balloons into the whole arena and pins the reserve at its ceiling
        -- same failure as too small a reserve. A session actively decoding is the steady vantage.
        """
        steady = any(
            candidate.key.device == device
            and candidate.session is not None
            and candidate.session.active_count > 0
            for candidate in runs.values()
        )
        if not steady:
            return
        resolved = torch.device(device)
        pool = self._pool(device)
        rows = _resident_rows(device, runs)
        if pool.begin_transient_measurement():
            # First steady moment on this device. Everything the peak holds now was allocated getting here (model load, arena construction) and none of it recurs,
            # so charging the reserve for it forever charges for finished work.
            # Rebase, sample from the next call.
            with contextlib.suppress(RuntimeError, ValueError):
                torch.cuda.reset_peak_memory_stats(resolved)
            pool.begin_width_measurement(rows)
            return
        if pool.begin_width_measurement(rows):
            # Width changed => the peak belongs to the old one. Carry it over and every later reading for this row count is really the widest arena that ever ran -- a number that can't tell a flat cost from a latched one.
            # Rebase, no sample from this call, same as the first steady moment.
            with contextlib.suppress(RuntimeError, ValueError):
                torch.cuda.reset_peak_memory_stats(resolved)
            return
        try:
            transient = int(torch.cuda.max_memory_allocated(resolved)) - int(
                torch.cuda.memory_allocated(resolved)
            )
            stats = torch.cuda.memory_stats(resolved)
        except (RuntimeError, ValueError, KeyError):
            return
        fragmentation = int(stats.get("inactive_split_bytes.all.current", 0))
        pool.observe_transient(transient)
        pool.observe_fragmentation(fragmentation)
        # Against DECLARED rows, not active: the transient scales with the per-row static state a decode step touches, and that exists as soon as the arena does.
        # Stored against this row count, not divided by it -- see `observe_transient_at_width`.
        pool.observe_transient_at_width(transient, rows)

    def _admittable_count(self, run: _PersistentRun[_ModelT]) -> int:
        """Rows this run may take now, under **both** conservation laws."""
        available = run.logical_available_count
        rows = self._run_page_rows(run)
        return available if rows is None else min(available, rows)

    def _run_page_rows(self, run: _PersistentRun[_ModelT]) -> int | None:
        """Rows this lane's page supply can still fund, or None when unpriced.

        One definition, three readers: legality refuses admission at zero, `_persistent_run_score` projects the width it can reach, and the admission cohort is capped by it.
        Bounding only the executor was tried and measured nothing -- the ranking still promised a wide admission the executor couldn't make, so the scheduler kept picking a run that then admitted one row.
        """
        if run.session is None:
            return None
        rows = _kv_row_capacity(run.key.device, run.key.model)
        return None if rows < 0 else rows

    def _run_floor_blocks(self, run: _PersistentRun[_ModelT]) -> int:
        """Blocks this run's own lane keeps, which it never reserves against."""
        cost = self._pool(run.key.device).lane(run.key.model)
        return 0 if cost is None else blocks_for(cost.floor_bytes)

    def _width_bound(
        self,
        run: _PersistentRun[_ModelT],
        cost: ArenaCost,
        available_blocks: int,
        *,
        ceiling: int,
    ) -> _WidthBound:
        """Width bounded by device bytes **and** by the KV supply, both kept.

        Separate resources, neither bounds the other: a row costs device bytes for cursors and sampling tables, and pages for its KV, and running out of either stops it.
        One `min`, in one place, so no caller solves half the problem.

        Both halves returned, not just the `min` -- the `min` alone can't say which resource binds, and the two call for opposite work (KV-bound => resize the page pool, byte-bound => the arena).
        """
        rows = _kv_fundable_rows(
            run.key.device,
            run.key.model,
            available_blocks,
            self._pool(run.key.device).marginal_transient_per_row_bytes(),
        )
        return _WidthBound(
            bytes_width=width_that_fits(cost, available_blocks, ceiling=ceiling),
            kv_width=ceiling if rows < 0 else rows,
        )

    def _width_that_fits_both(
        self,
        run: _PersistentRun[_ModelT],
        cost: ArenaCost,
        available_blocks: int,
        *,
        ceiling: int,
    ) -> int:
        return self._width_bound(
            run, cost, available_blocks, ceiling=ceiling
        ).width

    def _own_credits(self, run: _PersistentRun[_ModelT]) -> dict[str, int]:
        """What this question's asker already holds a place for, assembled once.

        Eleven sites used to build these by hand and five different subsets appeared. Every term adds to the *asker's* budget, never the device's, so an omission isn't unsafe
        -- it makes that site quietly more conservative than admission, and the same device state then answers "blocked" on one path and "admitted" on another.

        One term left. The `owner` parameter selected the second and had to be required (`None` was a real answer), because a promise is valid only for the task granted it.
        Both the promise and the check that misread it are gone (C4-2b), so the asker's identity no longer changes the answer.
        """
        return {"own_floor_blocks": self._run_floor_blocks(run)}

    @staticmethod
    def _run_session_owner_free(state: SchedulerRunState) -> bool:
        return state.session_exists and not any(
            (
                state.controls,
                state.prefill,
                state.in_flight,
                state.tracked,
                state.active,
                state.occupied,
                state.resident_handles,
                state.control_intents,
                state.producer_waits,
                state.bundle_owned_handles,
            )
        )

    @staticmethod
    def _run_owner_free(run: _PersistentRun[_ModelT]) -> bool:
        """A resident lane holding nothing -- the only state a resize is legal in.

        Direct reading of `_run_session_owner_free`, which asks the same of an assembled `SchedulerRunState`.
        Assembling one costs a device query per run per turn, and this is asked of every run every turn.
        """
        session = run.session
        return session is not None and not (
            session.occupied_count
            or run.prefill_pending
            or run.in_flight
            or run.pending_controls
            or run.resident_handles_by_key
        )

    def _run_affordable_bound(
        self,
        run: _PersistentRun[_ModelT],
        view: PoolView,
    ) -> _WidthBound:
        """The widest this lane could open right now, its own arena given back.

        Separate from `_run_demand_width` because the two are only equal by accident, and one number can't say which is binding
        -- "wants 13" and "can afford 13" call for opposite actions.
        """
        cost = self._effective_cost(run)
        if cost is None or view.unbounded:
            return _WidthBound(run.width, run.width)
        available = view.available_blocks(**self._own_credits(run)) + run.arena_blocks
        # `width_ceiling` is a forced `--batch-size` or nothing. Nothing forced => the byte arithmetic is its own ceiling and the `min` inside `width_that_fits` can't bind.
        # A lane is bounded by its demand and by the device, by no third number.
        ceiling = run.width_ceiling or self._device_ceiling(cost, available)
        return self._width_bound(run, cost, available, ceiling=ceiling)

    @staticmethod
    def _device_ceiling(cost: ArenaCost, available_blocks: int) -> int:
        """The widest the free blocks could cover, used where no ceiling exists."""
        return max(
            cost.minimum_width,
            max(0, available_blocks) * BLOCK_BYTES // max(1, cost.row_bytes),
        )

    def _run_affordable_width(
        self,
        run: _PersistentRun[_ModelT],
        view: PoolView,
    ) -> int:
        return self._run_affordable_bound(run, view).width

    def _run_demand_width(
        self,
        run: _PersistentRun[_ModelT],
        view: PoolView | None = None,
    ) -> int:
        """The width this lane should hold: what it used, bounded by the device.

        The one definition. `affordable` adds this lane's own arena back to what the pool has free -- the blocks it already holds aren't a bound on itself,
        since a resize releases them before claiming new ones. The question is what fits with this arena GONE, not what fits beside it.
        """
        cost = run.arena_cost
        if cost is None:
            return run.width
        if view is None:
            view = self._live_pool_view(run.key.device)
        if view.unbounded:
            return run.width
        affordable = self._run_affordable_width(run, view)
        if run.width_funded_ceiling >= 1:
            affordable = min(affordable, run.width_funded_ceiling)
        target = demand_width(
            held=run.width,
            demand_high_water=run.demand_high_water,
            minimum_width=max(1, cost.minimum_width),
            affordable=affordable,
        )
        # The floor is this lane's own and the pool holds it for exactly this lane, so `demand_width` can return a width the *delta* still can't buy when the device is oversubscribed.
        # Refuse to move rather than name a target legality won't honour.
        delta = self._arena_blocks(run, target) - run.arena_blocks
        available = view.available_blocks(**self._own_credits(run))
        if delta > available:
            return run.width
        # Never below the width already held. A narrowing returns the arena bytes and nothing else -- the KV supply, most of a row, is left alone because re-declaring it calls `torch.cuda.empty_cache()`.
        # So it trades negligible bytes against the ability to grow, and growth needs an empty slot set a busy lane doesn't reach.
        # Measured: a lane narrowed 34 -> 1 at t=4s, wanted 2 rows at t=7s, couldn't have them for the remaining 18s. Worth doing again only when narrowing re-declares the supply.
        return max(run.width, target)

    def _effective_cost(self, run: _PersistentRun[_ModelT]) -> ArenaCost | None:
        """This lane's row price, at the number every other site charges.

        `measured_row_bytes` beats the analytic price whenever larger -- the analytic one leaves the condition slots out on purpose.
        The *width* used to be solved at the analytic price while the *blocks* were charged at the measured one, so a width that fit produced a block delta that didn't:
        the lane was then suppressed for a resize legality refused, with no admission either. One price, both questions.
        """
        cost = run.arena_cost
        if cost is None:
            return None
        # Plus the transient this row will cause. Not an allocation the arena makes, which is why it was never here -- it's the peak a decode step reaches above steady state, and it scales with declared rows.
        # Charging it to the width is the only way a width is chosen against the memory it will actually need, not against a water line measured narrower that then rises behind the commitment (R13).
        # MARGINAL figure, because this prices a row about to be added: a slope between two measured widths, never a total / rows.
        # The total has a fixed term, so dividing overstates one more row most at narrow widths -- exactly where a lane is deciding whether it may grow.
        row_bytes = (
            max(cost.row_bytes, run.measured_row_bytes)
            + self._pool(run.key.device).marginal_transient_per_row_bytes()
        )
        if row_bytes == cost.row_bytes:
            return cost
        return ArenaCost(
            row_bytes=row_bytes,
            minimum_width=cost.minimum_width,
            maximum_width=cost.maximum_width,
        )

    @staticmethod
    def _arena_bytes(run: _PersistentRun[_ModelT], width: int | None = None) -> int:
        """What this run's arena occupies, at the width it will actually open.

        `measured_row_bytes` is what a live session reported for this run's own
        rows; it beats the analytic price whenever it is larger, because the
        analytic price deliberately leaves the condition slots out. Taking the
        maximum keeps a live workload from under-charging itself without ever
        letting an unmeasured run charge less than it costs.
        """
        cost = run.arena_cost
        if cost is None:
            return 0
        rows = run.width if width is None else width
        return max(cost.row_bytes, run.measured_row_bytes) * max(1, rows)

    def _arena_blocks(
        self,
        run: _PersistentRun[_ModelT],
        width: int | None = None,
    ) -> int:
        return blocks_for(self._arena_bytes(run, width))

    def _nonresident_arena_admission_safe(
        self,
        run: _PersistentRun[_ModelT],
        pool: PoolView,
    ) -> bool:
        """One comparison: does this arena, plus what it owes, fit right now?

        Everything the old design proved separately -- fixed-width arena against a device snapshot, then a banker's safe sequence over every outstanding dependency obligation -- is one sum against free blocks.
        Safe *because* nothing else ever asks: only an arena draws device blocks, and a recovery bundle's members run inside an arena that already exists, drawing pages instead (R1).
        One claimant on this resource => no admitted holder can wait on memory another admitted holder must release first.
        """
        if run.session is not None or not run.pending:
            return True
        if not self._run_capacity_ready(run):
            return False
        if run.arena_cost is None or pool.unbounded:
            return True
        return self._nonresident_arena_blocks(run, pool) is not None

    def _nonresident_arena_blocks(
        self,
        run: _PersistentRun[_ModelT],
        pool: PoolView,
    ) -> int | None:
        """Blocks this run would take if admitted now, or `None` if refused.

        Width is re-derived against `pool`, not read off `run.width`: that value was resolved before any co-resident lane opened,
        and a run pinned to it stays unadmittable forever once the device fills -- the only place that narrows it is the session creation admission refuses to reach.
        """
        cost = run.arena_cost
        if cost is None:
            return 0
        credits = self._own_credits(run)
        width = self._width_that_fits_both(
            run,
            cost,
            pool.available_blocks(**credits),
            ceiling=run.width_ceiling or run.width,
        )
        if width < cost.minimum_width:
            return None
        needed = blocks_for(max(cost.row_bytes, run.measured_row_bytes) * width)
        return (
            needed
            if pool.admits(needed, **credits)
            else None
        )

    def _reclaim_admission_states(
        self,
        runs: dict[int, _PersistentRun[_ModelT]],
        run_states: tuple[SchedulerRunState, ...],
        views: dict[str, PoolView],
    ) -> tuple[SchedulerReclaimAdmissionState, ...]:
        """Prove atomic owner-free arena reclamation for blocked admission."""
        states_by_id = {state.run_id: state for state in run_states}
        candidates: list[SchedulerReclaimAdmissionState] = []
        for target_id, target in runs.items():
            target_state = states_by_id[target_id]
            view = views[target.key.device]
            if (
                view.unbounded
                or target_state.session_exists
                or not target_state.pending
                or not target_state.capacity_ready
            ):
                continue
            donor_ids = tuple(
                run_id
                for run_id, donor in runs.items()
                if run_id != target_id
                and donor.key.device == target.key.device
                and self._run_session_owner_free(states_by_id[run_id])
            )
            if not donor_ids:
                continue
            reclaimable = sum(
                runs[run_id].capacity_total_bytes for run_id in donor_ids
            )
            donor_identities = {runs[run_id].key.identity for run_id in donor_ids}
            resident_ids_by_identity: dict[ModelIdentity, set[int]] = {}
            for run_id, run in runs.items():
                if run.key.device == target.key.device and run.session is not None:
                    resident_ids_by_identity.setdefault(run.key.identity, set()).add(
                        run_id
                    )
            for identity in donor_identities:
                if (
                    identity == target.key.identity
                    or not resident_ids_by_identity[identity].issubset(donor_ids)
                ):
                    continue
                cached = self._models.get(identity)
                if cached is None:
                    continue
                current_device = getattr(cached.model, "_device", None)
                if current_device is None or str(current_device) != target.key.device:
                    continue
                reclaimable += cached.weight_storage_bytes
            reclaimable_blocks = blocks_for(reclaimable)
            if not reclaimable_blocks:
                continue
            if self._nonresident_arena_blocks(target, view) is not None:
                # already admissible without a donor -> reclaiming buys nothing
                continue
            projected = replace(
                view,
                reclaimable_blocks=view.reclaimable_blocks + reclaimable_blocks,
            )
            needed = self._nonresident_arena_blocks(target, projected)
            if needed is None:
                continue
            shortfall = max(
                1,
                needed
                - view.available_blocks(**self._own_credits(target)),
            )
            candidates.append(
                SchedulerReclaimAdmissionState(
                    target_run_id=target_id,
                    donor_run_ids=donor_ids,
                    reclaimable_blocks=reclaimable_blocks,
                    required_blocks=shortfall,
                )
            )
        return tuple(candidates)

    def _scheduler_state(
        self,
    ) -> tuple[SchedulerState, dict[int, _PersistentRun[_ModelT]]]:
        started_at = time.perf_counter()
        runs = dict(enumerate(self._persistent_runs.values()))
        run_id_by_identity = {id(run): run_id for run_id, run in runs.items()}
        views = self._device_pool_views(runs)
        resources_finished_at = time.perf_counter()
        external_now = time.perf_counter()
        with self._state_lock:
            bundles = tuple(self._dependency_bundles.values())
            bundle_owned_handle_keys = tuple(
                handle_key
                for bundle in bundles
                for handle_key in (
                    *(
                        ()
                        if bundle.parent_handle_key is None
                        else (bundle.parent_handle_key,)
                    ),
                    *(
                        self._resident_handle_key(handle)
                        for handle in bundle.resident_handles
                        if handle is not None
                    ),
                )
            )
            producer_wait_snapshot = tuple(
                (wait.handle_key, wait.run_identity, wait.future is None)
                for wait in self._producer_waits.values()
            )
            producer_wait_keys = frozenset(
                handle_key
                for handle_key, _run_identity, _unbound in producer_wait_snapshot
            )
            # Mirrors `liveness_model.bundle_transaction_active`; the two
            # must agree or the proof stops covering the runtime. The note
            # there carries the measurement that keeps the rule.
            active_bundle = any(
                bundle.member_sequences and not bundle.published
                for bundle in bundles
            )
            bundle_states = tuple(
                SchedulerBundleState(
                    bundle_id=bundle.bundle_id,
                    run_id=run_id_by_identity[bundle.run_identity],
                    # Serialisation, nothing else. This also asked the pool whether members were affordable *now*; they cost `blocks_for(0)` since C4-1, so the question was `admits(0)`
                    # -- true on a device with nothing free, which is not a gate. A member needs pages and takes those when it runs, from a pool that guarantees progress rather than supply.
                    # Admitting a bundle commits no device memory, so no device reading can make it unsafe.
                    admission_safe=not active_bundle,
                )
                for bundle in bundles
                if not bundle.member_sequences
            )
            external_wait_states = tuple(
                SchedulerExternalWaitState(
                    owner_id=("handle", wait.handle_key),
                    run_id=run_id_by_identity[wait.run_identity],
                    phase=wait.phase,
                    failed=wait.future_failure is not None,
                    expired=external_now >= wait.deadline_at,
                )
                for wait in self._producer_waits.values()
            ) + tuple(
                SchedulerExternalWaitState(
                    owner_id=("claim", claim_id),
                    run_id=run_id_by_identity[claim.run_identity],
                    phase=claim.phase,
                    failed=claim.future_failure is not None,
                    expired=external_now >= claim.deadline_at,
                )
                for claim_id, claim in self._transitive_claims.items()
            )
        ownership_finished_at = time.perf_counter()
        if len(bundle_owned_handle_keys) != len(set(bundle_owned_handle_keys)):
            raise SchedulerInvariantError(
                "resident handle belongs to multiple dependency bundles"
            )
        resident_handle_keys = {
            handle_key
            for run in runs.values()
            for handle_key in run.resident_handles_by_key
        }
        bundle_owned_handle_key_set = frozenset(bundle_owned_handle_keys)
        if not bundle_owned_handle_key_set <= resident_handle_keys:
            raise SchedulerInvariantError(
                "dependency bundle owns a nonresident handle"
            )
        if bundle_owned_handle_key_set & producer_wait_keys:
            raise SchedulerInvariantError(
                "dependency bundle handle also has a producer wait"
            )
        run_phase_seconds: dict[str, float] = {}
        run_states = tuple(
            self._scheduler_run_state(
                run,
                run_id=run_id,
                pool=views[run.key.device],
                bundle_owned_handle_keys=bundle_owned_handle_key_set,
                producer_wait_snapshot=tuple(
                    (handle_key, unbound)
                    for handle_key, run_identity, unbound in producer_wait_snapshot
                    if run_identity == id(run)
                ),
                phase_seconds=run_phase_seconds,
            )
            for run_id, run in runs.items()
        )
        runs_finished_at = time.perf_counter()
        reclaim_admissions = self._reclaim_admission_states(
            runs,
            run_states,
            views,
        )
        contracts_finished_at = time.perf_counter()
        # Split before comparing. Every field named in `_MEASURED_RUN_FIELDS`
        # and its siblings is a function of a live `mem_get_info` reading, so
        # folding them into `version` lets a background process on a shared GPU
        # move the version with no scheduler action behind it -- which is
        # exactly what broke the REJECTED contract at `model_workers.py:1854`.
        decided_signature = (
            tuple(_decided_run_signature(run) for run in run_states),
            tuple((bundle.bundle_id, bundle.run_id) for bundle in bundle_states),
            external_wait_states,
            tuple(
                (admission.target_run_id, admission.donor_run_ids)
                for admission in reclaim_admissions
            ),
        )
        measured_signature = (
            tuple(_measured_run_signature(run) for run in run_states),
            tuple(
                (bundle.bundle_id, bundle.admission_safe)
                for bundle in bundle_states
            ),
            tuple(
                (
                    admission.target_run_id,
                    admission.reclaimable_blocks,
                    admission.required_blocks,
                )
                for admission in reclaim_admissions
            ),
        )
        if decided_signature != self._last_state_signature:
            self._state_version += 1
            self._last_state_signature = decided_signature
            self._rejected_actions.clear()
        if measured_signature != self._last_measured_signature:
            # No `_rejected_actions.clear()`: the epoch is part of the memo key,
            # so a changed measurement retires exactly the rejections that were
            # caused by the old reading and leaves the rest standing.
            self._measurement_epoch += 1
            self._last_measured_signature = measured_signature
        state = SchedulerState(
            version=self._state_version,
            measurement_epoch=self._measurement_epoch,
            progress_epoch=self._progress_epoch,
            runs=run_states,
            bundles=bundle_states,
            external_waits=external_wait_states,
            reclaim_admissions=reclaim_admissions,
        )
        finished_at = time.perf_counter()
        self._last_snapshot_phase_seconds = {
            "resources": resources_finished_at - started_at,
            "ownership": ownership_finished_at - resources_finished_at,
            "runs": runs_finished_at - ownership_finished_at,
            "contracts": contracts_finished_at - runs_finished_at,
            "signature": finished_at - contracts_finished_at,
            **{
                f"runs_{name}": seconds
                for name, seconds in run_phase_seconds.items()
            },
        }
        return state, runs

    def _validate_claim_lineage(self) -> None:
        if not hasattr(self, "_handle_claim_tokens"):
            return
        with self._state_lock:
            expected: dict[tuple[int, int, int], DependencyClaimToken] = {}
            for claim_id, owner in self._transitive_claims.items():
                if owner.handle_key is None:
                    continue
                if owner.handle_key in expected:
                    raise SchedulerInvariantError(
                        "multiple dependency claims own one resident handle"
                    )
                expected[owner.handle_key] = DependencyClaimToken(claim_id)
            if self._handle_claim_tokens != expected:
                raise SchedulerInvariantError(
                    "dependency claim lineage index does not match live claims"
                )

    _resident_handle_key = staticmethod(resident_handle_key)

    @staticmethod
    def _task_requires_dependency_credit(task: _BatchTask[_ModelT, Any]) -> bool:
        return bool(
            getattr(
                getattr(task, "item", None),
                "requires_dependency_credit",
                False,
            )
        )

    def _live_pool_view(self, device: str) -> PoolView:
        """A pool reading taken outside a scheduler snapshot."""
        return self._pool_view(
            device,
            dict(enumerate(self._persistent_runs.values())),
        )

    def _preempt_condition_resources_fit(
        self,
        run: _PersistentRun[_ModelT],
        *,
        refresh_device: bool,
    ) -> bool:
        if run.arena_cost is None or not _device_is_cuda(run.key.device):
            return True
        if not run.pending:
            return False
        memory_info = self._capacity_memory_info
        if refresh_device or memory_info is None:
            try:
                memory_info = torch.cuda.mem_get_info(torch.device(run.key.device))
            except (RuntimeError, ValueError):
                run.scheduler_observations["pool_memory_query_failures"] += 1
                return False
            self._capacity_memory_info = (int(memory_info[0]), int(memory_info[1]))
        free_bytes, total_bytes = map(int, memory_info)
        view = self._pool(run.key.device).view(
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            resident_lanes=self._resident_lane_models(run.key.device),
            resident_rows=_resident_rows(
                run.key.device, dict(enumerate(self._persistent_runs.values()))
            ),
        )
        # Freeing the slot costs **nothing** now. It used to allocate a second
        # GPU copy of the parked row, and this check priced that allocation
        # against the device; preemption releases the row's pages instead and
        # rebuilds it from its own tokens on resume. What is left to afford is
        # the claim the freed slot is being made for.
        allowed = view.admits(0, **self._own_credits(run))
        if not allowed:
            run.scheduler_observations["preempt_condition_pool_blocked"] += 1
        return allowed

    def _resident_lane_models(self, device: str) -> frozenset[str]:
        """Lanes on this device that already hold an arena."""
        return frozenset(
            run.key.model
            for run in self._persistent_runs.values()
            if run.key.device == device and run.session is not None
        )

    def _run_capacity_ready(self, run: _PersistentRun[_ModelT]) -> bool:
        """A lane with nothing to price needs no preparation step.

        An unpriced lane -- no probe, or no CUDA device -- is ready by
        definition: there is no row cost to measure and the pool never bounds
        it. Treating it as unprepared makes `CAPACITY_PREPARE` the preferred
        action forever on every CPU run.
        """
        return (
            run.session is not None
            or run.capacity_resolved
            or run.arena_cost_probe is None
            or not _device_is_cuda(run.key.device)
        )

    def _resolve_arena_width(
        self,
        run: _PersistentRun[_ModelT],
        view: PoolView | None = None,
    ) -> int:
        """The width this lane opens at: the same rule a resize would apply.

        One rule, both moments. Opening at "the largest whole number of rows the free blocks cover" starves the second lane once ceilings come off (R11)
        -- whoever opens first takes the device and never gives it back.
        A lane opens at the width it has work for, bounded by what the device can give -- exactly what `_run_demand_width` answers for a live lane.

        This does mean a lane opens NARROW when its work hasn't arrived: measured, the bootstrap lane has 1 job queued when its session opens and 15 two seconds later.
        That's correct information; growing on it is the resize driver's job, which is why the two ship together.

        The arena is sized against what's left AFTER the row's own obligations. Sizing against the raw budget produced an arena that fit and a run that couldn't be admitted,
        because the admission test also charges the checkpoint credit and the whole dependency claim -- the run then had no legal action and the device stopped with memory free.
        """
        if run.arena_cost is None:
            return run.width
        if view is None:
            view = self._live_pool_view(run.key.device)
        if view.unbounded:
            return run.width
        # Opened at what the device affords, and the supply is declared from the SAME number. That equality is load-bearing:
        # admission is bounded by the arena's free slots and not by the page pool, so an arena wider than its supply admits a row the pool can't fund -> `KVPagesExhausted`.
        # Opening at demand and growing into it was tried and fails exactly there, because a grown arena and a re-declared supply are two numbers that can diverge.
        # Bounding admission by the pool is what would make opening at demand possible, and it isn't built. Until it is, these stay one number.
        return self._run_affordable_width(run, view)

    def _prepare_run_capacity(
        self,
        run: _PersistentRun[_ModelT],
        model: _ModelT,
    ) -> None:
        """Ask the loaded model what a row costs, and declare the lane.

        Everything that survives `_ensure_initial_capacity`: one measurement, no probe ladder, no stored calibration to load or invalidate, no bootstrap bypass.
        An environment that has measured nothing is in exactly the position of one that has measured everything -- nothing is remembered between processes.
        """
        if self._run_capacity_ready(run):
            raise SchedulerInvariantError("capacity preparation selected for ready run")
        probe = run.arena_cost_probe
        declared = self._pool(run.key.device).lane(run.key.model)
        if probe is None or not _device_is_cuda(run.key.device):
            run.capacity_resolved = True
            run.scheduler_observations["capacity_prepare_actions"] += 1
            return
        cost = ArenaCost(
            row_bytes=probe(model, run.capacity, run.key.model).row_bytes,
            minimum_width=max(1, run.minimum_width),
            # a lane already declared on the pool keeps its band -- a submission may not widen what residency said
            maximum_width=(
                run.maximum_width
                if declared is None
                else declared.maximum_width
                if run.maximum_width is None
                else min(run.maximum_width, declared.maximum_width or run.maximum_width)
            ),
        )
        run.arena_cost = cost
        self._pool(run.key.device).declare_lane(run.key.model, cost)
        run.width = self._resolve_arena_width(run)
        run.capacity_resolved = True
        run.scheduler_observations["capacity_prepare_actions"] += 1
        run.scheduler_observations["arena_row_bytes"] = cost.row_bytes
        run.scheduler_observations["arena_pool_width"] = run.width
        # P4's verdict is NOT recorded here. Capacity preparation runs at t=0, before this lane has seen work, so a high-water read now is always its starting value
        # -- measured at 1 for a lane that went on to want 13. It goes in the per-turn histogram instead.
        run.scheduler_observations["arena_lane_maximum_width"] = (
            cost.maximum_width or 0
        )
        # What the arena was solved against, beside what it resolved to. A pool sized from arena demand has to know whether the width came out at the device's limit or the lane's ceiling:
        # only in the first case does "everything available" equal "no greedier than the arenas would have been". Indistinguishable from `arena_pool_width` alone => measured, not argued.
        # Declare the KV supply *after* the width is resolved, because the rule
        # is "no greedier than this arena would have been" and the width is the
        # only half of that the model cannot work out for itself. Model-side:
        # a page's geometry is the model's business, exactly as pricing a row
        # is. A lane with no declarator leaves its arenas private.
        if run.kv_pool_declaration is not None and run.width > 0:
            # **Sized from the demand, and the width then clamped to what it
            # funded.** Sizing the supply from the width it happens to open at
            # closes a circle: the pool funds `width + 1` rows, that is read
            # back as the affordable width, the target can never exceed it, no
            # resize is selected, and the pool is never re-declared. Reached in
            # production, with the device able to fund thousands of rows and
            # the pool answering **1** (R12).
            funded = self._declare_kv_rows(run, model, run.width)
            if funded is not None and funded >= 1:
                run.width = min(run.width, funded)
        run.scheduler_observations["arena_available_blocks"] = (
            self._live_pool_view(run.key.device).available_blocks(
                **self._own_credits(run)
            )
        )

    def _declare_kv_rows(
        self,
        run: _PersistentRun[_ModelT],
        model: _ModelT,
        width: int,
    ) -> int | None:
        """Re-size this lane's KV supply, and report the rows it actually funded.

        The declaration is sized against a live device reading and may come
        back **smaller** than asked. Nothing used to read that back, so an
        arena was built at the requested width against a supply that could not
        serve it -- `KVPagesExhausted: 32 pages wanted per layer, 2 available`,
        with the arena already committed. The number returned here is what the
        caller opens at.

        The supply is asked for the lane's **demand**, never for the width it
        happens to hold: sizing it from the width is what made the first
        declaration final (R12).
        """
        declare = run.kv_pool_declaration
        if declare is None:
            return None
        wanted = max(1, width, run.demand_high_water)
        role = (
            "recovery" if "recovery" in self._run_job_types(run) else "primary"
        )
        declared_bytes = int(
            declare(model, wanted, run.capacity, run.key.model, role, run.key.device)
        )
        run.scheduler_observations["kv_pool_declared_bytes"] = declared_bytes
        row_bytes = _kv_lane_row_bytes(run.key.device, run.key.model)
        if row_bytes < 1:
            return None
        funded = declared_bytes // row_bytes
        self._observe_scheduler_histogram(run, "kv_pool_funded_rows", funded)
        return funded

    def _resize_run_width(self, run: _PersistentRun[_ModelT], model: _ModelT) -> bool:
        """Move one lane to the width its demand asks for. Returns False if stale.

        The KV supply is re-declared ONLY when growing, and only from inside `resize_width`, between the old arena being dropped and the new one built -- the single moment nothing borrows the pool.
        A narrowing lane leaves its supply alone: pages it no longer needs cost nothing until another lane wants the bytes, and re-declaring calls `torch.cuda.empty_cache()`, which synchronises the whole device.
        """
        session = run.session
        if session is None:
            raise SchedulerInvariantError("resize selected for a lane with no session")
        resize = getattr(session, "resize_width", None)
        if not callable(resize):
            raise SchedulerInvariantError("session cannot change its width")
        before = run.width
        target = self._run_demand_width(run)
        if target == before:
            return False
        declare = run.kv_pool_declaration
        grow = (
            None
            if declare is None or target <= before
            else lambda width: self._declare_kv_rows(run, model, width)
        )
        resize(target, on_released=grow)
        # Read back, never assume: the session may open narrower than asked when the supply came back smaller, and a `run.width` disagreeing with the session's slots fails
        # `physical availability is inconsistent` on the next snapshot.
        run.width = int(getattr(session, "width", target))
        target = run.width
        if target == before:
            # supply came back at the width the lane already had -> nothing moved.
            # Record it as the funding ceiling and report stale; without that the same target is selected every turn and the scheduler spins on an action that can't progress.
            run.width_funded_ceiling = target
            run.scheduler_observations["resize_underfunded"] += 1
            return False
        run.width_funded_ceiling = 0
        self._refresh_capacity_measurement(run)
        run.arena_blocks = self._arena_blocks(run)
        # The high-water resets HERE and only here -- when the width it was measured against actually changed. It's the lane's memory of its last peak; clearing it anywhere else throws that away exactly when it's needed.
        # Resetting every owner-free turn was tried and measured: owner-free is precisely when the lane just finished everything, so the reading there is its LOWEST.
        # The peak of 13 was overwritten with 0 on the turn before the resize, and every width change the run made was a narrowing -- 3 -> 1, twice, against a demand of 13.
        # The lane is owner-free by legality, so committed prefill and decode are 0 and the queue is the whole demand. It climbs again with `max`, which lets a quiet phase narrow the lane with no decay constant.
        run.demand_high_water = len(run.pending)
        run.scheduler_observations["resize_actions"] += 1
        run.scheduler_observations[
            "resize_grew" if target > before else "resize_narrowed"
        ] += 1
        self._observe_scheduler_histogram(run, "resize_from_width", before)
        self._observe_scheduler_histogram(run, "resize_to_width", target)
        return True

    @staticmethod
    def _refresh_capacity_measurement(run: _PersistentRun[_ModelT]) -> None:
        if run.session is None:
            run.capacity_resident_bytes = 0
            run.capacity_graph_static_bytes = 0
            run.capacity_total_bytes = 0
            return
        report = getattr(run.session, "capacity_telemetry", None)
        if not callable(report):
            return
        values = dict(report())
        run.capacity_resident_bytes = int(values.get("resident_bytes", 0))
        run.capacity_graph_static_bytes = int(values.get("graph_static_bytes", 0))
        run.capacity_total_bytes = int(values.get("total_bytes", 0))
        measured = int(values.get("estimated_row_bytes", 0))
        if measured > 0:
            # live all-in row cost, conditions and masks included. What a later arena on this lane is sized from, and never written anywhere that outlives the process.
            run.measured_row_bytes = measured

    @staticmethod
    def _preferred_actions(
        actions: tuple[SchedulerAction, ...],
    ) -> dict[int, SchedulerAction]:
        """Apply run-local policy only after pure legality is established."""
        order = {
            SchedulerActionKind.CANCEL: 0,
            SchedulerActionKind.DISCARD: 1,
            SchedulerActionKind.CAPACITY_PREPARE: 2,
            # Beside capacity preparation, same reason: both settle how many rows this lane holds before anything is admitted into them.
            # Above admission deliberately -- a lane that admits first never reaches the empty slot set a resize needs, so below `condition_prefill` growth is unreachable for any lane that has work.
            SchedulerActionKind.RESIZE: 2,
            SchedulerActionKind.RELEASE_ADMIT: 2,
            SchedulerActionKind.BUNDLE_ADMIT: 2,
            SchedulerActionKind.PREFILL: 2,
            SchedulerActionKind.RESTORE: 3,
            SchedulerActionKind.PREEMPT_RESTORE: 4,
            SchedulerActionKind.PREEMPT_CONDITION: 5,
            SchedulerActionKind.CONDITION_BATCH: 7,
            SchedulerActionKind.CONDITION_PREFILL: 8,
            SchedulerActionKind.ADMIT: 9,
            SchedulerActionKind.DECODE: 10,
            SchedulerActionKind.FAIL: 11,
        }
        global_condition = {
            participant: action
            for action in actions
            if action.kind is SchedulerActionKind.CONDITION_BATCH
            for participant in action.participants
        }
        selected: dict[int, SchedulerAction] = {}
        for action in actions:
            if (
                action.kind is SchedulerActionKind.CONDITION_PREFILL
                and action.run_id in global_condition
            ):
                continue
            current = selected.get(action.run_id)
            if current is None or order[action.kind] < order[current.kind]:
                selected[action.run_id] = action
        return selected

    def _preferred_run_action(self, run: _PersistentRun[_ModelT]) -> str:
        state = SchedulerState(
            version=0,
            progress_epoch=0,
            runs=(self._scheduler_run_state(run, run_id=0),),
        )
        selected = self._preferred_actions(enabled_actions(state)).get(0)
        if selected is None:
            raise RuntimeError("persistent run has no enabled action")
        return selected.kind.value

    def _dependency_tier(
        self,
        run: _PersistentRun[_ModelT],
        action: str,
    ) -> tuple[int, int]:
        """Order legal work by dependency release, then completion value."""
        if action in {"cancel", "discard", "restore"}:
            return 0, 0
        if action in {"capacity_prepare", "release_admit", "resize"}:
            return 0, 0
        if action == "bundle_admit":
            return 0, 0
        if action in {"preempt_restore", "preempt_condition"}:
            return 0, 0
        if action == "prefill":
            return 0, 1
        if action == "condition_batch":
            return 1, 1
        # Recovery and control outrank a plain decode; removing that was tried and reverted.
        # The suspicion was that a nearly-empty recovery lane preempts a full primary one. It does -- and removing the tier widened recovery's batches without buying any wall,
        # because you can't batch what doesn't arrive together: a second recovery row has to come from another song, at an unrelated moment.
        # What it did buy was longer dependency latency, which is what this tier prevents (scheduler-fill arms).
        if self._run_job_types(run) & {"recovery", "control"}:
            return 2, 0
        if action == "decode":
            return 3, 0
        return 4, 2

    @staticmethod
    def _observe_scheduler_histogram(
        run: _PersistentRun[_ModelT],
        name: str,
        value: int,
    ) -> None:
        run.stats.scheduler_state_histograms.setdefault(name, Counter())[int(value)] += 1

    def _record_scheduler_snapshot(
        self,
        selected: _PersistentRun[_ModelT],
        runnable: list[_PersistentRun[_ModelT]],
        *,
        action: SchedulerActionKind | None = None,
    ) -> None:
        """Record turn-weighted readiness and cohort-split pressure."""
        observations = selected.scheduler_observations
        observations["turns"] += 1
        observations["ready_run_observations"] += len(runnable)
        selected_action = (
            self._preferred_run_action(selected) if action is None else action.value
        )
        observations[f"selected_action_{selected_action}_observations"] += 1

        condition_ready = 0
        prefill_ready = 0
        decode_ready = 0
        dependency_blocked = 0
        capacity_blocked = 0
        views: dict[str, PoolView] = {}
        for run in self._persistent_runs.values():
            condition, prefill, decode = self._run_phase_depths(run)
            condition_ready += condition
            prefill_ready += prefill
            decode_ready += decode
            if run.session is None:
                continue
            dependency_blocked += max(
                0,
                run.session.occupied_count - run.session.active_count,
            )
            if run.logical_available_count == 0:
                capacity_blocked += len(run.pending)
            # P4-3's premise, measured before anything acts on it. A width change needs every slot empty (same rule as `KVBlockPool.resize`),
            # so whether a lane must be *made* empty or is already empty often enough decides whether a drain mechanism is needed at all.
            # Per lane: the two lanes are in opposite positions and a device-wide sum hides exactly that.
            if not self._run_owner_free(run):
                continue
            run.scheduler_observations["resize_owner_free_turns"] += 1
            # Keyed by the width the lane held: observations are summed across jobs and a scalar can't say WHICH lane was empty.
            # One lane is empty and correctly sized, the other wrongly sized and never empty; a total that mixes them reads as "nothing to do".
            self._observe_scheduler_histogram(
                run, "resize_owner_free_width", run.width
            )
            # What the lane remembered it needed at the one moment it could act on it.
            # A target that never rises above the held width is either genuinely low demand or a high-water that isn't climbing -- opposite fixes.
            self._observe_scheduler_histogram(
                run, "resize_owner_free_demand", run.demand_high_water
            )
            device = run.key.device
            if device not in views:
                views[device] = self._live_pool_view(device)
            if self._run_demand_width(run, views[device]) != run.width:
                run.scheduler_observations["resize_opportunity_turns"] += 1
                self._observe_scheduler_histogram(
                    run, "resize_opportunity_width", run.width
                )

        observations["condition_ready_job_observations"] += condition_ready
        observations["decode_ready_row_observations"] += decode_ready
        observations["dependency_blocked_row_observations"] += dependency_blocked
        observations["capacity_blocked_job_observations"] += capacity_blocked
        # `prefill_ready_job_observations` and `compatibility_split_job_observations` stood here, both 0 across 32 runs and deleted with the conditions they counted.
        # Prefill: the `PREFILL` action has zero spans because `_fill_persistent_run`'s callers drain inside their own action -- the live tripwire is `dispatch_prefill_ready`, kept.
        # Compatibility: length left the run key with the rationing it was there for, so the split it counted cannot occur.

        selected_condition, selected_prefill, selected_active = self._run_phase_depths(
            selected
        )
        available = selected.logical_available_count
        affordable = self._run_affordable_bound(
            selected,
            views.get(selected.key.device)
            or self._live_pool_view(selected.key.device),
        )
        projected_width = selected_active + min(selected_condition, available)
        compatible_ready_width = (
            selected_condition + selected_prefill + selected_active
        )
        for name, value in (
            ("ready_runs", len(runnable)),
            ("condition_ready_jobs", condition_ready),
            ("prefill_ready_jobs", prefill_ready),
            ("decode_ready_rows", decode_ready),
            ("dependency_blocked_rows", dependency_blocked),
            ("selected_pending_jobs", selected_condition),
            ("selected_prefill_jobs", selected_prefill),
            ("selected_active_width", selected_active),
            ("selected_available_slots", available),
            ("selected_physical_width", selected.width),
            ("selected_projected_width", projected_width),
            ("selected_projected_width_gain", projected_width - selected_active),
            ("selected_compatible_ready_width", compatible_ready_width),
            # What P4 would ask this lane to hold, against `selected_physical_width` which is what it does hold.
            # Histogram not scalar, like every width observation here: observations are summed across jobs and a summed maximum is not a maximum.
            # Through `_run_demand_width`, not `demand_width` directly -- the first reading was taken against the lane's ceiling alone, which answers what the lane wants and not what the device can give.
            # Those agree only when memory isn't binding, the one case P4 doesn't have to solve.
            (
                "selected_demand_width",
                self._run_demand_width(
                    selected,
                    views.get(selected.key.device),
                ),
            ),
            # Beside it, because `demand_width` is a `min` and a `min` doesn't report which side won.
            # "Wants 13" and "can afford 13" are the same number and opposite situations: the first says give the rows back, the second says they were never available.
            (
                "selected_demand_high_water",
                selected.demand_high_water,
            ),
            # Split, because the `min` alone can't say which resource binds and the two call for opposite work: KV-bound => resize the page pool, byte-bound => the arena.
            (
                "selected_affordable_width",
                affordable.width,
            ),
            (
                "selected_affordable_bytes_width",
                affordable.bytes_width,
            ),
            (
                "selected_affordable_kv_width",
                affordable.kv_width,
            ),
        ):
            self._observe_scheduler_histogram(selected, name, value)

    @staticmethod
    def _task_prefill_budget(task: _BatchTask[_ModelT, Any]) -> int:
        if getattr(task.item, "resident_handle", None) is not None:
            return 0
        return int(getattr(task.item, "prefill_token_budget", 384))

    @staticmethod
    def _task_decode_budget(task: _BatchTask[_ModelT, Any]) -> int:
        handle = getattr(task.item, "resident_handle", None)
        if handle is not None:
            return int(getattr(handle, "remaining_token_budget", 2048))
        return int(getattr(task.item, "remaining_decode_token_budget", 2048))

    def _persistent_run_score(
        self,
        run: _PersistentRun[_ModelT],
        *,
        action: SchedulerAction | None = None,
        force_decode: bool = False,
    ) -> tuple[object, ...]:
        action_name = (
            self._preferred_run_action(run) if action is None else action.kind.value
        )
        dependency_tier = self._dependency_tier(run, action_name)
        if force_decode and action_name == "decode":
            dependency_tier = (1, 0)
        active = run.session.active_count if run.session is not None else 0
        available = self._admittable_count(run)
        admitted = min(len(run.pending), available)
        projected_width = active + admitted
        pending = self._ordered_pending(run)[:admitted]
        decode_budget = (
            int(
                getattr(
                    run.session,
                    "remaining_decode_token_budget",
                    run.session.active_count * 2048,
                )
            )
            if run.session is not None
            else 0
        ) + sum(self._task_decode_budget(task) for task in pending)
        completion_budget = (
            (decode_budget + projected_width - 1) // projected_width
            if projected_width
            else 0
        )
        prefill_budget = sum(self._task_prefill_budget(task) for task in pending)
        age_bonus = min(run.width, run.age // 4)
        oldest = pending[0].ready_order if pending else ""
        condition_batch_width = (
            len(action.participants)
            if action is not None
            and action.kind is SchedulerActionKind.CONDITION_BATCH
            else 0
        )
        phase_width = max(
            1,
            admitted,
            condition_batch_width,
            sum(len(batch.selected) for batch in run.prefill_pending),
        )
        # Action cost is the work the action is about to do, in the units the
        # run already tracks. It used to be a learned phase-time estimate
        # falling back to exactly these numbers; the estimate went with the
        # rest of the prediction machinery, and what is left is the fallback
        # that was always there -- an ordering heuristic, never a resource
        # decision.
        if action_name == "prefill":
            action_cost = float(
                sum(
                    self._task_prefill_budget(task)
                    for batch in run.prefill_pending
                    for task, _prepared in batch.selected
                )
            )
            effective_width = phase_width
        elif action_name in {
            "restore",
            "preempt_restore",
            "preempt_condition",
        }:
            action_cost = 0.0
            effective_width = 1
        elif action_name == "condition_batch":
            action_cost = float(phase_width)
            effective_width = phase_width
        elif action_name in {"condition_prefill", "release_admit", "admit"}:
            action_cost = float(prefill_budget)
            effective_width = max(1, admitted)
        else:
            action_cost = 4.0
            effective_width = max(1, active)
        efficiency_cost = action_cost / effective_width
        recovery_credit = (
            2.0 if "recovery" in self._run_job_types(run) else 0.0
        )
        aging_credit = min(256.0, run.age * 8.0)
        starvation_bound_reached = run.age >= 16
        # Do NOT add a minimum-width penalty here. One was tried: a large cost on any action below a target width, to make a lane accumulate before starting.
        # It works on the lane it targets (35% wider, 20% faster) and the co-resident lane loses exactly what that one gains, because the device is one serial resource.
        # Widening a lane redistributes throughput, it doesn't create it.
        # A penalty also only *ranks*: when the narrow action is the only candidate it still wins, and it usually is.
        # Waiting would need an action kind meaning "none of these", which `SchedulerActionKind` doesn't have -- no enabled action raises (scheduler-fill arms).
        return (
            *dependency_tier,
            0 if starvation_bound_reached else 1,
            -run.age if starvation_bound_reached else 0,
            efficiency_cost - recovery_credit - aging_credit,
            completion_budget,
            -(projected_width + age_bonus),
            "recovery" if "recovery" in self._run_job_types(run) else "primary",
            efficiency_cost,
            prefill_budget,
            oldest,
            run.capacity,
        )

    @staticmethod
    def _drain_phase_gpu_ms(
        run: _PersistentRun[_ModelT],
        session: _ResumableSession,
    ) -> None:
        """Accumulate completed CUDA Event timings for this run's record.

        Reporting only. The learned phase-cost curve these used to feed was
        deleted with the rest of the prediction surface; the timings remain
        because the run record and the waterfall are read by people.
        """
        drain = getattr(session, "drain_phase_gpu_ms", None)
        phase_timings = {} if drain is None else dict(drain())
        drain_capture = getattr(session, "drain_cuda_graph_capture_gpu_ms", None)
        if drain_capture is not None:
            capture_ms = float(drain_capture())
            if capture_ms:
                phase_timings["graph_capture"] = capture_ms
        for phase, milliseconds in phase_timings.items():
            run.stats.phase_gpu_ms[phase] = run.stats.phase_gpu_ms.get(phase, 0.0) + float(
                milliseconds
            )

    def _admit_conditioned_batch(self, run: _PersistentRun[_ModelT]) -> None:
        """Move one condition-ready cohort into the session's KV-prefill phase."""
        if run.session is None:
            raise RuntimeError("condition-ready rows require a persistent session")
        queued = run.prefill_pending.pop(0)
        admit = getattr(run.session, "admit_conditioned_many", None)
        if not callable(admit):
            raise RuntimeError("phase-queue session lacks conditioned admission")
        if not admit(queued.conditioned):
            raise RuntimeError("condition-ready cohort could not be admitted")
        if run.quantum_count:
            run.stats.hot_replacements += len(queued.selected)
        for task, prepared_item in queued.selected:
            run.active_by_item[id(prepared_item.session_item)] = (task, prepared_item)

    @staticmethod
    def _persistent_session_owner_free(run: _PersistentRun[_ModelT]) -> bool:
        session = run.session
        return session is not None and not any(
            (
                run.pending_controls,
                run.prefill_pending,
                run.in_flight,
                run.active_by_item,
                run.resident_handles_by_key,
                session.active_count,
                session.occupied_count,
                int(getattr(session, "displaced_count", 0)),
            )
        )

    def _reject_cancelled_dependency_owner(
        self,
        run: _PersistentRun[_ModelT],
        task: _BatchTask[_ModelT, Any],
    ) -> None:
        """Fail loudly when a cancelled task still owes a dependency tuple."""
        if task.dependency_bundle_id is None and task.inherited_claim_token is None:
            return
        bundle = (
            None
            if task.dependency_bundle_id is None
            else self._dependency_bundles.get(task.dependency_bundle_id)
        )
        raise SchedulerInvariantError(
            "cancelled task retained dependency lineage: "
            f"bundle={task.dependency_bundle_id} member={task.dependency_bundle_index} "
            f"claim={None if task.inherited_claim_token is None else task.inherited_claim_token.value} "
            f"model={run.key.model} job_type={task.job_type} "
            f"bundle_completed={None if bundle is None else list(bundle.completed)} "
            f"bundle_published={None if bundle is None else bundle.published}"
        )

    def _release_idle_session(self, run: _PersistentRun[_ModelT]) -> None:
        if not self._persistent_session_owner_free(run):
            raise SchedulerInvariantError(
                "reclaim donor retained continuous-session ownership"
            )
        session = run.session
        assert session is not None
        started_at = time.perf_counter()
        slots_before = self._waterfall_slots(run)
        self._collect_preemption_telemetry(run)
        close = getattr(session, "close", None)
        if callable(close):
            close()
        run.session = None
        run.quantum_count = 0
        self._refresh_capacity_measurement(run)
        run.scheduler_observations["session_release_actions"] += 1
        self._record_waterfall_event(
            run,
            "session_release",
            started_at,
            slots_before,
        )

    def _execute_reclaim_admission(
        self,
        action: SchedulerAction,
        runs_by_id: dict[int, _PersistentRun[_ModelT]],
    ) -> tuple[bool, float, float]:
        """Atomically reclaim owner-free arenas and admit one target row."""
        target = runs_by_id[action.run_id]
        if target.session is not None or not target.pending:
            raise SchedulerInvariantError("reclaim target is no longer nonresident")
        donors = tuple(runs_by_id[run_id] for run_id in action.participants)
        if not donors:
            raise SchedulerInvariantError("reclaim admission has no donor arena")
        device = torch.device(target.key.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for donor in donors:
            self._release_idle_session(donor)
        gc.collect()
        self._offload_models(exclude=target.key)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

        model_started = time.perf_counter()
        model, loaded = self._model_for(target.key)
        model_finished = time.perf_counter()
        # The donors are closed and the device re-read, so the same one
        # comparison that authorised this action has to still hold. It is the
        # only proof: the previous design needed a second, separate walk over
        # every outstanding obligation here.
        view = self._live_pool_view(target.key.device)
        if not view.unbounded and not self._nonresident_arena_admission_safe(
            target, view
        ):
            raise SchedulerInvariantError(
                "reclaimed arena did not satisfy pool admission"
            )
        # Re-price the arena against what the reclaim actually freed.
        target.width = self._resolve_arena_width(target)
        pending_before = len(target.pending)
        deferred_prefill = self._fill_persistent_run(target, model)
        if len(target.pending) >= pending_before:
            raise SchedulerInvariantError("reclaim admission consumed no pending row")
        if deferred_prefill:
            self._admit_conditioned_batch(target)
        target.model_loaded = target.model_loaded or loaded
        target.scheduler_observations["release_admit_actions"] += 1
        return loaded, model_started, model_finished

    @staticmethod
    def _prepare_persistent_selection(
        model: _ModelT,
        selected: list[tuple[_BatchTask[_ModelT, Any], _PersistentRun[_ModelT] | None]],
    ) -> tuple[PreparedWorkItem, ...]:
        """Prepare heterogeneous row roles while preserving admission order."""
        prepared: list[PreparedWorkItem | None] = [None] * len(selected)
        groups: list[
            tuple[
                Callable[
                    [_ModelT, tuple[object, ...]], tuple[PreparedWorkItem, ...]
                ],
                list[tuple[int, object]],
            ]
        ] = []
        for index, (task, _borrowed_from) in enumerate(selected):
            if task.prepare is None:
                prepared[index] = PreparedWorkItem(task.item)
                continue
            group = next(
                (entries for strategy, entries in groups if strategy is task.prepare),
                None,
            )
            if group is None:
                group = []
                groups.append((task.prepare, group))
            group.append((index, task.item))

        for strategy, entries in groups:
            batch = strategy(model, tuple(item for _index, item in entries))
            if len(batch) != len(entries):
                raise RuntimeError("persistent preparation returned the wrong row count")
            if any(not isinstance(item, PreparedWorkItem) for item in batch):
                raise TypeError("persistent preparation returned an untyped row")
            for (index, _item), prepared_item in zip(entries, batch, strict=True):
                prepared[index] = prepared_item

        if any(item is None for item in prepared):
            raise RuntimeError("persistent preparation left an unresolved row")
        return tuple(item for item in prepared if item is not None)

    def _fill_persistent_run(
        self,
        run: _PersistentRun[_ModelT],
        model: _ModelT,
    ) -> bool:
        if run.session is None and not self._nonresident_arena_admission_safe(
            run,
            self._live_pool_view(run.key.device),
        ):
            raise SchedulerInvariantError(
                "nonresident admission lost its fresh pool proof"
            )
        if not run.allow_replacement and run.session is not None:
            if run.quantum_count and run.session.occupied_count:
                return False
            if not run.session.occupied_count:
                self._collect_preemption_telemetry(run)
                run.session = None
                run.quantum_count = 0
        prefill_budget = 4096
        # Bounded by the page supply as well as the arena. Slots and pages are separate conservation laws (R1) and admission only ever consulted the first:
        # an arena with a free slot admits a row the supply can't fund, and the row fails at its first block boundary with `KVPagesExhausted`.
        # Through `_admittable_count`, which legality and the ranking also use. Bounding only here was tried and measured nothing -- the ranking kept promising a wide admission this couldn't make.
        available = self._admittable_count(run)
        if (
            run.session is None
            and run.pending
            and self._task_requires_dependency_credit(run.pending[0])
        ):
            # Establish one measured session row before crediting its peers.
            available = min(available, 1)
        selected: list[
            tuple[_BatchTask[_ModelT, Any], _PersistentRun[_ModelT] | None]
        ] = []
        require_equal_prompt = run.session is None
        cohort_dimension: int | None = None
        while available > 0:
            borrowed_from: _PersistentRun[_ModelT] | None = None
            eligible = [
                (index, candidate)
                for index, candidate in enumerate(run.pending)
                if not require_equal_prompt
                or cohort_dimension is None
                or candidate.batch_dimension == cohort_dimension
            ]
            pending_index = (
                None
                if not eligible
                else min(eligible, key=lambda pair: self._task_schedule_key(pair[1]))[0]
            )
            if pending_index is not None:
                task = run.pending[pending_index]
            else:
                borrowed = self._borrow_matching_pending(
                    run,
                    cohort_dimension,
                )
                if borrowed is None:
                    if run.pending:
                        pending_index = 0
                        task = run.pending[0]
                    else:
                        borrowed = self._borrow_compatible_pending(run)
                        if borrowed is None:
                            break
                        task, borrowed_from = borrowed
                else:
                    task, borrowed_from = borrowed
            if require_equal_prompt and cohort_dimension is None:
                cohort_dimension = task.batch_dimension
            task_prefill = self._task_prefill_budget(task)
            if prefill_budget < task_prefill and run.session is not None:
                if borrowed_from is not None:
                    self._return_borrowed_task(task, borrowed_from)
                break
            if borrowed_from is None:
                assert pending_index is not None
                run.pending.pop(pending_index)
            prefill_budget -= task_prefill
            with self._state_lock:
                if not task.future.set_running_or_notify_cancel():
                    # A plain task may vanish on cancellation, but one that
                    # already owns dependency lineage cannot: its bundle
                    # reserved every member's path to a decision boundary and
                    # has no way to retire a member that never arrives.
                    self._reject_cancelled_dependency_owner(run, task)
                    self._pending.pop(task.sequence, None)
                    continue
                self._pending.pop(task.sequence, None)
            run.started_at[task.sequence] = time.perf_counter()
            if borrowed_from is not None:
                run.stats.borrowed_admissions += 1
            selected.append((task, borrowed_from))
            available -= 1

        if not selected:
            return False
        run.in_flight.extend(task for task, _borrowed_from in selected)
        prepared = self._prepare_persistent_selection(model, selected)
        session_items = tuple(item.session_item for item in prepared)
        replacement_start = 0
        if run.session is None:
            if not self._run_capacity_ready(run):
                raise SchedulerInvariantError(
                    "condition admission preceded capacity preparation"
                )
            # Width is decided here, against the pool as it stands, rather
            # than read back from a solved vector. Nothing between preparation
            # and this point can widen the arena, so a device that filled in
            # the meantime narrows the session instead of failing it.
            run.width = self._resolve_arena_width(run)
            # P4-3a, measured before it is acted on. Opening at what fits is
            # what gave one lane 19 rows and the other 1 once the ceilings were
            # raised (`REPORT.md`, R11); opening at demand is the proposed fix.
            # But a lane whose work has not arrived yet has a *small* demand,
            # and opening narrow with no resize driver built would strand it
            # there -- the same failure the other way round. So the two numbers
            # are recorded side by side first, and the second is not used.
            self._observe_scheduler_histogram(run, "arena_open_width", run.width)
            self._observe_scheduler_histogram(
                run, "arena_open_demand_width", self._run_demand_width(run)
            )
            run.session = run.factory(
                model,
                session_items,
                run.width,
                run.capacity,
            )
            self._refresh_capacity_measurement(run)
            run.arena_blocks = self._arena_blocks(run)
            drain_init_spans = getattr(run.session, "drain_init_wall_spans", None)
            if callable(drain_init_spans) and self._observer.trace_spans:
                for phase, started_at, finished_at in drain_init_spans():
                    self._observer.record_span(
                        phase,
                        started_at,
                        finished_at,
                        resource="gpu",
                        lane=f"{self._session_label(run)} init",
                        model=run.key.model,
                        session_id=run.session.session_id,
                        session_label=self._session_label(run),
                    )
            replacement_start = len(session_items)
        replacements = session_items[replacement_start:]
        if (
            replacements
            and callable(getattr(run.session, "prepare_conditions", None))
            and callable(getattr(run.session, "admit_conditioned_many", None))
        ):
            conditioned = run.session.prepare_conditions(replacements)
            run.prefill_pending.append(
                _PendingPrefill(
                    selected=tuple(
                        zip((task for task, _ in selected), prepared, strict=True)
                    ),
                    conditioned=conditioned,
                )
            )
            run.in_flight.clear()
            return True
        if replacements:
            admit_many = getattr(run.session, "admit_many", None)
            if admit_many is not None:
                admitted = bool(admit_many(replacements))
            else:
                admitted = all(run.session.admit(item) for item in replacements)
            if not admitted:
                raise RuntimeError(
                    "compatible persistent replacement group was not admitted"
                )
            if run.quantum_count:
                run.stats.hot_replacements += len(replacements)
        for (task, _), prepared_item in zip(selected, prepared, strict=True):
            run.active_by_item[id(prepared_item.session_item)] = (
                task,
                prepared_item,
            )
        run.in_flight.clear()
        return False


    def _borrow_matching_pending(
        self,
        target: _PersistentRun[_ModelT],
        batch_dimension: int | None,
    ) -> (
        tuple[
            _BatchTask[_ModelT, Any],
            _PersistentRun[_ModelT],
        ]
        | None
    ):
        """Fill a cohort from compatible runs without padding prompt work."""
        donors = [
            candidate
            for candidate in self._persistent_runs.values()
            if candidate is not target
            and candidate.key == target.key
            and candidate.compatibility_key == target.compatibility_key
            and candidate.allow_replacement == target.allow_replacement
            and any(
                task.batch_dimension == batch_dimension for task in candidate.pending
            )
        ]
        if donors:
            donor = min(
                donors,
                key=lambda candidate: (
                    candidate.pending[0].ready_order,
                    candidate.pending[0].sequence,
                ),
            )
            index = next(
                index
                for index, task in enumerate(donor.pending)
                if task.batch_dimension == batch_dimension
            )
            return donor.pending.pop(index), donor
        return None

    def _borrow_compatible_pending(
        self,
        target: _PersistentRun[_ModelT],
    ) -> (
        tuple[
            _BatchTask[_ModelT, Any],
            _PersistentRun[_ModelT],
        ]
        | None
    ):
        """Fill remaining slots after exact-prompt candidates are exhausted."""
        donors = [
            candidate
            for candidate in self._persistent_runs.values()
            if candidate is not target
            and candidate.key == target.key
            and candidate.compatibility_key == target.compatibility_key
            and candidate.allow_replacement == target.allow_replacement
            and candidate.pending
        ]
        if not donors:
            return None
        donor = min(
            donors,
            key=lambda candidate: (
                candidate.pending[0].ready_order,
                candidate.pending[0].sequence,
            ),
        )
        return donor.pending.pop(0), donor

    @staticmethod
    def _return_borrowed_task(
        task: _BatchTask[_ModelT, Any],
        donor: _PersistentRun[_ModelT],
    ) -> None:
        donor.pending.append(task)
        donor.pending.sort(key=lambda candidate: candidate.sequence)

    def _record_persistent_timing(
        self,
        run: _PersistentRun[_ModelT],
        task: _BatchTask[_ModelT, Any],
        report_stats: bool,
        *,
        result: object | None = None,
        source_item: object | None = None,
    ) -> None:
        finished_at = time.perf_counter()
        started_at = run.started_at.pop(task.sequence)
        telemetry_item = (
            source_item
            if getattr(task.item, "action", None) in {"resume", "discard"}
            and source_item is not None
            else task.item
        )
        result_tokens = tuple(getattr(result, "tokens", ()))
        prompt_token_count = len(getattr(telemetry_item, "prompt_ids", ()))
        max_generation_tokens = int(getattr(telemetry_item, "max_gen_len", 0))
        emitted_eos = bool(getattr(result, "emitted_eos", False))
        guard_action = str(getattr(result, "guard_action", ""))
        if guard_action in {"continue", "accept_candidate"}:
            guard_action = ""
        guard_findings = tuple(getattr(result, "guard_findings", ()))
        candidate_name = str(getattr(telemetry_item, "candidate_name", ""))
        verify_prefix_token_count, verify_boundary_reached = (
            self._recovery_verify_prefix(telemetry_item, result_tokens)
        )
        with self._state_lock:
            producer_waits = tuple(
                wait
                for wait in self._producer_waits.values()
                if wait.run_identity == id(run)
                and wait.pipeline_session_id == task.pipeline_session_id
            )
        timing = ModelTaskTiming(
            sequence=task.sequence,
            key=task.key,
            priority=task.priority,
            job_type=task.job_type,
            queue_wait_seconds=started_at - task.submitted_at,
            run_seconds=finished_at - started_at,
            succeeded=True,
            model_loaded=run.model_loaded and report_stats,
            pipeline_session_id=task.pipeline_session_id,
            batch_size=run.width,
            generation_steps=run.stats.physical_steps if report_stats else 0,
            wasted_token_rows=run.stats.wasted_token_rows if report_stats else 0,
            hot_replacements=run.stats.hot_replacements if report_stats else 0,
            resident_checkpoints=int(
                getattr(result, "resident_handle", None) is not None
            ),
            resident_resumes=int(getattr(task.item, "action", None) == "resume"),
            resident_discards=int(getattr(task.item, "action", None) == "discard"),
            discarded_token_budget=(
                int(
                    getattr(
                        getattr(task.item, "resident_handle", None),
                        "remaining_token_budget",
                        0,
                    )
                )
                if getattr(task.item, "action", None) == "discard"
                else 0
            ),
            bucket_borrows=run.stats.borrowed_admissions if report_stats else 0,
            ready_order=task.ready_order,
            estimated_prefill_tokens=self._task_prefill_budget(task),
            estimated_decode_tokens=self._task_decode_budget(task),
            active_steps_by_width=(
                dict(run.stats.active_steps_by_width) if report_stats else {}
            ),
            prefill_batches_by_width=(
                dict(run.stats.prefill_batches_by_width) if report_stats else {}
            ),
            condition_batches_by_width=(
                dict(run.stats.condition_batches_by_width) if report_stats else {}
            ),
            first_token_batches_by_width=(
                dict(run.stats.first_token_batches_by_width) if report_stats else {}
            ),
            packed_prefill_batches_by_width=(
                dict(run.stats.packed_prefill_batches_by_width) if report_stats else {}
            ),
            quantum_steps_by_size=(
                dict(run.stats.quantum_steps_by_size) if report_stats else {}
            ),
            scheduler_decision_cpu_us=(
                run.stats.scheduler_decision_cpu_seconds * 1_000_000 if report_stats else 0.0
            ),
            scheduler_decision_phase_us=(
                {
                    name: seconds * 1_000_000
                    for name, seconds in run.stats.scheduler_decision_phase_seconds.items()
                }
                if report_stats
                else {}
            ),
            scheduler_action_wall_us=(
                {
                    name: seconds * 1_000_000
                    for name, seconds in run.stats.scheduler_action_wall_seconds.items()
                }
                if report_stats
                else {}
            ),
            scheduler_boundary_gap_us=(
                run.stats.scheduler_boundary_gap_seconds * 1_000_000
                if report_stats
                else 0.0
            ),
            scheduler_boundary_gap_count=(
                run.stats.scheduler_boundary_gap_count if report_stats else 0
            ),
            scheduler_watchdog_timeout_seconds=self._watchdog_config.timeout_seconds,
            producer_decision_waits_outstanding=len(producer_waits),
            producer_decision_waits_unbound=sum(
                wait.future is None for wait in producer_waits
            ),
            phase_gpu_ms=(dict(run.stats.phase_gpu_ms) if report_stats else {}),
            prefill_graph=(dict(run.stats.prefill_graph) if report_stats else {}),
            prefill_graph_refusals=(
                dict(run.stats.prefill_graph_refusals) if report_stats else {}
            ),
            cuda_graph_captures=(run.stats.cuda_graph_captures if report_stats else 0),
            cuda_graph_replays=(run.stats.cuda_graph_replays if report_stats else 0),
            cuda_graph_refusals=(run.stats.cuda_graph_refusals if report_stats else 0),
            cuda_graph_evictions=(run.stats.cuda_graph_evictions if report_stats else 0),
            cuda_graph_variants=(
                {key: dict(value) for key, value in run.stats.cuda_graph_variants.items()}
                if report_stats
                else {}
            ),
            cuda_graph_cache_entries_peak=(
                run.stats.cuda_graph_cache_entries_peak if report_stats else 0
            ),
            cuda_graph_cache_static_bytes_peak=(
                run.stats.cuda_graph_cache_static_bytes_peak if report_stats else 0
            ),
            cuda_graph_cache_pool_bytes_peak=(
                run.stats.cuda_graph_cache_pool_bytes_peak if report_stats else 0
            ),
            scheduler_observations=(
                dict(run.scheduler_observations) if report_stats else {}
            ),
            scheduler_gauges=(dict(run.stats.scheduler_gauges) if report_stats else {}),
            scheduler_state_histograms=(
                {
                    name: dict(histogram)
                    for name, histogram in run.stats.scheduler_state_histograms.items()
                }
                if report_stats
                else {}
            ),
            candidate_name=candidate_name,
            trace_context=tuple(getattr(telemetry_item, "trace_context", ())),
            prompt_token_count=prompt_token_count,
            max_generation_tokens=max_generation_tokens,
            result_token_count=len(result_tokens),
            generated_token_count=max(0, len(result_tokens) - prompt_token_count),
            termination_reason=(
                "eos"
                if emitted_eos
                else "guard_interrupt"
                if guard_action in {"interrupt_to_recovery", "reject_candidate"}
                else "verify_boundary"
                if verify_boundary_reached
                else "max_length"
                if candidate_name and result is not None
                else ""
            ),
            verify_boundary_reached=verify_boundary_reached,
            verify_prefix_token_count=verify_prefix_token_count,
            guard_action=guard_action,
            guard_warning_findings=sum(
                getattr(finding, "status", None) == "warning"
                for finding in guard_findings
            ),
            guard_critical_findings=sum(
                getattr(finding, "status", None) == "critical"
                for finding in guard_findings
            ),
            guard_reasons=tuple(
                str(getattr(finding, "reason", ""))
                for finding in guard_findings
                if getattr(finding, "reason", None)
            ),
        )
        if report_stats:
            # One assignment, not 27 clears. Whatever `_RunTelemetry` grows next resets with it, which is the point of the type existing:
            # a counter added to the record and forgotten here reported a run's totals as the whole process's.
            run.stats = _RunTelemetry()
            run.scheduler_observations.clear()
            run.model_loaded = False
        with self._state_lock:
            self._observer.task_completed(timing)

    @staticmethod
    def _recovery_verify_prefix(
        item: object,
        tokens: tuple[int, ...],
    ) -> tuple[int, bool]:
        boundary = getattr(item, "verify_shift_value", None)
        vocab = getattr(item, "expected_vocab", ())
        if boundary is None or not vocab:
            return 0, False
        for index, token in enumerate(tokens):
            if not 0 <= token < len(vocab):
                continue
            event = vocab[token]
            if getattr(event, "type", None) == "shift" and int(
                getattr(event, "value", -1)
            ) >= int(boundary):
                return index + 1, True
        return len(tokens), False

    def _fail_persistent_run(
        self,
        run: _PersistentRun[_ModelT],
        error: BaseException,
    ) -> None:
        with self._state_lock:
            waits = tuple(
                (handle_key, wait)
                for handle_key, wait in self._producer_waits.items()
                if wait.run_identity == id(run)
            )
            for handle_key, wait in waits:
                del self._producer_waits[handle_key]
                self._producer_wait_tombstones[handle_key] = wait.binding_token
            claims = tuple(
                (claim_id, claim)
                for claim_id, claim in self._transitive_claims.items()
                if claim.run_identity == id(run)
            )
            for claim_id, claim in claims:
                del self._transitive_claims[claim_id]
                if claim.handle_key is not None:
                    self._handle_claim_tokens.pop(claim.handle_key, None)
                self._claim_tombstones[claim_id] = claim.binding_token
        for _handle_key, wait in waits:
            if wait.delivery_future is not None and not wait.delivery_future.done():
                wait.delivery_future.set_exception(error)
            if wait.future is not None and not wait.future.done():
                wait.future.cancel()
        for _claim_id, claim in claims:
            if claim.delivery_future is not None and not claim.delivery_future.done():
                claim.delivery_future.set_exception(error)
            if claim.future is not None and not claim.future.done():
                claim.future.cancel()
        tasks = (
            tuple(run.pending)
            + tuple(run.in_flight)
            + tuple(task for task, _prepared in run.active_by_item.values())
            + tuple(run.pending_controls)
            + tuple(
                task
                for batch in run.prefill_pending
                for task, _prepared in batch.selected
            )
        )
        session = run.session
        if session is not None:
            discard = getattr(session, "discard", None)
            if callable(discard):
                for handle in run.resident_handles_by_key.values():
                    try:
                        discard(handle)
                    except Exception:
                        run.scheduler_observations[
                            "terminal_handle_release_failures"
                        ] += 1
        run.resident_handles_by_key.clear()
        self._collect_preemption_telemetry(run)
        run.session = None
        run.pending.clear()
        run.in_flight.clear()
        run.pending_controls.clear()
        run.prefill_pending.clear()
        run.active_by_item.clear()
        for task in tasks:
            with self._state_lock:
                self._pending.pop(task.sequence, None)
            if not task.future.done():
                task.future.set_exception(error)

    def _run_inline(
        self,
        key: ModelKey,
        operation: Callable[[_ModelT], _ResultT],
        *,
        priority: int,
    ) -> _ResultT:
        """Execute nested model work without queueing behind the caller."""
        sequence = next(self._sequence)
        started_at = time.perf_counter()
        succeeded = False
        model_loaded = False
        try:
            model, model_loaded = self._model_for(key)
            result = operation(model)
            succeeded = True
            return result
        finally:
            finished_at = time.perf_counter()
            timing = ModelTaskTiming(
                sequence=sequence,
                key=key,
                priority=priority,
                job_type="primary",
                queue_wait_seconds=0.0,
                run_seconds=finished_at - started_at,
                succeeded=succeeded,
                model_loaded=model_loaded,
                pipeline_session_id=None,
            )
            with self._state_lock:
                self._observer.task_completed(timing)

    def _model_for(self, key: ModelKey) -> tuple[_ModelT, bool]:
        cached = self._models.get(key.identity)
        if cached is not None:
            if cached.key.execution_profile != key.execution_profile:
                raise RuntimeError(
                    "one model identity requested conflicting execution profiles"
                )
            self._activate_model(key, cached.model)
            return cached.model, False
        model, load_growth_bytes = self._load_model(key)
        self._models[key.identity] = _CachedModel(
            key,
            model,
            load_growth_bytes=load_growth_bytes,
            weight_storage_bytes=_model_weight_storage_bytes(model),
        )
        return model, True

    @staticmethod
    def _allocator_reserved_bytes(device: torch.device) -> int:
        return max(
            int(torch.cuda.memory_allocated(device)),
            int(torch.cuda.memory_reserved(device)),
        )

    def _load_model(self, key: ModelKey) -> tuple[_ModelT, int]:
        def load_once() -> tuple[_ModelT, int]:
            baseline: int | None = None
            device = torch.device(key.device)
            if device.type == "cuda":
                try:
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    baseline = self._allocator_reserved_bytes(device)
                except (AssertionError, RuntimeError, ValueError):
                    baseline = None
            model = self._loader(key)
            if baseline is None:
                return model, 0
            torch.cuda.synchronize(device)
            return model, max(
                0,
                self._allocator_reserved_bytes(device) - baseline,
            )

        try:
            return load_once()
        except torch.cuda.OutOfMemoryError:
            if not self._offload_models():
                raise
            torch.cuda.empty_cache()
            return load_once()

    def _activate_model(self, key: ModelKey, model: _ModelT) -> None:
        move_to = getattr(model, "move_to", None)
        current_device = getattr(model, "_device", None)
        if (
            move_to is None
            or current_device is None
            or str(current_device) == key.device
        ):
            return
        try:
            move_to(key.device)
        except torch.cuda.OutOfMemoryError:
            self._offload_models(exclude=key)
            torch.cuda.empty_cache()
            move_to(key.device)

    def _offload_models(self, *, exclude: ModelKey | None = None) -> bool:
        moved = False
        excluded_identity = None if exclude is None else exclude.identity
        resident_identities = {
            run.key.identity
            for run in self._persistent_runs.values()
            if run.session is not None
        }
        for identity, cached in self._models.items():
            if identity == excluded_identity or identity in resident_identities:
                continue
            cached_model = cached.model
            move_to = getattr(cached_model, "move_to", None)
            current_device = getattr(cached_model, "_device", None)
            if (
                move_to is None
                or current_device is None
                or str(current_device) == "cpu"
            ):
                continue
            move_to("cpu")
            moved = True
        return moved

    def _watch_stop_event(self) -> None:
        assert self._stop_event is not None
        self._stop_event.wait()
        if self._worker_stopped.is_set():
            return
        with self._state_lock:
            self._begin_close_locked(cancel_pending=True)

    def _begin_close_locked(self, *, cancel_pending: bool) -> None:
        if cancel_pending:
            self._cancel_pending_locked()
            self._abort_active_on_close = True
        if self._closed:
            return
        self._closed = True
        sequence = next(self._sequence)
        self._queue.put((float("inf"), sequence, self._stop))

    def _cancel_pending_locked(self) -> int:
        cancelled = 0
        for task in self._pending.values():
            if not task.future.cancelled() and task.future.cancel():
                cancelled += 1
        return cancelled


@dataclass
class _ArenaDeclarationState:
    """One pipeline session's declared residency on one device.

    All that is left of `_InitialAllocationState`. It carries no solved
    vector, no evidence, and no bootstrap bypass -- only the measured row cost
    of each declared lane, and the two fields R7 proved a wait outside the
    liveness model needs.
    """

    declaration: ArenaDeclaration
    future: Future[None] = field(default_factory=Future)
    # Set once resolution has run to completion, successfully or not. A
    # submitter waits on `future` from outside the scheduler's liveness model,
    # so this is the only signal that separates "still loading" from "resolved
    # nothing" -- see `_await_arena_declaration` (R7).
    resolution_finished: threading.Event = field(default_factory=threading.Event)
    costs_by_identity: dict[ModelIdentity, ArenaCost] = field(default_factory=dict)


class ModelWorkerRegistry(Generic[_ModelT]):
    """Create one serialized worker per device/stem execution lane."""

    def __init__(
        self,
        loader: Callable[[ModelKey], _ModelT],
        *,
        stop_event: threading.Event | None = None,
        observer: SchedulerObserver | None = None,
        watchdog_config: SchedulerWatchdogConfig | None = None,
    ) -> None:
        self._loader = loader
        self._stop_event = stop_event
        self._observer = observer or SchedulerObserver()
        self._watchdog_config = watchdog_config or SchedulerWatchdogConfig()
        self._workers: dict[_WorkerLane, ModelWorkerPool[_ModelT]] = {}
        self._device_execution_locks: dict[str, threading.RLock] = {}
        self._device_terminal_futures: dict[str, Future[None]] = {}
        self._arena_declarations: dict[
            tuple[str, str], _ArenaDeclarationState
        ] = {}
        self._lock = threading.Lock()
        self._closed = False

    def declare_arenas(
        self,
        declaration: ArenaDeclaration,
        *,
        pipeline_session_id: str,
    ) -> Future[None]:
        """Load every declared lane and measure what one of its rows costs.

        The whole of what used to be `declare_initial_allocation`: no arena-family evidence, no throughput ranking, no joint solve, no `_bootstrap_uncalibrated_family`.
        Declaring residency is now loading the models and asking each what a row costs, the one thing the pool can't work out itself.

        Runs HERE, before any submission, not at first admission: the pool must know a lane's floor before another lane sizes an arena against the device.
        """
        identity = (pipeline_session_id, declaration.device)
        preload_futures: list[tuple[ArenaLaneDeclaration, Future[ArenaCost | None]]] = []
        with self._lock:
            if self._closed:
                raise RuntimeError("model worker registry is closed")
            existing = self._arena_declarations.get(identity)
            if existing is not None and existing.declaration != declaration:
                raise SchedulerInvariantError(
                    "arena declaration changed after registration"
                )
            if existing is not None:
                return existing.future
            state = _ArenaDeclarationState(declaration)
            self._arena_declarations[identity] = state
            for lane in declaration.lanes:
                worker = self._worker_for(lane.key)

                def measure(
                    model: _ModelT,
                    selected: ArenaLaneDeclaration = lane,
                ) -> ArenaCost | None:
                    if selected.arena_cost is None:
                        return None
                    cost = selected.arena_cost(
                        model, selected.kv_capacity, selected.key.model
                    )
                    return ArenaCost(
                        row_bytes=cost.row_bytes,
                        minimum_width=max(
                            cost.minimum_width, selected.minimum_width
                        ),
                        maximum_width=selected.maximum_width,
                    )

                preload_futures.append(
                    (
                        lane,
                        worker.submit(
                            lane.key,
                            measure,
                            priority=-100,
                            job_type="control",
                            record_timing=False,
                            trace_model_load=True,
                            pipeline_session_id=pipeline_session_id,
                        ),
                    )
                )
        for lane, future in preload_futures:
            future.add_done_callback(
                lambda completed, selected=lane: self._arena_lane_measured(
                    identity,
                    selected,
                    completed,
                )
            )
        return state.future

    def _arena_lane_measured(
        self,
        identity: tuple[str, str],
        lane: ArenaLaneDeclaration,
        completed: Future[ArenaCost | None],
    ) -> None:
        try:
            cost = completed.result()
        except BaseException as error:
            with self._lock:
                state = self._arena_declarations.get(identity)
            if state is not None and not state.future.done():
                state.future.set_exception(error)
            return
        with self._lock:
            state = self._arena_declarations.get(identity)
            if state is None or state.future.done():
                return
            if cost is not None:
                state.costs_by_identity[lane.key.identity] = cost
            measured = len(state.costs_by_identity)
            priced = sum(
                1 for declared in state.declaration.lanes
                if declared.arena_cost is not None
            )
            ready = measured >= priced
        if not ready:
            return
        # This runs as a `Future` done-callback, and `concurrent.futures`
        # logs-and-swallows anything raised from one. An exception escaping
        # resolution therefore left `state.future` pending forever, parking
        # every submitter inside `_require_arena_declaration` with the device
        # allocated and idle, invisible to every watchdog (R7). Resolution must
        # publish an outcome on every path.
        try:
            self._publish_arena_declaration(state)
        except BaseException as error:
            if not state.future.done():
                state.future.set_exception(error)
        finally:
            state.resolution_finished.set()

    def _publish_arena_declaration(self, state: _ArenaDeclarationState) -> None:
        """Declare every measured lane on the device's pool, then release."""
        device = state.declaration.device
        pool = device_block_pool(device)
        for lane in state.declaration.lanes:
            cost = state.costs_by_identity.get(lane.key.identity)
            if cost is not None:
                pool.declare_lane(lane.key.model, cost)
        recorded_at = time.perf_counter()
        self._observer.record_span(
            "arena_declaration",
            recorded_at,
            recorded_at,
            resource="gpu",
            lane=f"{device} arena declaration",
            device=device,
            block_bytes=pool.block_bytes,
            row_bytes={
                lane.key.model: cost.row_bytes
                for lane in state.declaration.lanes
                if (cost := state.costs_by_identity.get(lane.key.identity))
                is not None
            },
            floor_blocks=pool.floor_blocks(),
        )
        state.future.set_result(None)

    def _require_arena_declaration(
        self,
        key: ModelKey,
        pipeline_session_id: str | None,
    ) -> None:
        """Block until this device's lanes are priced, or fail loudly."""
        if not _device_is_cuda(key.device):
            return
        if pipeline_session_id is None:
            raise SchedulerInvariantError(
                "CUDA submission lacks pipeline ownership"
            )
        with self._lock:
            state = self._arena_declarations.get(
                (pipeline_session_id, key.device)
            )
        if state is None:
            raise SchedulerInvariantError(
                "CUDA submission precedes its arena declaration"
            )
        if key.identity not in {
            lane.key.identity for lane in state.declaration.lanes
        }:
            raise SchedulerInvariantError(
                "CUDA submission lane is absent from its arena declaration"
            )
        self._await_arena_declaration(state)

    @staticmethod
    def _await_arena_declaration(state: _ArenaDeclarationState) -> None:
        """Block for the declared lanes, but never through a contradiction.

        Every submitter parks here, on a future resolved by a preload done-callback rather than any scheduler action.
        That puts the wait outside the liveness model entirely -- no legal action missing, no run unresolved, nothing for the watchdog to fire on
        -- so a lost resolution once held a device at 0% until it was killed, and wrote no run metadata (R7).

        Resolution finishing without publishing an outcome is not slowness, it's a contradiction, and reporting beats sleeping on it.
        Waiting stays unbounded while resolution has NOT finished: that time is a model load owned by the worker's own watchdog.
        """
        while True:
            try:
                state.future.result(timeout=_DECLARATION_POLL_SECONDS)
                return
            except FutureTimeoutError:
                pass
            if state.resolution_finished.is_set() and not state.future.done():
                raise SchedulerInvariantError(
                    "arena declaration resolved without publishing an outcome "
                    f"for {state.declaration.device}"
                )

    def _resolve_dependency_claim(
        self,
        claim: TransitiveDependencyClaim | None,
        pipeline_session_id: str | None,
    ) -> TransitiveDependencyClaim | None:
        """Guarantee every member's lane is priced before the parent admits.

        The claim itself is unchanged -- members carry identity only. What has
        to happen before it can be reserved is that each member's lane cost is
        on the pool, which is what awaiting the declaration establishes.
        """
        if claim is None:
            return None
        for member in claim.members:
            self._require_arena_declaration(member.key, pipeline_session_id)
        return claim

    def release_arena_declaration(
        self,
        device: str,
        *,
        pipeline_session_id: str,
    ) -> None:
        with self._lock:
            self._arena_declarations.pop((pipeline_session_id, device), None)

    def _worker_for(self, key: ModelKey) -> ModelWorkerPool[_ModelT]:
        """Return the lane worker, creating it under the registry lock."""
        if self._closed or (self._stop_event is not None and self._stop_event.is_set()):
            raise RuntimeError("model worker registry is closed")
        lane = _WorkerLane.from_key(key)
        worker = self._workers.get(lane)
        if worker is None:
            execution_lock = self._device_execution_locks.setdefault(
                key.device, threading.RLock()
            )
            worker = ModelWorkerPool(
                self._loader,
                name=f"model-{key.device}-{key.stem or 'shared'}",
                stop_event=self._stop_event,
                observer=self._observer,
                execution_lock=execution_lock,
                watchdog_config=self._watchdog_config,
                device=key.device,
                device_terminal=self.terminate_device,
            )
            self._workers[lane] = worker
        return worker

    def device_terminal_future(self, device: str) -> Future[None]:
        """Return the stable event-driven terminal signal for one device."""
        with self._lock:
            return self._device_terminal_futures.setdefault(device, Future())

    def wait_for_first(
        self,
        futures: set[Future[Any]] | dict[Future[Any], Any],
    ) -> tuple[set[Future[Any]], set[Future[Any]]]:
        """Wait for direct-registry work or raise a device terminal error."""
        work = set(futures)
        with self._lock:
            devices = tuple(
                {lane.device for lane in self._workers}
                | set(self._device_terminal_futures)
            )
        terminals = {self.device_terminal_future(device) for device in devices}
        done, pending = wait(work | terminals, return_when=FIRST_COMPLETED)
        for terminal in terminals & done:
            terminal.result()
        return done & work, pending & work

    def terminate_device(self, device: str, error: BaseException) -> None:
        """Publish one terminal error and wake every lane on the device."""
        with self._lock:
            terminal = self._device_terminal_futures.setdefault(device, Future())
            workers = tuple(
                worker
                for lane, worker in self._workers.items()
                if lane.device == device
            )
        if not terminal.done():
            terminal.set_exception(error)
        for worker in workers:
            worker.request_device_terminal(error)

    def submit(
        self,
        key: ModelKey,
        operation: Callable[[_ModelT], _ResultT],
        *,
        priority: int = 10,
        pipeline_session_id: str | None = None,
    ) -> Future[_ResultT]:
        with self._lock:
            worker = self._worker_for(key)
        return worker.submit(
            key,
            operation,
            priority=priority,
            pipeline_session_id=pipeline_session_id,
        )

    def preload(
        self, key: ModelKey, *, pipeline_session_id: str | None = None
    ) -> Future[None]:
        """Load one model early without counting it as scheduler work."""
        with self._lock:
            worker = self._worker_for(key)

        def load_only(_model: _ModelT) -> None:
            return None

        future = worker.submit(
            key,
            load_only,
            priority=-100,
            job_type="control",
            record_timing=False,
            trace_model_load=True,
            pipeline_session_id=pipeline_session_id,
        )
        return future

    def record_waterfall_span(
        self,
        action: str,
        started_at: float,
        finished_at: float,
        *,
        resource: str,
        lane: str,
        **details: object,
    ) -> None:
        self._observer.record_span(
            action,
            started_at,
            finished_at,
            resource=resource,
            lane=lane,
            **details,
        )

    def run(
        self,
        key: ModelKey,
        operation: Callable[[_ModelT], _ResultT],
        *,
        priority: int = 10,
    ) -> _ResultT:
        with self._lock:
            worker = self._worker_for(key)
        return worker.run(key, operation, priority=priority)

    def submit_batch(
        self,
        key: ModelKey,
        compatibility_key: object,
        item: object,
        *,
        condition_compatibility_key: object | None = None,
        max_batch_size: int,
        arena_cost: ArenaCostProbe | None = None,
        kv_pool: KVPoolDeclaration | None = None,
        minimum_width: int = 1,
        maximum_width: int | None = None,
        priority: int = 10,
        job_type: GenerationJobType = "primary",
        batch_dimension: int | None = None,
        continuous_factory: Callable[..., object] | None = None,
        prepare: Callable[[_ModelT, tuple[object, ...]], tuple[PreparedWorkItem, ...]]
        | None = None,
        continuous_replacement: bool = True,
        ready_order: str | None = None,
        dependency_claim: TransitiveDependencyClaim | None = None,
        pipeline_session_id: str | None = None,
    ) -> Future[_ResultT]:
        self._require_arena_declaration(key, pipeline_session_id)
        with self._lock:
            worker = self._worker_for(key)
        return worker.submit_batch(
            key,
            compatibility_key,
            item,
            condition_compatibility_key=condition_compatibility_key,
            max_batch_size=max_batch_size,
            arena_cost=arena_cost,
            kv_pool=kv_pool,
            minimum_width=minimum_width,
            maximum_width=maximum_width,
            priority=priority,
            job_type=job_type,
            batch_dimension=batch_dimension,
            continuous_factory=continuous_factory,
            prepare=prepare,
            continuous_replacement=continuous_replacement,
            ready_order=ready_order,
            dependency_claim=self._resolve_dependency_claim(
                dependency_claim,
                pipeline_session_id,
            ),
            pipeline_session_id=pipeline_session_id,
        )

    def submit_prepared(
        self,
        submission: BatchSubmission,
        *,
        pipeline_session_id: str | None = None,
    ) -> Future[_ResultT]:
        """Queue an existing submission on the lane it names."""
        self._require_arena_declaration(submission.key, pipeline_session_id)
        with self._lock:
            worker = self._worker_for(submission.key)
        return worker.submit_prepared(
            replace(
                submission,
                dependency_claim=self._resolve_dependency_claim(
                    submission.dependency_claim,
                    pipeline_session_id,
                ),
            ),
            pipeline_session_id=pipeline_session_id,
        )

    def submit_batch_group(
        self,
        parent_key: ModelKey,
        claim_token: DependencyClaimToken,
        parent_resident_handle: object | None,
        members: tuple[BatchSubmission, ...],
        *,
        pipeline_session_id: str | None = None,
    ) -> Future[tuple[object, ...]]:
        for member in members:
            self._require_arena_declaration(member.key, pipeline_session_id)
        resolved_members = tuple(
            replace(
                member,
                dependency_claim=self._resolve_dependency_claim(
                    member.dependency_claim,
                    pipeline_session_id,
                ),
            )
            for member in members
        )
        with self._lock:
            worker = self._worker_for(parent_key)
            if any(
                self._worker_for(member.key) is not worker
                for member in resolved_members
            ):
                raise ValueError("dependency bundle crossed worker lanes")
        return worker.submit_batch_group(
            claim_token,
            parent_resident_handle,
            resolved_members,
            pipeline_session_id=pipeline_session_id,
        )

    def bind_producer_decision_waits(
        self,
        key: ModelKey,
        handles: tuple[object, ...],
        future: Future[object],
        *,
        claim_tokens: tuple[DependencyClaimToken, ...] = (),
        pipeline_session_id: str | None = None,
    ) -> Future[object]:
        with self._lock:
            worker = self._worker_for(key)
        return worker.bind_producer_decision_waits(
            handles,
            future,
            claim_tokens=claim_tokens,
            pipeline_session_id=pipeline_session_id,
        )

    def transfer_producer_decision_waits(
        self,
        key: ModelKey,
        handles: tuple[object, ...],
        future: Future[object],
        *,
        claim_tokens: tuple[DependencyClaimToken, ...] = (),
        pipeline_session_id: str | None = None,
    ) -> Future[object]:
        with self._lock:
            worker = self._worker_for(key)
        return worker.transfer_producer_decision_waits(
            handles,
            future,
            claim_tokens=claim_tokens,
            pipeline_session_id=pipeline_session_id,
        )

    def release_dependency_claim(
        self,
        key: ModelKey,
        token: DependencyClaimToken,
        *,
        pipeline_session_id: str | None = None,
    ) -> None:
        with self._lock:
            worker = self._worker_for(key)
        worker.release_dependency_claim(
            token,
            pipeline_session_id=pipeline_session_id,
        )

    @property
    def timings(self) -> tuple[ModelTaskTiming, ...]:
        return tuple(sorted(self._observer.timings, key=lambda timing: timing.sequence))

    @property
    def waterfall_events(self) -> tuple[dict[str, object], ...]:
        return tuple(self._observer.events)

    def cancel_pending(self) -> int:
        """Cancel queued work across all workers without stopping active tasks."""
        with self._lock:
            workers = tuple(self._workers.values())
        return sum(worker.cancel_pending() for worker in workers)

    def close(
        self, *, cancel_pending: bool = False, wait: bool = True
    ) -> None:
        with self._lock:
            self._closed = True
            workers = tuple(self._workers.values())
        if cancel_pending:
            for worker in workers:
                worker.cancel_pending()
        for worker in workers:
            worker.close(wait=wait)

    def __enter__(self) -> ModelWorkerRegistry[_ModelT]:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
