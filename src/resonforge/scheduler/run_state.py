"""The worker's own run and task state, and the two protocols around it.

Split out of `model_workers` so that code operating on a run can live outside
`ModelWorkerPool`. Measured before the split: the capacity cluster reads 15 of
`_PersistentRun`'s 35 fields and only 2 of the pool's own, so its coupling runs
through the run object, not through pool state -- but it could not be imported
out while the run type was defined in the module importing it.

Nothing here decides anything. `ModelWorkerPool` still owns every decision;
these are the values it decides about.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

from resonforge.scheduler.device.block_pool import ArenaCost, blocks_for
from resonforge.scheduler.model_types import (
    ArenaCostProbe,
    DependencyClaimToken,
    GenerationJobType,
    KVPoolDeclaration,
    ModelKey,
    PreparedWorkItem,
    TransitiveDependencyClaim,
)

_ModelT = TypeVar("_ModelT")
_ResultT = TypeVar("_ResultT")

@dataclass
class _PendingPrefill(Generic[_ModelT]):
    """Condition-complete rows waiting for their independent KV-prefill turn."""

    selected: tuple[tuple[_BatchTask[_ModelT, Any], PreparedWorkItem], ...]
    conditioned: object

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

    #: The outcome to publish for a control action this session executed. The
    #: scheduler decides *that* a row is discarded; the result type saying so
    #: belongs to the backend, and building one here is what made the
    #: scheduler import its transcriber.
    def control_outcome(self, *, discarded: bool) -> object: ...

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

    def dependency_tier(
        self,
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
        if self.job_types() & {"recovery", "control"}:
            return 2, 0
        if action == "decode":
            return 3, 0
        return 4, 2

    def waterfall_slots(self) -> list[dict[str, object]]:
        if self.session is None:
            return []
        slots: list[dict[str, object]] = []
        for index, slot in enumerate(getattr(self.session, "slots", ())):
            row = getattr(slot, "row", None)
            if row is None:
                continue
            request = getattr(row, "request", None)
            task_entry = self.active_by_item.get(id(request))
            task = None if task_entry is None else task_entry[0]
            slots.append(
                {
                    "session_id": self.session.session_id,
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

    def drain_phase_gpu_ms(
        self,
        session: _ResumableSession,
    ) -> None:
        """Accumulate completed CUDA Event timings for this self's record.

        Reporting only. The learned phase-cost curve these used to feed was
        deleted with the rest of the prediction surface; the timings remain
        because the self record and the waterfall are read by people.
        """
        drain = getattr(session, "drain_phase_gpu_ms", None)
        phase_timings = {} if drain is None else dict(drain())
        drain_capture = getattr(session, "drain_cuda_graph_capture_gpu_ms", None)
        if drain_capture is not None:
            capture_ms = float(drain_capture())
            if capture_ms:
                phase_timings["graph_capture"] = capture_ms
        for phase, milliseconds in phase_timings.items():
            self.stats.phase_gpu_ms[phase] = self.stats.phase_gpu_ms.get(phase, 0.0) + float(
                milliseconds
            )

    def owner_free(self) -> bool:
        """A resident lane holding nothing -- the only state a resize is legal in.

        Same question as `run_owner_free`, read straight off the run: assembling a `SchedulerRunState` costs a device query per self per turn, and this is asked of every self every turn.
        The two must not disagree, so here is why the shorter list is the SAME list -- five of the assembled fields are implied, and none of them is dropped by choice:
          active <= occupied; control_intents is built from `pending_controls`;
          producer_waits and bundle_owned_handles are both proven subsets of `resident_handles_by_key` by invariants asserted where the snapshot is built.
        `tracked` is implied by nothing, was missing, and is the one field this predicate used to be looser by.
        """
        session = self.session
        return session is not None and not (
            session.occupied_count
            or self.prefill_pending
            or self.in_flight
            or self.pending_controls
            or self.resident_handles_by_key
            or self.active_by_item
        )

    def releasable(self) -> bool:
        session = self.session
        return session is not None and not any(
            (
                self.pending_controls,
                self.prefill_pending,
                self.in_flight,
                self.active_by_item,
                self.resident_handles_by_key,
                session.active_count,
                session.occupied_count,
                int(getattr(session, "displaced_count", 0)),
            )
        )

    def refresh_capacity_measurement(self) -> None:
        if self.session is None:
            self.capacity_resident_bytes = 0
            self.capacity_graph_static_bytes = 0
            self.capacity_total_bytes = 0
            return
        report = getattr(self.session, "capacity_telemetry", None)
        if not callable(report):
            return
        values = dict(report())
        self.capacity_resident_bytes = int(values.get("resident_bytes", 0))
        self.capacity_graph_static_bytes = int(values.get("graph_static_bytes", 0))
        self.capacity_total_bytes = int(values.get("total_bytes", 0))
        measured = int(values.get("estimated_row_bytes", 0))
        if measured > 0:
            # live all-in row cost, conditions and masks included. What a later arena on this lane is sized from, and never written anywhere that outlives the process.
            self.measured_row_bytes = measured

    def observe_lane_gauge(
        self,
        name: str,
        value: int,
    ) -> None:
        """A LEVEL that belongs to one lane -- a width, a price, a declaration.

        Not `scheduler_observations`: that bag is SUMMED, across the lanes of a
        self and again across the runs of an arm, and a level summed is not that
        level. `arena_pool_width` read a stable 54 through a whole A/B because
        one lane at 54 and two lanes at 31 + 23 are the same number (R18).
        Keyed by lane so the gauge aggregation -- `max` per name, in
        `run_metadata.py` -- runs per lane instead of across them.
        """
        self.stats.scheduler_gauges[f"{name}:{self.key.model}"] = int(value)

    def observe_device_gauge(
        self,
        name: str,
        value: int,
    ) -> None:
        """A LEVEL that belongs to the device, recorded against one self of it.

        Unkeyed: every lane on the device would report the same reading, and
        `max` is the aggregation that survives two of them doing so.
        """
        self.stats.scheduler_gauges[name] = int(value)

    def observe_scheduler_histogram(
        self,
        name: str,
        value: int,
    ) -> None:
        self.stats.scheduler_state_histograms.setdefault(name, Counter())[int(value)] += 1

    def admit_conditioned_batch(self) -> None:
        """Move one condition-ready cohort into the session's KV-prefill phase."""
        if self.session is None:
            raise RuntimeError("condition-ready rows require a persistent session")
        queued = self.prefill_pending.pop(0)
        admit = getattr(self.session, "admit_conditioned_many", None)
        if not callable(admit):
            raise RuntimeError("phase-queue session lacks conditioned admission")
        if not admit(queued.conditioned):
            raise RuntimeError("condition-ready cohort could not be admitted")
        if self.quantum_count:
            self.stats.hot_replacements += len(queued.selected)
        for task, prepared_item in queued.selected:
            self.active_by_item[id(prepared_item.session_item)] = (task, prepared_item)

    def arena_bytes(self,
        width: int | None = None) -> int:
        """What this self's arena occupies, at the width it will actually open.

        `measured_row_bytes` is what a live session reported for this self's own
        rows; it beats the analytic price whenever it is larger, because the
        analytic price deliberately leaves the condition slots out. Taking the
        maximum keeps a live workload from under-charging itself without ever
        letting an unmeasured self charge less than it costs.
        """
        cost = self.arena_cost
        if cost is None:
            return 0
        rows = self.width if width is None else width
        return max(cost.row_bytes, self.measured_row_bytes) * max(1, rows)

    def arena_blocks_at(
        self,
        width: int | None = None,
    ) -> int:
        return blocks_for(self.arena_bytes(width))

    def scheduling_priority(self) -> int:
        tasks = self.tasks()
        return min(
            (getattr(task, "priority", self.priority) for task in tasks),
            default=self.priority,
        )

    def accumulate_quantum_stats(
        self,
        stats: object,
    ) -> None:
        """Fold one consumed quantum's session stats into the self's telemetry."""
        self.quantum_count += 1
        self.stats.physical_steps += int(getattr(stats, "physical_steps", 0))
        self.stats.wasted_token_rows += int(getattr(stats, "wasted_token_rows", 0))
        for width, steps in dict(getattr(stats, "active_steps_by_width", {})).items():
            self.stats.active_steps_by_width[int(width)] = self.stats.active_steps_by_width.get(
                int(width), 0
            ) + int(steps)

    def collect_preemption_telemetry(self) -> None:
        """Move what preemption cost this self's session into its observations.

        Called once per quantum and once more wherever the session is about to be let go, so nothing is attributed to a self that didn't incur it and nothing is dropped because the last quantum was the one that preempted.
        Not bracketed around a single action on purpose: the admission path preempts too, and bracketing decode alone is what reported a self's 66 preemptions as none.
        """
        drain = getattr(self.session, "drain_preemption_telemetry", None)
        if not callable(drain):
            return
        for name, value in dict(drain()).items():
            if value:
                self.scheduler_observations[name] += int(value)

    def phase_depths(self) -> tuple[int, int, int]:
        """The three depths, with no side effect. **The one definition.**

        Split out from `_run_phase_depths` so a second reader can sample them
        without also moving `demand_high_water`, which P4 sizes an arena from:
        a telemetry sample that changes an allocation is two variables, not one.
        """
        return (
            len(self.pending),
            sum(len(batch.selected) for batch in self.prefill_pending),
            self.session.active_count if self.session is not None else 0,
        )

    def record_phase_depths(self) -> tuple[int, int, int]:
        condition_ready, prefill_ready, decode_ready = (
            self.phase_depths()
        )
        # Recorded here because this is the one place the three depths are
        # resolved together, and their sum is the demand P4 sizes against. A
        # second definition elsewhere is how "ready width" would come to mean
        # two things.
        self.demand_high_water = max(
            self.demand_high_water, condition_ready + prefill_ready + decode_ready
        )
        return condition_ready, prefill_ready, decode_ready

    def tasks(self) -> tuple[_BatchTask[_ModelT, Any], ...]:
        return (
            *self.pending,
            *self.pending_controls,
            *self.in_flight,
            *(task for batch in self.prefill_pending for task, _item in batch.selected),
            *(task for task, _item in self.active_by_item.values()),
        )

    def pipeline_session_ids(self) -> set[str]:
        return {
            str(task.pipeline_session_id)
            for task in self.tasks()
            if getattr(task, "pipeline_session_id", None) is not None
        }

    def session_label(self) -> str:
        order = {"bootstrap": 0, "primary": 1, "recovery": 2, "control": 3}
        purposes = "/".join(
            sorted(self.purposes, key=lambda value: (order.get(value, 9), value))
        )
        return f"{self.key.model} {purposes}".rstrip()

    def job_types(self) -> set[GenerationJobType]:
        return {
            getattr(task, "job_type", "primary")
            for task in self.tasks()
        }

    def take_back(self, task: _BatchTask[_ModelT, Any]) -> None:
        """Put a task borrowed from this run back in its pending queue."""
        self.pending.append(task)
        self.pending.sort(key=lambda candidate: candidate.sequence)

    def ordered_pending(self) -> list[_BatchTask[_ModelT, Any]]:
        return sorted(self.pending, key=task_schedule_key)


def task_requires_dependency_credit(task: _BatchTask[_ModelT, Any]) -> bool:
    return bool(
        getattr(
            getattr(task, "item", None),
            "requires_dependency_credit",
            False,
        )
    )


def task_schedule_key(task: _BatchTask[_ModelT, Any]) -> tuple[object, ...]:
    return (
        getattr(task, "priority", 10),
        0 if getattr(task, "dependency_bundle_id", None) is not None else 1,
        getattr(task, "sequence", 0),
        getattr(task, "ready_order", ""),
    )


def task_decode_budget(task: _BatchTask[_ModelT, Any]) -> int:
    handle = getattr(task.item, "resident_handle", None)
    if handle is not None:
        return int(getattr(handle, "remaining_token_budget", 2048))
    return int(getattr(task.item, "remaining_decode_token_budget", 2048))


def task_prefill_budget(task: _BatchTask[_ModelT, Any]) -> int:
    if getattr(task.item, "resident_handle", None) is not None:
        return 0
    return int(getattr(task.item, "prefill_token_budget", 384))


def task_purpose(task: _BatchTask[_ModelT, Any]) -> str:
    if str(getattr(task.item, "candidate_name", "")) == "bootstrap":
        return "bootstrap"
    return task.job_type


def prepare_persistent_selection(
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
