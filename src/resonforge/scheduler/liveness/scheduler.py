"""Pure scheduler state, legal-action, and transition-result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SchedulerActionKind(StrEnum):
    """One state transition the runtime adapter can execute."""

    CANCEL = "cancel"
    DISCARD = "discard"
    FAIL = "fail"
    CONDITION_BATCH = "condition_batch"
    CONDITION_PREFILL = "condition_prefill"
    PREFILL = "prefill"
    RESTORE = "restore"
    PREEMPT_RESTORE = "preempt_restore"
    PREEMPT_CONDITION = "preempt_condition"
    BUNDLE_ADMIT = "bundle_admit"
    CAPACITY_PREPARE = "capacity_prepare"
    RELEASE_ADMIT = "release_admit"
    RESIZE = "resize"
    ADMIT = "admit"
    DECODE = "decode"


class ActionResultStatus(StrEnum):
    """Exhaustive outcome of one selected scheduler action."""

    APPLIED = "applied"
    TERMINAL = "terminal"
    WAITING_EXTERNAL = "waiting_external"
    STALE = "stale"
    REJECTED = "rejected"


class SchedulerExternalWaitPhase(StrEnum):
    """Exact event owner between checkpoint publication and consumption."""

    AWAITING_BIND = "awaiting_bind"
    PREPARATION_PENDING = "preparation_pending"
    DELIVERY_READY = "delivery_ready"


class SchedulerInvariantError(RuntimeError):
    """The runtime adapter exposed an impossible ownership state."""


@dataclass(frozen=True)
class SchedulerRunState:
    """Immutable facts needed to decide legal actions for one run."""

    run_id: int
    physical_width: int
    session_exists: bool
    pending: int
    controls: int
    prefill: int
    in_flight: int
    tracked: int
    active: int
    occupied: int
    physical_available: int
    logical_available: int
    paused: int
    displaced: int
    resident_handles: int
    control_intents: int
    unique_control_intents: int
    remaining_decode_tokens: int
    completed_quanta: int
    cancelled_pending: int
    control_action: str | None
    control_can_resume: bool
    preempt_allowed: bool
    admission_safe: bool
    capacity_ready: bool = True
    arena_admission_safe: bool = True
    # Blocks this run may still claim from its device pool, already net of
    # every other holder's reservation and of the floors other declared lanes
    # keep. `None` means nothing bounds it: off CUDA, or an unpriced lane.
    pool_available_blocks: int | None = None
    # Rows this lane's **page** supply can still fund. `None` means unpriced:
    # off CUDA, or a lane whose arenas are private.
    #
    # The second conservation law -- pages, against the block pool's bytes -- and until 2026-08-25 this
    # layer did not have it. Device bytes were here from the start
    # (`pool_available_blocks`); pages were consulted only when a *session* was
    # opened, and `_nonresident_arena_admission_safe` returns True for anything
    # resident -- so admitting a row into a live session asked nothing about
    # pages at all. The row then failed at its first block boundary with
    # `KVPagesExhausted`, which stops the device rather than refusing an
    # admission, because no legal state ever accounted for it.
    page_rows_available: int | None = None
    producer_waits: int = 0
    bundle_owned_handles: int = 0
    unbound_producer_waits: int = 0
    condition_group: str | None = None
    condition_batch_limit: int = 0
    # P4. The width this lane has been told to hold, and the device blocks the
    # change claims -- signed, so a narrowing carries a negative. `None` means
    # no outstanding instruction. Two numbers because rows and blocks are
    # separate conservation laws and no rate converts one
    # lane's rows into another lane's bytes.
    resize_target_width: int | None = None
    resize_block_delta: int = 0

    @property
    def resizing(self) -> bool:
        """A live session has been told to change width and has not yet.

        A lane with no session has no width to change: its next session opens
        at whatever the instruction asked for. Were a nonresident lane allowed
        to stay `resizing` it would suppress its own admissions with nothing
        able to clear the suppression -- a deadlock introduced by the fix for
        one.

        **The pool test belongs here, not only in legality.** Suppression and
        legality have to be the same condition: a lane suppressed because it is
        resizing, whose resize the pool then refuses, has no admission and no
        resize, and `scheduler has no resource-feasible transition` is what
        that looks like from outside. Measured, not reasoned about -- it is how
        the first run of this cut died.
        """
        return (
            self.session_exists
            and self.resize_target_width is not None
            and (
                self.resize_target_width != self.physical_width
                or self.resize_block_delta != 0
            )
            and _pool_admits(self, self.resize_block_delta)
        )

    @property
    def unresolved(self) -> bool:
        return bool(
            self.pending
            or self.controls
            or self.prefill
            or self.in_flight
            or self.tracked
            or self.occupied
        )


@dataclass(frozen=True)
class SchedulerState:
    """One immutable device-worker scheduling observation."""

    version: int
    progress_epoch: int
    runs: tuple[SchedulerRunState, ...]
    bundles: tuple[SchedulerBundleState, ...] = ()
    external_waits: tuple[SchedulerExternalWaitState, ...] = ()
    reclaim_admissions: tuple[SchedulerReclaimAdmissionState, ...] = ()
    # `version` is the identity of *decided* state -- what the scheduler owns
    # and only the scheduler moves. `measurement_epoch` is the identity of the
    # live device readings folded into this observation: free VRAM, and every
    # feasibility flag derived from it.
    #
    # They are separate because mixing them makes "did anything change" mean two
    # incompatible things at once. The optimistic-concurrency contract needs
    # "no scheduler action happened between snapshot and apply", which a varying
    # `mem_get_info` must not be able to falsify; the rejection memo needs "the
    # world that made this action illegal still holds", which a changed
    # measurement *must* be able to falsify. One counter cannot answer both, and
    # a scheduler on a shared device asks both continuously.
    measurement_epoch: int = 0

    @property
    def unresolved(self) -> bool:
        return bool(self.bundles or self.external_waits) or any(
            run.unresolved for run in self.runs
        )


@dataclass(frozen=True)
class SchedulerBundleState:
    """One claimed recovery group waiting for atomic member reservation."""

    bundle_id: int
    run_id: int
    admission_safe: bool


@dataclass(frozen=True)
class SchedulerExternalWaitState:
    """Typed non-scheduler owner whose completion emits a scheduler event."""

    owner_id: object
    run_id: int
    phase: SchedulerExternalWaitPhase
    failed: bool = False
    expired: bool = False


@dataclass(frozen=True)
class SchedulerReclaimAdmissionState:
    """One atomic owner-free arena release followed by target admission."""

    target_run_id: int
    donor_run_ids: tuple[int, ...]
    reclaimable_blocks: int
    required_blocks: int


@dataclass(frozen=True)
class SchedulerAction:
    run_id: int
    kind: SchedulerActionKind
    participants: tuple[int, ...] = ()


def _pages_admit(run: SchedulerRunState) -> bool:
    """Can this lane's page supply fund one more row?

    Deliberately only ever asked of an admission, never of a decode: pages
    return as rows end, so a lane refused here always has a running row whose
    completion clears the refusal. That is what keeps this from being the
    admission suppression that deadlocked the resize drain -- a lane with
    nothing running holds no pages, so its own supply is whole and it is never
    refused.
    """
    rows = run.page_rows_available
    return rows is None or rows > 0


def _pool_admits(run: SchedulerRunState, extra_blocks: int) -> bool:
    """The one resource question the model asks: do these blocks fit?

    Not a safe-sequence search. `pool_available_blocks` is already net of
    every outstanding promise, and an admitted holder's promise covers
    everything it will go on to need -- so a state that passes this comparison
    has a legal action from every state it can reach.
    """
    available = run.pool_available_blocks
    return available is None or extra_blocks <= available


@dataclass(frozen=True)
class ActionResult:
    """Typed runtime-adapter result for one selected legal action."""

    status: ActionResultStatus
    action: SchedulerAction
    observed_version: int
    reason: str = ""
    completion_source: str | None = None

    def __post_init__(self) -> None:
        if self.status is ActionResultStatus.WAITING_EXTERNAL:
            if not self.completion_source:
                raise ValueError("external wait requires a completion source")
        elif self.completion_source is not None:
            raise ValueError("only an external wait may carry wait metadata")


def enabled_actions(state: SchedulerState) -> tuple[SchedulerAction, ...]:
    """Enumerate every legal transition from one observed scheduler state."""
    actions: list[SchedulerAction] = []
    condition_groups: dict[str, list[SchedulerRunState]] = {}
    # **Never suppressed by a pending resize.** Suppression exists to stop a
    # lane taking *new* work so it can reach an empty slot set; a bundle admit
    # resolves a group whose parent handle is already resident, so blocking it
    # blocks the drain it was supposed to allow. Measured: a lane at width 2
    # with one paused bundle-owned handle and a target of 1 had no admission,
    # no resize, and no way to release the handle -- `scheduler has no
    # resource-feasible transition`.
    actions.extend(
        SchedulerAction(bundle.run_id, SchedulerActionKind.BUNDLE_ADMIT, (bundle.bundle_id,))
        for bundle in state.bundles
        if bundle.admission_safe
    )
    for run in state.runs:
        if run.cancelled_pending:
            actions.append(SchedulerAction(run.run_id, SchedulerActionKind.CANCEL))
        if run.session_exists and run.controls and run.control_action == "discard":
            actions.append(SchedulerAction(run.run_id, SchedulerActionKind.DISCARD))
        # Condition-ready rows already own logical slots. Their KV commit is a
        # barrier: no same-run action may consume or mutate those slots first.
        if run.prefill:
            actions.append(SchedulerAction(run.run_id, SchedulerActionKind.PREFILL))
            continue
        if (
            run.pending > run.cancelled_pending
            and not run.session_exists
            and not run.capacity_ready
        ):
            actions.append(
                SchedulerAction(run.run_id, SchedulerActionKind.CAPACITY_PREPARE)
            )
            continue
        if (
            run.session_exists
            and run.controls
            and run.control_action == "resume"
        ):
            if run.control_can_resume:
                actions.append(
                    SchedulerAction(run.run_id, SchedulerActionKind.RESTORE)
                )
            elif run.preempt_allowed:
                actions.append(
                    SchedulerAction(run.run_id, SchedulerActionKind.PREEMPT_RESTORE)
                )
        # A width change needs every slot empty -- the same rule as
        # `KVBlockPool.resize`, for the same reason. It is taken
        # **opportunistically**, at a moment the lane is already empty, and
        # ranked above admission so an empty lane with a target resizes before
        # it refills.
        #
        # **Admission is deliberately not suppressed to force that moment.**
        # Draining a lane on purpose was built, proved deadlock-free in the CPU
        # model, and killed three runs anyway: the model's lanes are
        # independent and production's are not -- one lane's output is the
        # other's input, so silencing the lane that is producing stalls the
        # device, and a lane whose only resident handle is owned by an
        # in-flight bundle can then never empty at all. What the suppression
        # would buy is the shrink case, and with the width ceilings gone the
        # device has blocks to spare, so there is nothing to reclaim. The
        # growth case does not need it either: the lane that has to grow is
        # already empty most of the time.
        if (
            run.resizing
            and not run.occupied
            and not run.prefill
            and not run.controls
            and not run.resident_handles
            and not run.producer_waits
        ):
            actions.append(SchedulerAction(run.run_id, SchedulerActionKind.RESIZE))
        if (
            run.pending
            and run.admission_safe
            and (run.session_exists or run.arena_admission_safe)
            and _pool_admits(run, 0)
            and _pages_admit(run)
            and (not run.session_exists or run.logical_available > 0)
        ):
            actions.append(
                SchedulerAction(run.run_id, SchedulerActionKind.CONDITION_PREFILL)
            )
        elif (
            run.pending
            and run.admission_safe
            and run.session_exists
            # **Either conservation law, not just slots.** This read
            # `physical_available == 0` alone, which is the pre-paging
            # question: back then a blocked lane was always a lane with no
            # free slot. Since R1 there are two supplies, and **a lane can sit
            # on plenty of free slots with zero pages** -- which is how a
            # deadlock reached `enabled_actions == [fail]` while the cure sat
            # one condition away (`docs/deadlock/`).
            and (run.physical_available == 0 or not _pages_admit(run))
            and run.occupied > run.active
            and run.preempt_allowed
            and _pool_admits(run, 0)
            # **Not `_pages_admit`.** Preemption is what *returns* pages, so
            # gating it on pages being available is the action that frees the
            # resource asking the resource to be free first. The admission it
            # unblocks is tested separately, after the pages are back.
        ):
            actions.append(
                SchedulerAction(run.run_id, SchedulerActionKind.PREEMPT_CONDITION)
            )
        if run.session_exists and run.active:
            actions.append(SchedulerAction(run.run_id, SchedulerActionKind.DECODE))
        if (
            run.condition_group is not None
            and run.condition_batch_limit > 1
            and run.session_exists
            and not run.prefill
            and run.pending
            and run.admission_safe
            and run.logical_available > 0
            and _pool_admits(run, 0)
            and _pages_admit(run)
        ):
            condition_groups.setdefault(run.condition_group, []).append(run)
    for candidates in condition_groups.values():
        candidates.sort(key=lambda run: run.run_id)
        limit = min(run.condition_batch_limit for run in candidates)
        for start in range(0, len(candidates), limit):
            cohort = candidates[start : start + limit]
            if len(cohort) < 2:
                continue
            available = cohort[0].pool_available_blocks
            if available is not None and available < 0:
                continue
            participants = tuple(run.run_id for run in cohort)
            actions.append(
                SchedulerAction(
                    participants[0],
                    SchedulerActionKind.CONDITION_BATCH,
                    participants,
                )
            )
    waits_cover_state = external_waits_cover_state(state)
    if not actions and not waits_cover_state:
        actions.extend(
            SchedulerAction(
                reclaim.target_run_id,
                SchedulerActionKind.RELEASE_ADMIT,
                reclaim.donor_run_ids,
            )
            for reclaim in state.reclaim_admissions
        )
    if not actions and not waits_cover_state:
        blocked = next((run for run in state.runs if run.pending), None)
        if blocked is None and state.bundles:
            blocked_id = state.bundles[0].run_id
        else:
            blocked_id = None if blocked is None else blocked.run_id
    else:
        blocked_id = None
    if blocked_id is not None:
        actions.append(SchedulerAction(blocked_id, SchedulerActionKind.FAIL))
    return tuple(actions)


def external_waits_cover_state(state: SchedulerState) -> bool:
    """Return whether events uniquely own every actionless dependency."""
    undecided_handles = sum(
        run.resident_handles - run.control_intents for run in state.runs
    )
    registered_waits = sum(run.producer_waits for run in state.runs)
    bundle_owned_handles = sum(
        run.bundle_owned_handles for run in state.runs
    )
    registered_handle_owners = registered_waits + bundle_owned_handles
    handle_waits = tuple(
        wait
        for wait in state.external_waits
        if isinstance(wait.owner_id, tuple)
        and len(wait.owner_id) == 2
        and wait.owner_id[0] == "handle"
    )
    return (
        registered_handle_owners == undecided_handles
        and len(handle_waits) == registered_waits
        and bool(registered_waits or state.external_waits)
        and not any(wait.failed or wait.expired for wait in state.external_waits)
    )


def validate_scheduler_state(state: SchedulerState) -> None:
    """Enforce width-independent resource and owner conservation."""
    bundle_ids = [bundle.bundle_id for bundle in state.bundles]
    if len(bundle_ids) != len(set(bundle_ids)):
        raise SchedulerInvariantError("scheduler has duplicate bundle identities")
    run_ids = {run.run_id for run in state.runs}
    runs_by_id = {run.run_id: run for run in state.runs}
    # Ownership only. A bundle used to have to promise at least one block, and
    # that block was a member's spill credit -- a copy the runtime no longer
    # makes. What a queued bundle owns is its run; what its members will need
    # is pages, taken from the pool when they run and never promised ahead.
    if any(bundle.run_id not in run_ids for bundle in state.bundles):
        raise SchedulerInvariantError("scheduler bundle has invalid ownership")
    reclaim_targets = [item.target_run_id for item in state.reclaim_admissions]
    if len(reclaim_targets) != len(set(reclaim_targets)):
        raise SchedulerInvariantError("scheduler has duplicate reclaim targets")
    for reclaim in state.reclaim_admissions:
        if (
            reclaim.target_run_id not in run_ids
            or not reclaim.donor_run_ids
            or reclaim.target_run_id in reclaim.donor_run_ids
            or len(reclaim.donor_run_ids) != len(set(reclaim.donor_run_ids))
            or any(run_id not in run_ids for run_id in reclaim.donor_run_ids)
            or reclaim.reclaimable_blocks < reclaim.required_blocks
            or reclaim.required_blocks < 1
        ):
            raise SchedulerInvariantError(
                "scheduler reclaim admission has invalid ownership or capacity"
            )
        target = runs_by_id[reclaim.target_run_id]
        if target.session_exists or not target.pending or not target.capacity_ready:
            raise SchedulerInvariantError(
                "scheduler reclaim target is not a prepared nonresident run"
            )
        for donor_id in reclaim.donor_run_ids:
            donor = runs_by_id[donor_id]
            if not donor.session_exists or any(
                (
                    donor.controls,
                    donor.prefill,
                    donor.in_flight,
                    donor.tracked,
                    donor.active,
                    donor.occupied,
                    donor.resident_handles,
                    donor.control_intents,
                    donor.producer_waits,
                    donor.bundle_owned_handles,
                )
            ):
                raise SchedulerInvariantError(
                    "scheduler reclaim donor retains session ownership"
                )
    external_ids = [wait.owner_id for wait in state.external_waits]
    if len(external_ids) != len(set(external_ids)) or any(
        wait.run_id not in run_ids for wait in state.external_waits
    ):
        raise SchedulerInvariantError("scheduler external wait has invalid ownership")
    handle_wait_counts = {
        run_id: sum(
            wait.run_id == run_id
            and isinstance(wait.owner_id, tuple)
            and len(wait.owner_id) == 2
            and wait.owner_id[0] == "handle"
            for wait in state.external_waits
        )
        for run_id in run_ids
    }
    for run in state.runs:
        if handle_wait_counts[run.run_id] != run.producer_waits:
            raise SchedulerInvariantError(
                f"run {run.run_id} producer wait count lacks exact typed owners"
            )
        counters = {
            "physical_width": run.physical_width,
            "pending": run.pending,
            "controls": run.controls,
            "prefill": run.prefill,
            "in_flight": run.in_flight,
            "tracked": run.tracked,
            "active": run.active,
            "occupied": run.occupied,
            "physical_available": run.physical_available,
            "logical_available": run.logical_available,
            "paused": run.paused,
            "displaced": run.displaced,
            "resident_handles": run.resident_handles,
            "remaining_decode_tokens": run.remaining_decode_tokens,
            "completed_quanta": run.completed_quanta,
        }
        negative = [name for name, value in counters.items() if value < 0]
        if negative:
            raise SchedulerInvariantError(
                f"run {run.run_id} has negative counters: {', '.join(negative)}"
            )
        # `pool_available_blocks` is deliberately absent: it is a *balance*,
        # not a count, and it goes negative when outstanding promises exceed
        # what the device now has free -- reachable whenever another process
        # takes memory after a reservation was granted. That is a state to
        # refuse admissions from, not an impossible one. Every value below is
        # a count, where negative really is impossible.
        resource_values = {
            "producer_waits": run.producer_waits,
            "bundle_owned_handles": run.bundle_owned_handles,
            "unbound_producer_waits": run.unbound_producer_waits,
        }
        negative_resources = [
            name for name, value in resource_values.items() if value < 0
        ]
        if negative_resources:
            raise SchedulerInvariantError(
                f"run {run.run_id} has negative resources: "
                f"{', '.join(negative_resources)}"
            )
        if run.condition_batch_limit < 0:
            raise SchedulerInvariantError(
                f"run {run.run_id} has a negative condition batch limit"
            )
        if run.resize_target_width is not None and run.resize_target_width < 1:
            raise SchedulerInvariantError(
                f"run {run.run_id} has a resize target below one row"
            )
        if run.unbound_producer_waits > run.producer_waits:
            raise SchedulerInvariantError(
                f"run {run.run_id} has more unbound waits than producer waits"
            )
        if run.control_intents <= run.resident_handles and (
            run.producer_waits + run.bundle_owned_handles
            > run.resident_handles - run.control_intents
        ):
            raise SchedulerInvariantError(
                f"run {run.run_id} has blocking owners without undecided handles"
            )
        if not run.session_exists:
            if any(
                (
                    run.active,
                    run.occupied,
                    run.paused,
                    run.displaced,
                    run.resident_handles,
                    run.prefill,
                )
            ):
                raise SchedulerInvariantError(
                    f"run {run.run_id} owns session state without a session"
                )
            if run.logical_available != run.physical_width:
                raise SchedulerInvariantError(
                    f"run {run.run_id} has invalid nonresident availability"
                )
            continue
        if not run.capacity_ready:
            raise SchedulerInvariantError(
                f"run {run.run_id} owns a session without ready capacity"
            )
        if not run.active <= run.occupied <= run.physical_width:
            raise SchedulerInvariantError(
                f"run {run.run_id} violates active/occupied/width ordering"
            )
        if run.paused != run.occupied - run.active:
            raise SchedulerInvariantError(
                f"run {run.run_id} paused rows do not match occupied ownership"
            )
        if run.physical_available != run.physical_width - run.occupied:
            raise SchedulerInvariantError(
                f"run {run.run_id} physical availability is inconsistent"
            )
        schedulable_free = run.physical_available - run.prefill
        if run.logical_available != schedulable_free:
            raise SchedulerInvariantError(
                f"run {run.run_id} logical availability ignores committed prefill"
            )
        if schedulable_free + run.prefill + run.active + run.paused != run.physical_width:
            raise SchedulerInvariantError(
                f"run {run.run_id} violates slot conservation"
            )
        if run.tracked != run.active:
            raise SchedulerInvariantError(
                f"run {run.run_id} active rows lack unique task owners"
            )
        if run.resident_handles != run.paused + run.displaced:
            raise SchedulerInvariantError(
                f"run {run.run_id} resident handle location is not unique"
            )
        if run.control_intents != run.unique_control_intents:
            raise SchedulerInvariantError(
                f"run {run.run_id} resident handle has duplicate control intent"
            )
