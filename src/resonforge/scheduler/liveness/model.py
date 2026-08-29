"""Bounded pure-state model for scheduler liveness verification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class ModelActionKind(StrEnum):
    PARENT_ADMIT = "parent_admit"
    PARENT_CHECKPOINT = "parent_checkpoint"
    PARENT_COMPLETE = "parent_complete"
    PARENT_RESUME = "parent_resume"
    PARENT_RECHECKPOINT = "parent_recheckpoint"
    PARENT_RESUME_COMPLETE = "parent_resume_complete"
    BUNDLE_ADMIT = "bundle_admit"
    BUNDLE_CHECKPOINT = "bundle_checkpoint"
    BUNDLE_PUBLISH = "bundle_publish"
    BUNDLE_BIND = "bundle_bind"
    BUNDLE_RELEASE = "bundle_release"
    CAPACITY_PREPARE = "capacity_prepare"
    RELEASE_ADMIT = "release_admit"
    RESIZE = "resize"
    CONDITION = "condition"
    CONDITION_BATCH = "condition_batch"
    CONDITION_PREFILL = "condition_prefill"
    PREFILL = "prefill"
    DECODE_COMPLETE = "decode_complete"
    CHECKPOINT_WAIT = "checkpoint_wait"
    BIND_PRODUCER_WAIT = "bind_producer_wait"
    PRODUCER_COMPLETE = "producer_complete"
    PRODUCER_RESUME = "producer_resume"
    PRODUCER_DISCARD = "producer_discard"
    WATCHDOG = "watchdog"
    PRODUCER_FAILURE = "producer_failure"
    PREEMPT_RESTORE = "preempt_restore"
    PREEMPT_CONDITION = "preempt_condition"
    RESTORE_PAUSED = "restore_paused"
    RESTORE_DISPLACED = "restore_displaced"
    DISCARD_PAUSED = "discard_paused"
    DISCARD_DISPLACED = "discard_displaced"
    FAIL = "fail"


class ModelBundlePhase(StrEnum):
    PARENT_QUEUED = "parent_queued"
    PARENT_ACTIVE = "parent_active"
    PARENT_WAIT = "parent_wait"
    PARENT_RESUMED = "parent_resumed"
    PARENT_CLAIM = "parent_claim"
    BUNDLE_ACTIVE = "bundle_active"
    BUNDLE_CHECKPOINTED = "bundle_checkpointed"
    PUBLISHED = "published"
    TRANSFERRED = "transferred"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class ModelBundleState:
    """One parent claim and its all-or-none downstream recovery group."""

    parent_run: int
    member_runs: tuple[int, ...]
    member_checkpointed: tuple[bool, ...] = ()
    parent_resident: bool = False
    phase: ModelBundlePhase = ModelBundlePhase.PARENT_QUEUED

    @property
    def unresolved(self) -> bool:
        return self.phase not in {
            ModelBundlePhase.TRANSFERRED,
            ModelBundlePhase.TERMINAL,
        }


@dataclass(frozen=True)
class ModelResizeTarget:
    """One instruction to a lane: hold this many rows, at this arena cost.

    Both numbers, because device bytes and KV pages are separate conservation laws and a width alone does not say what the arena costs --
    a `medium` row is 24 layers of 1,024-wide pages against 14 of 768, so no
    rate converts one lane's rows into another's blocks.
    """

    width: int
    arena_blocks: int


@dataclass(frozen=True)
class ModelRunState:
    width: int
    pending: int = 0
    committed_prefill: int = 0
    active: int = 0
    paused: int = 0
    displaced: int = 0
    restore_controls: int = 0
    discard_controls: int = 0
    producer_waits: int = 0
    bound_producer_waits: int = 0
    delivery_ready_waits: int = 0
    credits: int = 0
    admission_safe: bool = True
    preempt_condition_safe: bool = True
    failure_pending: bool = False
    terminal: int = 0
    condition_group: int | None = None
    condition_batch_limit: int = 0
    resource_blocked: bool = False
    capacity_ready: bool = True
    arena_admission_safe: bool = True
    session_resident: bool = True
    arena_blocks: int = 0
    resize_target: ModelResizeTarget | None = None

    @property
    def free(self) -> int:
        return self.width - self.committed_prefill - self.active - self.paused

    @property
    def resizing(self) -> bool:
        """A live session has been told to change width and has not yet.

        Only a resident session: a lane with no session has no width to
        change, and its next session is opened at whatever the instruction
        asked for. Were a nonresident lane allowed to stay `resizing` it would
        suppress its own admissions with no action able to clear the
        suppression -- a deadlock introduced by the fix for one.
        """
        target = self.resize_target
        return (
            self.session_resident
            and target is not None
            and (
                target.width != self.width
                or target.arena_blocks != self.arena_blocks
            )
        )

    @property
    def unresolved(self) -> bool:
        return bool(
            self.pending
            or self.committed_prefill
            or self.active
            or self.restore_controls
            or self.discard_controls
            or self.producer_waits
            or self.paused
            or self.displaced
            or self.failure_pending
        )


@dataclass(frozen=True)
class ModelState:
    runs: tuple[ModelRunState, ...]
    bundles: tuple[ModelBundleState, ...] = ()
    # Blocks the device pool holds in total. `None` means unbounded: off
    # CUDA, or a lane nothing priced.
    pool_blocks: int | None = None
    # Blocks held by something no run in this state owns.
    fixed_blocks: int = 0

    @property
    def unresolved(self) -> bool:
        return any(run.unresolved for run in self.runs) or any(
            bundle.unresolved for bundle in self.bundles
        )


@dataclass(frozen=True)
class ModelAction:
    run: int
    kind: ModelActionKind
    participants: tuple[int, ...] = ()
    bundle: int | None = None
    member: int | None = None


def _bundle_member_counts(
    state: ModelState,
    *,
    phase: ModelBundlePhase,
    checkpointed: bool | None = None,
) -> tuple[int, ...]:
    counts = [0] * len(state.runs)
    for bundle in state.bundles:
        if bundle.phase is not phase:
            continue
        for member, run_index in enumerate(bundle.member_runs):
            if checkpointed is not None and bundle.member_checkpointed[member] is not checkpointed:
                continue
            counts[run_index] += 1
    return tuple(counts)


def _bundle_active_counts(state: ModelState) -> tuple[int, ...]:
    counts = [0] * len(state.runs)
    for bundle in state.bundles:
        if bundle.phase in {
            ModelBundlePhase.PARENT_ACTIVE,
            ModelBundlePhase.PARENT_RESUMED,
        }:
            counts[bundle.parent_run] += 1
        elif bundle.phase is ModelBundlePhase.BUNDLE_ACTIVE:
            for member, run_index in enumerate(bundle.member_runs):
                if not bundle.member_checkpointed[member]:
                    counts[run_index] += 1
    return tuple(counts)


def _bundle_internal_handle_counts(state: ModelState) -> tuple[int, ...]:
    counts = [0] * len(state.runs)
    for bundle in state.bundles:
        if bundle.phase not in {
            ModelBundlePhase.PARENT_WAIT,
            ModelBundlePhase.BUNDLE_ACTIVE,
            ModelBundlePhase.BUNDLE_CHECKPOINTED,
        }:
            continue
        if bundle.parent_resident:
            counts[bundle.parent_run] += 1
        if not bundle.member_checkpointed:
            continue
        for member, run_index in enumerate(bundle.member_runs):
            if bundle.member_checkpointed[member]:
                counts[run_index] += 1
    return tuple(counts)


def _bundle_wait_counts(state: ModelState) -> tuple[int, ...]:
    counts = [0] * len(state.runs)
    for bundle in state.bundles:
        if bundle.phase not in {
            ModelBundlePhase.PUBLISHED,
            ModelBundlePhase.TRANSFERRED,
        }:
            continue
        if bundle.parent_resident:
            counts[bundle.parent_run] += 1
        for run_index in bundle.member_runs:
            counts[run_index] += 1
    return tuple(counts)


def _pool_admits(state: ModelState, extra_blocks: int = 0) -> bool:
    """The whole resource question: one comparison, mirroring `PoolView.admits`.

    Only a resident arena holds device blocks. A recovery bundle holds none:
    its members run inside an arena that already exists and draw **pages**,
    which is the page pool's conservation law, not this one. With nothing declaring a future block demand there is no
    Need to fund, so there is no hold-and-wait on device bytes and no safe
    sequence to search for.

    Retired here with the production banker (`_claim_admission_safe`, paging
    C4-2a): the safe-sequence search existed to admit a parent against blocks
    its members would later demand, and members no longer demand any.
    """
    budget = state.pool_blocks
    if budget is None:
        return True
    return state.fixed_blocks + _resident_blocks(state) + extra_blocks <= budget


def _resident_blocks(state: ModelState) -> int:
    return sum(
        run.arena_blocks for run in state.runs if run.session_resident
    )


def _session_owner_free(run: ModelRunState) -> bool:
    return run.session_resident and not any(
        (
            run.committed_prefill,
            run.active,
            run.paused,
            run.displaced,
            run.restore_controls,
            run.discard_controls,
            run.producer_waits,
        )
    )


def _can_create_session(state: ModelState, run_index: int) -> bool:
    run = state.runs[run_index]
    if run.session_resident or run.arena_blocks == 0:
        return True
    return _pool_admits(state, run.arena_blocks)


def validate_model_state(state: ModelState) -> None:
    if state.pool_blocks is not None and state.pool_blocks < 0:
        raise ValueError("model has a negative pool")
    if state.fixed_blocks < 0:
        raise ValueError("model has negative fixed blocks")
    if not _pool_admits(state):
        raise ValueError("model resident arenas exceed the transient budget")
    active_bundle_members = _bundle_active_counts(state)
    internal_bundle_handles = _bundle_internal_handle_counts(state)
    bundle_waits = _bundle_wait_counts(state)
    for bundle_index, bundle in enumerate(state.bundles):
        if bundle.parent_run < 0 or bundle.parent_run >= len(state.runs):
            raise ValueError(f"bundle {bundle_index} has an invalid parent run")
        if len(bundle.member_runs) < 2:
            raise ValueError(f"bundle {bundle_index} requires at least two members")
        if any(index < 0 or index >= len(state.runs) for index in bundle.member_runs):
            raise ValueError(f"bundle {bundle_index} has an invalid run index")
        if bundle.phase in {
            ModelBundlePhase.PARENT_QUEUED,
            ModelBundlePhase.PARENT_ACTIVE,
            ModelBundlePhase.PARENT_WAIT,
            ModelBundlePhase.PARENT_RESUMED,
            ModelBundlePhase.PARENT_CLAIM,
            ModelBundlePhase.TERMINAL,
        }:
            if bundle.member_checkpointed:
                raise ValueError(
                    f"inactive bundle {bundle_index} owns member state"
                )
        elif len(bundle.member_checkpointed) != len(bundle.member_runs):
            raise ValueError(f"bundle {bundle_index} has incomplete member state")
        if (
            bundle.phase is ModelBundlePhase.BUNDLE_ACTIVE
            and all(bundle.member_checkpointed)
        ):
            raise ValueError(f"bundle {bundle_index} must publish completed members")
        if bundle.phase in {
            ModelBundlePhase.BUNDLE_CHECKPOINTED,
            ModelBundlePhase.PUBLISHED,
            ModelBundlePhase.TRANSFERRED,
        } and not all(bundle.member_checkpointed):
            raise ValueError(f"bundle {bundle_index} published partial members")
    for index, run in enumerate(state.runs):
        values = (
            run.width,
            run.pending,
            run.committed_prefill,
            run.active,
            run.paused,
            run.displaced,
            run.restore_controls,
            run.discard_controls,
            run.producer_waits,
            run.bound_producer_waits,
            run.delivery_ready_waits,
            run.credits,
            run.terminal,
        )
        if run.width < 1 or any(value < 0 for value in values):
            raise ValueError(f"run {index} has a negative model counter")
        if run.arena_blocks < 0:
            raise ValueError(f"run {index} has negative arena resources")
        target = run.resize_target
        if target is not None and (target.width < 1 or target.arena_blocks < 0):
            raise ValueError(f"run {index} has an invalid resize target")
        if run.condition_batch_limit < 0:
            raise ValueError(f"run {index} has a negative condition batch limit")
        if run.free < 0:
            raise ValueError(f"run {index} overcommits physical slots")
        handles = run.paused + run.displaced
        if run.restore_controls + run.discard_controls > handles:
            raise ValueError(f"run {index} has more intents than handles")
        if run.bound_producer_waits > run.producer_waits:
            raise ValueError(f"run {index} has more bound waits than waits")
        if run.delivery_ready_waits > run.bound_producer_waits:
            raise ValueError(f"run {index} has more ready deliveries than bound waits")
        if active_bundle_members[index] > run.active:
            raise ValueError(f"run {index} has unreserved active bundle members")
        if internal_bundle_handles[index] > handles:
            raise ValueError(f"run {index} has unowned bundle handles")
        if run.producer_waits > (
            handles
            - internal_bundle_handles[index]
            - run.restore_controls
            - run.discard_controls
        ):
            raise ValueError(f"run {index} has waits without undecided handles")
        if bundle_waits[index] > run.producer_waits:
            raise ValueError(f"run {index} lacks published bundle waits")
        if run.credits != run.committed_prefill + run.active + handles:
            raise ValueError(f"run {index} violates dependency-credit ownership")
        if not run.session_resident and any(
            (
                run.committed_prefill,
                run.active,
                run.paused,
                run.displaced,
                run.restore_controls,
                run.discard_controls,
                run.producer_waits,
            )
        ):
            raise ValueError(f"run {index} owns state without a resident session")


def model_enabled_actions(state: ModelState) -> tuple[ModelAction, ...]:
    validate_model_state(state)
    actions: list[ModelAction] = []
    condition_groups: dict[int, list[int]] = {}
    active_bundle_members = _bundle_active_counts(state)
    published_bundle_waits = _bundle_member_counts(
        state,
        phase=ModelBundlePhase.PUBLISHED,
    )
    transferred_bundle_waits = _bundle_member_counts(
        state,
        phase=ModelBundlePhase.TRANSFERRED,
    )
    for bundle in state.bundles:
        if bundle.phase is ModelBundlePhase.PUBLISHED:
            published_bundle_waits = tuple(
                count + int(index == bundle.parent_run)
                for index, count in enumerate(published_bundle_waits)
            )
        elif bundle.phase is ModelBundlePhase.TRANSFERRED:
            transferred_bundle_waits = tuple(
                count + int(index == bundle.parent_run)
                for index, count in enumerate(transferred_bundle_waits)
            )
    # One recovery group at a time, device-wide. Removing this was tried twice
    # and bought nothing: the `medium` lane barely widened either time, because
    # it is **demand-limited, not rule-limited**. The first attempt ran at two
    # songs, where the lane's demand sits below its ceiling anyway and the rule
    # could not have been binding -- the re-ask at six songs, in the regime that
    # could detect it, gave the same answer. Arms: `docs/scheduler-fill/`.
    #
    # The rule stays: it costs nothing and it keeps one dimension out of the
    # state space this model has to prove.
    bundle_transaction_active = any(
        bundle.phase
        in {
            ModelBundlePhase.BUNDLE_ACTIVE,
            ModelBundlePhase.BUNDLE_CHECKPOINTED,
        }
        for bundle in state.bundles
    )
    for bundle_index, bundle in enumerate(state.bundles):
        if bundle.phase is ModelBundlePhase.PARENT_QUEUED:
            run = state.runs[bundle.parent_run]
            # A bundle admit is **never** suppressed by a pending resize: it
            # resolves a group already claimed against this run, and a lane
            # whose only resident handle is bundle-owned can reach an empty
            # slot set no other way. See `enabled_actions` for the state that
            # proved it.
            if (
                run.capacity_ready
                and run.free
                and run.admission_safe
                and not run.resource_blocked
            ):
                actions.append(
                    ModelAction(
                        bundle.parent_run,
                        ModelActionKind.PARENT_ADMIT,
                        bundle=bundle_index,
                    )
                )
        elif bundle.phase is ModelBundlePhase.PARENT_ACTIVE:
            actions.extend(
                (
                    ModelAction(
                        bundle.parent_run,
                        ModelActionKind.PARENT_CHECKPOINT,
                        bundle=bundle_index,
                    ),
                    ModelAction(
                        bundle.parent_run,
                        ModelActionKind.PARENT_COMPLETE,
                        bundle=bundle_index,
                    ),
                )
            )
        elif bundle.phase in {
            ModelBundlePhase.PARENT_WAIT,
            ModelBundlePhase.PARENT_CLAIM,
        }:
            if bundle.phase is ModelBundlePhase.PARENT_WAIT:
                actions.append(
                    ModelAction(
                        bundle.parent_run,
                        ModelActionKind.PARENT_RESUME,
                        bundle=bundle_index,
                    )
                )
            required = {
                run_index: bundle.member_runs.count(run_index)
                for run_index in set(bundle.member_runs)
            }
            if not bundle_transaction_active and all(
                state.runs[run_index].free >= count
                and state.runs[run_index].capacity_ready
                and state.runs[run_index].admission_safe
                and not state.runs[run_index].resource_blocked
                for run_index, count in required.items()
            ):
                actions.append(
                    ModelAction(
                        bundle.member_runs[0],
                        ModelActionKind.BUNDLE_ADMIT,
                        bundle.member_runs,
                        bundle_index,
                    )
                )
        elif bundle.phase is ModelBundlePhase.PARENT_RESUMED:
            actions.extend(
                (
                    ModelAction(
                        bundle.parent_run,
                        ModelActionKind.PARENT_RECHECKPOINT,
                        bundle=bundle_index,
                    ),
                    ModelAction(
                        bundle.parent_run,
                        ModelActionKind.PARENT_RESUME_COMPLETE,
                        bundle=bundle_index,
                    ),
                )
            )
        elif bundle.phase is ModelBundlePhase.BUNDLE_ACTIVE:
            for member, checkpointed in enumerate(bundle.member_checkpointed):
                if not checkpointed:
                    actions.append(
                        ModelAction(
                            bundle.member_runs[member],
                            ModelActionKind.BUNDLE_CHECKPOINT,
                            bundle=bundle_index,
                            member=member,
                        )
                    )
        elif bundle.phase is ModelBundlePhase.BUNDLE_CHECKPOINTED:
            actions.append(
                ModelAction(
                    bundle.member_runs[0],
                    ModelActionKind.BUNDLE_PUBLISH,
                    bundle=bundle_index,
                )
            )
        elif bundle.phase is ModelBundlePhase.PUBLISHED:
            actions.append(
                ModelAction(
                    bundle.member_runs[0],
                    ModelActionKind.BUNDLE_BIND,
                    bundle=bundle_index,
                )
            )
        elif bundle.phase is ModelBundlePhase.TRANSFERRED:
            actions.append(
                ModelAction(
                    bundle.parent_run,
                    ModelActionKind.BUNDLE_RELEASE,
                    bundle=bundle_index,
                )
            )
    for index, run in enumerate(state.runs):
        if run.failure_pending:
            actions.append(ModelAction(index, ModelActionKind.FAIL))
            continue
        if run.pending and not run.capacity_ready:
            actions.append(ModelAction(index, ModelActionKind.CAPACITY_PREPARE))
            continue
        if run.producer_waits:
            ordinary_unbound_waits = (
                run.producer_waits
                - run.bound_producer_waits
                - published_bundle_waits[index]
            )
            if ordinary_unbound_waits > 0:
                actions.append(ModelAction(index, ModelActionKind.BIND_PRODUCER_WAIT))
            ordinary_bound_waits = (
                run.bound_producer_waits - transferred_bundle_waits[index]
            )
            ordinary_ready_waits = (
                run.delivery_ready_waits - transferred_bundle_waits[index]
            )
            if ordinary_bound_waits > ordinary_ready_waits:
                actions.append(
                    ModelAction(index, ModelActionKind.PRODUCER_COMPLETE)
                )
            if ordinary_ready_waits > 0:
                actions.extend(
                    (
                        ModelAction(index, ModelActionKind.PRODUCER_RESUME),
                        ModelAction(index, ModelActionKind.PRODUCER_DISCARD),
                        ModelAction(index, ModelActionKind.PRODUCER_FAILURE),
                    )
                )
        if run.committed_prefill:
            actions.append(ModelAction(index, ModelActionKind.PREFILL))
            continue
        if run.discard_controls:
            if run.paused:
                actions.append(ModelAction(index, ModelActionKind.DISCARD_PAUSED))
            if run.displaced:
                actions.append(ModelAction(index, ModelActionKind.DISCARD_DISPLACED))
        if run.restore_controls:
            if run.paused:
                actions.append(ModelAction(index, ModelActionKind.RESTORE_PAUSED))
            if run.displaced and run.free:
                actions.append(ModelAction(index, ModelActionKind.RESTORE_DISPLACED))
            elif run.displaced and run.paused:
                actions.append(ModelAction(index, ModelActionKind.PREEMPT_RESTORE))
        # A width change needs every slot empty -- the same rule as
        # `KVBlockPool.resize`, for the same reason -- so the lane must first
        # drain. Nothing preempts it: its rows return at their own quantum
        # boundaries, and the instruction below suppresses admission so they
        # are not immediately replaced. A lane that keeps taking work never
        # reaches an empty slot set, which is why the suppression is the
        # mechanism and not a policy.
        if run.resizing and _session_owner_free(run):
            target = run.resize_target
            assert target is not None
            if _pool_admits(state, target.arena_blocks - run.arena_blocks):
                actions.append(ModelAction(index, ModelActionKind.RESIZE))
        if (
            run.pending
            and run.admission_safe
            and (run.session_resident or run.arena_admission_safe)
            and run.free
            and not run.resource_blocked
            and _can_create_session(state, index)
        ):
            actions.extend(
                (
                    ModelAction(index, ModelActionKind.CONDITION),
                    ModelAction(index, ModelActionKind.CONDITION_PREFILL),
                )
            )
            if run.condition_group is not None and run.condition_batch_limit > 1:
                condition_groups.setdefault(run.condition_group, []).append(index)
        elif run.pending and run.paused and run.preempt_condition_safe:
            actions.append(ModelAction(index, ModelActionKind.PREEMPT_CONDITION))
        ordinary_active = run.active - active_bundle_members[index]
        if ordinary_active:
            actions.extend(
                (
                    ModelAction(index, ModelActionKind.DECODE_COMPLETE),
                    ModelAction(index, ModelActionKind.CHECKPOINT_WAIT),
                )
            )
    for candidates in condition_groups.values():
        limit = min(state.runs[index].condition_batch_limit for index in candidates)
        for start in range(0, len(candidates), limit):
            cohort = tuple(candidates[start : start + limit])
            if len(cohort) > 1:
                actions.append(
                    ModelAction(
                        cohort[0],
                        ModelActionKind.CONDITION_BATCH,
                        cohort,
                    )
                )
    if not actions:
        owner_free_donors = tuple(
            index
            for index, run in enumerate(state.runs)
            if _session_owner_free(run) and run.arena_blocks > 0
        )
        for target_index, target in enumerate(state.runs):
            donors = tuple(
                index for index in owner_free_donors if index != target_index
            )
            if (
                not target.pending
                or not target.capacity_ready
                or target.session_resident
                or not target.admission_safe
                or target.resource_blocked
                or not donors
                or target.arena_blocks < 1
            ):
                continue
            remaining = sum(
                run.arena_blocks
                for index, run in enumerate(state.runs)
                if run.session_resident and index not in donors
            )
            budget = state.pool_blocks
            if budget is not None and (
                state.fixed_blocks + remaining + target.arena_blocks > budget
            ):
                continue
            actions.append(
                ModelAction(
                    target_index,
                    ModelActionKind.RELEASE_ADMIT,
                    donors,
                )
            )
    watchdog_run = next(
        (index for index, run in enumerate(state.runs) if run.producer_waits),
        None,
    )
    if watchdog_run is not None:
        actions.append(ModelAction(watchdog_run, ModelActionKind.WATCHDOG))
    blocked_run = next(
        (index for index, run in enumerate(state.runs) if run.unresolved),
        None,
    )
    if not actions and blocked_run is not None:
        actions.append(ModelAction(blocked_run, ModelActionKind.FAIL))
    return tuple(actions)


def apply_model_action(state: ModelState, action: ModelAction) -> ModelState:
    kind = action.kind
    if action not in model_enabled_actions(state):
        raise ValueError("model action is not enabled")
    if kind is ModelActionKind.CAPACITY_PREPARE:
        runs = list(state.runs)
        runs[action.run] = replace(runs[action.run], capacity_ready=True)
        result = replace(state, runs=tuple(runs))
        validate_model_state(result)
        return result
    if kind is ModelActionKind.RELEASE_ADMIT:
        runs = list(state.runs)
        for donor in action.participants:
            runs[donor] = replace(runs[donor], session_resident=False)
        target = runs[action.run]
        runs[action.run] = replace(
            target,
            pending=target.pending - 1,
            active=target.active + 1,
            credits=target.credits + 1,
            session_resident=True,
        )
        result = replace(state, runs=tuple(runs))
        validate_model_state(result)
        return result
    if kind is ModelActionKind.PARENT_ADMIT:
        assert action.bundle is not None
        bundle = state.bundles[action.bundle]
        run = state.runs[bundle.parent_run]
        runs = list(state.runs)
        runs[bundle.parent_run] = replace(
            run,
            active=run.active + 1,
            credits=run.credits + 1,
        )
        bundles = list(state.bundles)
        bundles[action.bundle] = replace(
            bundle,
            phase=ModelBundlePhase.PARENT_ACTIVE,
        )
        result = replace(state, runs=tuple(runs), bundles=tuple(bundles))
        validate_model_state(result)
        return result
    if kind is ModelActionKind.PARENT_CHECKPOINT:
        assert action.bundle is not None
        bundle = state.bundles[action.bundle]
        run = state.runs[bundle.parent_run]
        runs = list(state.runs)
        runs[bundle.parent_run] = replace(
            run,
            active=run.active - 1,
            paused=run.paused + 1,
        )
        bundles = list(state.bundles)
        bundles[action.bundle] = replace(
            bundle,
            phase=ModelBundlePhase.PARENT_WAIT,
            parent_resident=True,
        )
        result = replace(state, runs=tuple(runs), bundles=tuple(bundles))
        validate_model_state(result)
        return result
    if kind is ModelActionKind.PARENT_COMPLETE:
        assert action.bundle is not None
        bundle = state.bundles[action.bundle]
        run = state.runs[bundle.parent_run]
        runs = list(state.runs)
        runs[bundle.parent_run] = replace(
            run,
            active=run.active - 1,
            credits=run.credits - 1,
            terminal=run.terminal + 1,
        )
        bundles = list(state.bundles)
        bundles[action.bundle] = replace(
            bundle,
            phase=ModelBundlePhase.PARENT_CLAIM,
            parent_resident=False,
        )
        result = replace(state, runs=tuple(runs), bundles=tuple(bundles))
        validate_model_state(result)
        return result
    if kind is ModelActionKind.PARENT_RESUME:
        assert action.bundle is not None
        bundle = state.bundles[action.bundle]
        run = state.runs[bundle.parent_run]
        runs = list(state.runs)
        runs[bundle.parent_run] = replace(
            run,
            active=run.active + 1,
            paused=run.paused - 1,
        )
        bundles = list(state.bundles)
        bundles[action.bundle] = replace(
            bundle,
            phase=ModelBundlePhase.PARENT_RESUMED,
            parent_resident=False,
        )
        result = replace(state, runs=tuple(runs), bundles=tuple(bundles))
        validate_model_state(result)
        return result
    if kind is ModelActionKind.PARENT_RECHECKPOINT:
        assert action.bundle is not None
        bundle = state.bundles[action.bundle]
        run = state.runs[bundle.parent_run]
        runs = list(state.runs)
        runs[bundle.parent_run] = replace(
            run,
            active=run.active - 1,
            paused=run.paused + 1,
        )
        bundles = list(state.bundles)
        bundles[action.bundle] = replace(
            bundle,
            phase=ModelBundlePhase.PARENT_WAIT,
            parent_resident=True,
        )
        result = replace(state, runs=tuple(runs), bundles=tuple(bundles))
        validate_model_state(result)
        return result
    if kind is ModelActionKind.PARENT_RESUME_COMPLETE:
        assert action.bundle is not None
        bundle = state.bundles[action.bundle]
        run = state.runs[bundle.parent_run]
        runs = list(state.runs)
        runs[bundle.parent_run] = replace(
            run,
            active=run.active - 1,
            credits=run.credits - 1,
            terminal=run.terminal + 1,
        )
        bundles = list(state.bundles)
        bundles[action.bundle] = replace(
            bundle,
            phase=ModelBundlePhase.PARENT_CLAIM,
            parent_resident=False,
        )
        result = replace(state, runs=tuple(runs), bundles=tuple(bundles))
        validate_model_state(result)
        return result
    if kind is ModelActionKind.BUNDLE_ADMIT:
        assert action.bundle is not None
        runs = list(state.runs)
        bundle = state.bundles[action.bundle]
        for run_index in bundle.member_runs:
            run = runs[run_index]
            runs[run_index] = replace(
                run,
                active=run.active + 1,
                credits=run.credits + 1,
            )
        bundles = list(state.bundles)
        bundles[action.bundle] = replace(
            bundle,
            phase=ModelBundlePhase.BUNDLE_ACTIVE,
            member_checkpointed=(False,) * len(bundle.member_runs),
        )
        result = replace(state, runs=tuple(runs), bundles=tuple(bundles))
        validate_model_state(result)
        return result
    if kind is ModelActionKind.BUNDLE_CHECKPOINT:
        assert action.bundle is not None and action.member is not None
        bundle = state.bundles[action.bundle]
        run_index = bundle.member_runs[action.member]
        run = state.runs[run_index]
        runs = list(state.runs)
        runs[run_index] = replace(
            run,
            active=run.active - 1,
            paused=run.paused + 1,
        )
        checkpointed = list(bundle.member_checkpointed)
        checkpointed[action.member] = True
        bundles = list(state.bundles)
        bundles[action.bundle] = replace(
            bundle,
            member_checkpointed=tuple(checkpointed),
            phase=(
                ModelBundlePhase.BUNDLE_CHECKPOINTED
                if all(checkpointed)
                else ModelBundlePhase.BUNDLE_ACTIVE
            ),
        )
        result = replace(state, runs=tuple(runs), bundles=tuple(bundles))
        validate_model_state(result)
        return result
    if kind is ModelActionKind.BUNDLE_PUBLISH:
        assert action.bundle is not None
        bundle = state.bundles[action.bundle]
        runs = list(state.runs)
        owners = (
            (bundle.parent_run, *bundle.member_runs)
            if bundle.parent_resident
            else bundle.member_runs
        )
        for run_index in owners:
            run = runs[run_index]
            runs[run_index] = replace(
                run,
                producer_waits=run.producer_waits + 1,
            )
        bundles = list(state.bundles)
        bundles[action.bundle] = replace(
            bundle,
            phase=ModelBundlePhase.PUBLISHED,
        )
        result = replace(state, runs=tuple(runs), bundles=tuple(bundles))
        validate_model_state(result)
        return result
    if kind is ModelActionKind.BUNDLE_BIND:
        assert action.bundle is not None
        bundle = state.bundles[action.bundle]
        runs = list(state.runs)
        owners = (
            (bundle.parent_run, *bundle.member_runs)
            if bundle.parent_resident
            else bundle.member_runs
        )
        for run_index in owners:
            run = runs[run_index]
            runs[run_index] = replace(
                run,
                bound_producer_waits=run.bound_producer_waits + 1,
                delivery_ready_waits=run.delivery_ready_waits + 1,
            )
        bundles = list(state.bundles)
        bundles[action.bundle] = replace(
            bundle,
            phase=ModelBundlePhase.TRANSFERRED,
        )
        result = replace(state, runs=tuple(runs), bundles=tuple(bundles))
        validate_model_state(result)
        return result
    if kind is ModelActionKind.BUNDLE_RELEASE:
        assert action.bundle is not None
        bundle = state.bundles[action.bundle]
        runs = list(state.runs)
        owners = (
            (bundle.parent_run, *bundle.member_runs)
            if bundle.parent_resident
            else bundle.member_runs
        )
        for run_index in owners:
            run = runs[run_index]
            runs[run_index] = replace(
                run,
                paused=run.paused - 1,
                producer_waits=run.producer_waits - 1,
                bound_producer_waits=run.bound_producer_waits - 1,
                delivery_ready_waits=run.delivery_ready_waits - 1,
                credits=run.credits - 1,
                terminal=run.terminal + 1,
            )
        bundles = list(state.bundles)
        bundles[action.bundle] = replace(
            bundle,
            phase=ModelBundlePhase.TERMINAL,
            member_checkpointed=(),
        )
        result = replace(state, runs=tuple(runs), bundles=tuple(bundles))
        validate_model_state(result)
        return result
    if kind is ModelActionKind.CONDITION_BATCH:
        runs = list(state.runs)
        for index in action.participants:
            run = runs[index]
            runs[index] = replace(
                run,
                pending=run.pending - 1,
                committed_prefill=run.committed_prefill + 1,
                credits=run.credits + 1,
            )
        result = replace(state, runs=tuple(runs))
        validate_model_state(result)
        if result == state:
            raise ValueError("enabled model action made no progress")
        return result
    run = state.runs[action.run]
    if kind is ModelActionKind.CONDITION:
        updated = replace(
            run,
            pending=run.pending - 1,
            committed_prefill=run.committed_prefill + 1,
            credits=run.credits + 1,
            session_resident=True,
        )
    elif kind is ModelActionKind.CONDITION_PREFILL:
        updated = replace(
            run,
            pending=run.pending - 1,
            active=run.active + 1,
            credits=run.credits + 1,
            session_resident=True,
        )
    elif kind is ModelActionKind.RESIZE:
        target = run.resize_target
        assert target is not None
        updated = replace(
            run,
            width=target.width,
            arena_blocks=target.arena_blocks,
            resize_target=None,
        )
    elif kind is ModelActionKind.PREFILL:
        updated = replace(
            run,
            committed_prefill=run.committed_prefill - 1,
            active=run.active + 1,
        )
    elif kind is ModelActionKind.DECODE_COMPLETE:
        updated = replace(
            run,
            active=run.active - 1,
            credits=run.credits - 1,
            terminal=run.terminal + 1,
        )
    elif kind is ModelActionKind.CHECKPOINT_WAIT:
        updated = replace(
            run,
            active=run.active - 1,
            paused=run.paused + 1,
            producer_waits=run.producer_waits + 1,
        )
    elif kind is ModelActionKind.BIND_PRODUCER_WAIT:
        updated = replace(
            run,
            bound_producer_waits=run.bound_producer_waits + 1,
        )
    elif kind is ModelActionKind.PRODUCER_COMPLETE:
        updated = replace(
            run,
            delivery_ready_waits=run.delivery_ready_waits + 1,
        )
    elif kind in {
        ModelActionKind.PRODUCER_RESUME,
        ModelActionKind.PRODUCER_DISCARD,
    }:
        updated = replace(
            run,
            producer_waits=run.producer_waits - 1,
            bound_producer_waits=run.bound_producer_waits - 1,
            delivery_ready_waits=run.delivery_ready_waits - 1,
            restore_controls=run.restore_controls
            + int(kind is ModelActionKind.PRODUCER_RESUME),
            discard_controls=run.discard_controls
            + int(kind is ModelActionKind.PRODUCER_DISCARD),
        )
    elif kind is ModelActionKind.PREEMPT_CONDITION:
        updated = replace(
            run,
            pending=run.pending - 1,
            committed_prefill=run.committed_prefill + 1,
            paused=run.paused - 1,
            displaced=run.displaced + 1,
            credits=run.credits + 1,
        )
    elif kind is ModelActionKind.PREEMPT_RESTORE:
        updated = replace(
            run,
            active=run.active + 1,
            paused=run.paused - 1,
            restore_controls=run.restore_controls - 1,
        )
    elif kind is ModelActionKind.RESTORE_PAUSED:
        updated = replace(
            run,
            paused=run.paused - 1,
            active=run.active + 1,
            restore_controls=run.restore_controls - 1,
        )
    elif kind is ModelActionKind.RESTORE_DISPLACED:
        updated = replace(
            run,
            displaced=run.displaced - 1,
            active=run.active + 1,
            restore_controls=run.restore_controls - 1,
        )
    elif kind is ModelActionKind.DISCARD_PAUSED:
        updated = replace(
            run,
            paused=run.paused - 1,
            discard_controls=run.discard_controls - 1,
            credits=run.credits - 1,
            terminal=run.terminal + 1,
        )
    elif kind is ModelActionKind.DISCARD_DISPLACED:
        updated = replace(
            run,
            displaced=run.displaced - 1,
            discard_controls=run.discard_controls - 1,
            credits=run.credits - 1,
            terminal=run.terminal + 1,
        )
    elif kind in {
        ModelActionKind.FAIL,
        ModelActionKind.WATCHDOG,
        ModelActionKind.PRODUCER_FAILURE,
    }:
        def terminate(candidate: ModelRunState) -> ModelRunState:
            return replace(
                candidate,
                pending=0,
                committed_prefill=0,
                active=0,
                paused=0,
                displaced=0,
                restore_controls=0,
                discard_controls=0,
                producer_waits=0,
                bound_producer_waits=0,
                delivery_ready_waits=0,
                credits=0,
                failure_pending=False,
                session_resident=False,
                terminal=(
                    candidate.terminal
                    + candidate.pending
                    + candidate.committed_prefill
                    + candidate.active
                    + candidate.paused
                    + candidate.displaced
                ),
            )

        result = replace(
            state,
            runs=tuple(terminate(candidate) for candidate in state.runs),
            bundles=tuple(
                replace(
                    bundle,
                    phase=ModelBundlePhase.TERMINAL,
                    member_checkpointed=(),
                )
                for bundle in state.bundles
            ),
        )
        validate_model_state(result)
        if result == state:
            raise ValueError("enabled model action made no progress")
        return result
    else:
        raise AssertionError(f"unhandled model action: {kind}")
    runs = list(state.runs)
    runs[action.run] = updated
    result = replace(state, runs=tuple(runs))
    validate_model_state(result)
    if result == state:
        raise ValueError("enabled model action made no progress")
    return result


def explore_reachable(
    initial: ModelState,
    *,
    max_states: int = 100_000,
) -> frozenset[ModelState]:
    """Explore all finite reachable states and assert local deadlock freedom."""
    validate_model_state(initial)
    seen = {initial}
    frontier = [initial]
    while frontier:
        state = frontier.pop()
        actions = model_enabled_actions(state)
        if state.unresolved and not actions:
            raise AssertionError(f"unresolved model state has no action: {state}")
        for action in actions:
            target = apply_model_action(state, action)
            if target in seen:
                continue
            seen.add(target)
            if len(seen) > max_states:
                raise AssertionError("bounded model exceeded state budget")
            frontier.append(target)
    return frozenset(seen)
