"""MuScriptor pipeline backend."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch

from resonforge.audio.activity_regions import (
    ActivityRegionAnalysis,
    AudioRegion,
    analyze_activity_buffer,
    analyze_activity_regions,
)
from resonforge.midi.jsonl import GM, convert_events_to_midi
from resonforge.runtime.gpu import muscriptor_dtype
from resonforge.runtime.logging_config import configure_muscriptor_loggers
from resonforge.runtime.process_runner import Cancelled, ProcessRunner
from resonforge.scheduler.device.block_pool import ArenaCost
from resonforge.scheduler.model_types import (
    BatchSubmission,
    ClaimedBatchResult,
    DependencyClaimMember,
    DependencyClaimToken,
    ModelIdentity,
    ModelKey,
    PreparedWorkItem,
    TransitiveDependencyClaim,
)
from resonforge.scheduler.model_workers import (
    ModelWorkerRegistry,
    resident_handle_key,
)
from resonforge.scheduler.preparation_service import acquire_preparation_service
from resonforge.transcribers.base import (
    BackendOptions,
    MemoryArtifact,
    ModelDescriptor,
    ModelLane,
    StemTask,
    TranscriptionRequest,
    TranscriptionResult,
)
from resonforge.transcribers.muscriptor.execution_profile import (
    DEFAULT_GRAPH_BUCKET_SIZE,
    DEFAULT_GRAPH_CACHE_SIZE,
    MuscriptorExecutionProfile,
    resolve_muscriptor_execution_profile,
)
from resonforge.transcribers.muscriptor.quality_plan import MuscriptorQualityPlan
from resonforge.transcribers.muscriptor.region_producer import (
    ProducerHandoff,
    RegionProducerSession,
    RegionStep,
)

from .options import MuscriptorOptions

LOGGER = logging.getLogger(__name__)


_DEFAULT_BOOTSTRAP_MODELS = {
    "small": "medium",
    "medium": "large",
    "large": None,
}
_MODEL_SIZE_ORDER = {"small": 0, "medium": 1, "large": 2}


def _resolve_bootstrap_model(primary_model: str) -> str | None:
    """First-chunk bootstrap model: the next size up, None for large."""
    return _DEFAULT_BOOTSTRAP_MODELS[primary_model]


def _resolve_fresh_reanchor_model(
    recovery_model: str,
    bootstrap_model: str | None,
) -> str:
    """Choose the larger configured independent model through typed tiers."""
    candidates = (recovery_model,) if bootstrap_model is None else (
        recovery_model,
        bootstrap_model,
    )
    return max(candidates, key=_MODEL_SIZE_ORDER.__getitem__)


@dataclass(frozen=True)
class TranscriptionPlan:
    """Immutable work inventory produced before transcription starts."""

    activity: ActivityRegionAnalysis
    total_chunks: int


def plan_transcription(
    task: StemTask,
    *,
    args: MuscriptorOptions,
) -> TranscriptionPlan:
    """Analyze regions before model work so the scheduler sees full inventory."""
    analyzer = analyze_activity_buffer if task.buffer is not None else analyze_activity_regions
    activity = analyzer(
        task.buffer if task.buffer is not None else task.audio,
        **({"source": task.audio} if task.buffer is not None else {}),
        enabled=_require_options(args).muscriptor_silence_split,
        mode="rms",
        threshold_dbfs=ACTIVITY_THRESHOLD_DBFS,
        minimum_silence_seconds=SILENCE_SPLIT_MINIMUM_SECONDS,
        pre_roll_seconds=SILENCE_SPLIT_PRE_ROLL_SECONDS,
        post_roll_seconds=SILENCE_SPLIT_POST_ROLL_SECONDS,
    )
    overlap = REPLAY_SECONDS + MIN_VERIFY_SECONDS
    return TranscriptionPlan(
        activity=activity,
        total_chunks=sum(
            _region_chunk_count(region.duration_seconds, overlap)
            for region in activity.regions
        ),
    )


def worker_model_key(
    model: str,
    *,
    args: MuscriptorOptions,
    device: str,
) -> ModelKey:
    """Resolve the canonical shared model key used by preload and execution."""
    # The last public entry point without this guard, found by the CUDA gate
    # raising `AttributeError: no attribute 'muscriptor_runtime'` instead of
    # naming the mis-routed type.
    args = _require_options(args)
    gpu = int(device.partition(":")[2]) if device.startswith("cuda:") else None
    dtype = muscriptor_dtype(gpu)
    return ModelKey(
        identity=ModelIdentity(
            scope="shared",
            device=device,
            model=model,
            dtype=dtype,
            stem=None,
        ),
        execution_profile=resolve_muscriptor_execution_profile(
            runtime=args.muscriptor_runtime,
            device=device,
            prefill=args.muscriptor_prefill_runtime,
        ),
    )


def worker_device_for_gpu(gpu: int | None) -> str:
    """Resolve the same device identity used by deferred MuScriptor requests."""
    if gpu is not None:
        return f"cuda:{gpu}"
    from muscriptor.accelerator import current_accelerator

    return str(current_accelerator())


def initial_allocation_models(model: str, args: MuscriptorOptions) -> tuple[str, ...]:
    """Resolve every model that may own a production session for this run."""
    args = _require_options(args)
    bootstrap = _resolve_bootstrap_model(model)
    recovery = args.muscriptor_recovery_model
    fresh = _resolve_fresh_reanchor_model(recovery, bootstrap)
    return tuple(
        dict.fromkeys(
            candidate
            for candidate in (model, recovery, bootstrap, fresh)
            if candidate is not None
        )
    )


def _transitive_dependency_claim(
    request: object,
    *,
    primary_key: ModelKey,
    recovery_model_name: str,
    profiles_by_device: dict[str, MuscriptorExecutionProfile],
    configured_batch_size: int,
) -> TransitiveDependencyClaim | None:
    claim = getattr(request, "recovery_group_claim", None)
    if claim is None:
        return None
    members = []
    for member in claim.members:
        target_model = member.target_model
        if target_model is None:
            target_model = (
                recovery_model_name
                if member.model_role == "recovery"
                else primary_key.model
            )
        key = (
            primary_key
            if target_model == primary_key.model
            else ModelKey(
                identity=ModelIdentity(
                    scope="shared",
                    device=primary_key.device,
                    model=target_model,
                    dtype=primary_key.dtype,
                    stem=primary_key.stem,
                ),
                execution_profile=profiles_by_device[primary_key.device],
            )
        )
        members.append(DependencyClaimMember(key=key))
    return TransitiveDependencyClaim(tuple(members))


def _prepare_recovery_continuous_work(
    model_obj: object,
    requests: tuple[object, ...],
) -> tuple[PreparedWorkItem, ...]:
    """Build recovery session rows and retain role-specific result adapters."""
    from resonforge.transcribers.muscriptor.quality.recovery_runtime import (
        complete_recovery_candidate,
        prepare_recovery_candidates,
    )

    prepared = prepare_recovery_candidates(model_obj, requests)
    return tuple(
        PreparedWorkItem(
            item.request,
            lambda result, selected=item: complete_recovery_candidate(
                selected,
                result,
            ),
        )
        for item in prepared
    )


# `MAX_ARENA_WIDTH = 20` and `MAX_SECONDARY_ARENA_WIDTH = 2` stood here, deleted with the condition they defended against.
# Both were bands against "the lane that opens first takes the device" -- real and repeatedly reproduced: a lane opened wide, used a fraction, finished its burst in seconds and held the rest all run
# while the co-resident lane sat on its floor of 1, roughly doubling the wall (R11).
# That condition holds only while a lane opens at *what fits*. It doesn't survive `_resolve_arena_width` asking `_run_demand_width` instead -- a lane now opens at the rows it has work for, so there's nothing for the first lane to take.
# ! re-check if either is ever reintroduced: the transient reserve is a bootstrap constant until the first measurement lands, and transient scales with width, so a width chosen before that measurement is sized against a water line that then rises behind it.
# `--batch-size` still forces a per-model ceiling, now the only way one exists.


def arena_width_ceiling(model: str, args: MuscriptorOptions) -> int:
    """A forced ceiling for this lane, or **0 for none**.

    Forcing `--batch-size` asks for an exact width, so the ceiling becomes that number and the pool refuses rather than narrows if it doesn't fit.
    Nothing forced = no ceiling at all: the width is bounded by the lane's own demand and what the device can give, a fact about this run rather than a number chosen for a host.
    """
    forced = dict(getattr(args, "batch_size_by_model", ()) or ())
    if getattr(args, "batch_size", 0) > 0:
        return int(args.batch_size)
    return max(0, int(forced.get(model, 0)))


def release_idle_kv_supply() -> tuple[int, int, int]:
    """Shrink every KV pool to its lent extent. `(bytes, pools, releasable)`.

    A2e's policy call and the reason the mechanism exists: another process is about to want this card.
    Nothing refused, nothing lost, and a pool grows back into the space on its next lease -- the cost is the remap it pays then, bounded by the granules it gave up.

    Lives here because this module owns the MuScriptor boundary: where the pool is declared is where it's asked to shrink.
    Caller is `pipeline.separation`, which must not know what a KV page is.

    **Three numbers rather than one, because zero has three causes and they
    call for opposite responses.** No pool declared yet means the trigger fires
    too early; a pinned tail means lowest-first is not draining; no device
    support means there is no mechanism at all. Reported separately after a
    two-song run released nothing and the log could not say which it was --
    "it did not happen" and "it was not recorded" reading the same is the
    failure this project has already paid for twice.

    `releasable` is what the tails held *before* the release, so
    `releasable - bytes` is what granule alignment cost.
    """
    try:
        from muscriptor.modules.paged_kv import (
            pool_tail_occupancy,
            release_pool_tails,
        )
    except ImportError:
        return (0, 0, 0)
    occupancy = pool_tail_occupancy()
    releasable = sum(tail * page for _total, tail, page in occupancy.values())
    return (release_pool_tails(), len(occupancy), releasable)


def _measure_arena_cost(
    model_obj: object, kv_capacity: int, lane: str
) -> ArenaCost:
    """Price one logical row of this model, and put this lane's KV floor on record.

    Two things, because this is the one call that happens **per lane, before any
    lane has a width**. A pool sized later reads `mem_get_info` to decide what it
    may take, and a lane that has not declared yet is invisible to that reading:
    the first lane takes the card and the second cannot open at all. Recording
    the floor here costs nothing -- a declaration allocates no memory -- and it
    is what `declared_lane_count` counts, and what `device_ledger` divides by.
    """
    from resonforge.transcribers.muscriptor.runtime.arena_cost import (
        measure_arena_cost,
    )

    _declare_kv_floor(model_obj, kv_capacity, lane)
    return measure_arena_cost(
        model_obj._model,
        kv_capacity=kv_capacity,
        cfg_enabled=bool(getattr(model_obj._model, "cfg_enabled", False)),
    )


def _kv_geometry(model_obj: object, kv_capacity: int) -> tuple[int, int, int, object]:
    """`(blocks one row needs across every layer, heads, head_dim, dtype)`.

    One definition, used by the floor declaration and by the supply that grows
    past it. Two copies of this arithmetic disagreeing is how a pool once came
    out one block short of what `init_state` asked it for.
    """
    from muscriptor.modules.paged_kv import blocks_for_tokens, padded_length_for

    lm = model_obj._model
    layers = lm.transformer.layers
    attention = layers[0].self_attn
    backend = attention.attention_backend
    weight = attention.in_proj_weight
    dtype = getattr(backend, "cache_dtype", None) or weight.dtype
    bounds = getattr(lm, "condition_span_bounds", None) or {}
    length = kv_capacity + sum(max(0, int(bound)) for bound in bounds.values())
    per_row = blocks_for_tokens(padded_length_for(length)) * len(layers)
    return per_row, attention.num_heads, attention.dim_per_head, dtype


def _prefill_row_blocks(model_obj: object) -> int:
    """Blocks one **fresh** row's prefill takes across every layer.

    Prefill draws from the same supply since it takes a contiguous lease instead of allocating privately, so the pool has a consumer it didn't have when `wanted` was written.
    Much smaller than a resident row: prefill writes the condition prefix and one sampled token, a resident row addresses the whole generation.

    Exact for a fresh row, from the same declared ceiling `_kv_geometry` uses (`condition_span_bounds`) and the same `+ 1` sampled token `prefill.prepare._prefill_states` allocates for.

    A RESTORED row is not priced here. Its prefill replays a forced prompt of up to `max_gen_len`, so it approaches a whole resident row
    -- but `restore_preempted` takes one handle, so it's one row at a time and the `+ 1` row of headroom in `wanted` covers it.
    """
    from muscriptor.modules.paged_kv import blocks_for_tokens, padded_length_for

    lm = model_obj._model
    bounds = getattr(lm, "condition_span_bounds", None) or {}
    prepend = sum(max(0, int(bound)) for bound in bounds.values())
    return blocks_for_tokens(padded_length_for(prepend + 1)) * len(
        lm.transformer.layers
    )


def _declare_kv_floor(model_obj: object, kv_capacity: int, lane: str) -> None:
    """Register this lane's KV floor before anybody sizes against the device.

    A number rather than a pool -- `register_lane_floor` records why. Idempotent,
    which matters because `_measure_arena_cost` runs twice: at preload, and again
    from capacity preparation immediately before the width is resolved.
    """
    from muscriptor.modules.paged_kv import page_bytes_for, register_lane_floor

    device = model_obj._model.transformer.layers[0].self_attn.in_proj_weight.device
    per_row, num_heads, head_dim, dtype = _kv_geometry(model_obj, kv_capacity)
    register_lane_floor(
        device,
        lane,
        floor_bytes=per_row
        * page_bytes_for(num_heads=num_heads, head_dim=head_dim, dtype=dtype),
    )


def _declare_kv_pool(
    model_obj: object,
    width: int,
    kv_capacity: int,
    lane: str,
    role: str,
    device: str,
) -> int:
    """Size this lane's KV supply to what the device can spare.

    This rule replaced its opposite and the measurement is why. A2d sized the pool to exactly what `init_states` was about to ask (one range per layer, `width x cfg_width` rows),
    reasoning that sizing from free device memory is the greed that starves the separation subprocess sharing the card.

    Right when a pool was a RESERVATION. Since A3 it's a SUPPLY: an arena takes no page until a row reaches a block boundary and returns every page when the row ends,
    so a large declaration doesn't occupy a large amount -- a small one just leaves the card empty. Measured at the moment it failed:
        KVPagesExhausted   37 pages wanted per layer, 36 available
        device free        4.5 GB
        pool_reserve_mib   1,472 (already measured down from the bootstrap)
    The byte budget said plenty, the page pool said 36, and 4.5 GB sat untouched -- because the recovery lane's width is 2, so its pool was declared for two rows.

    `width` is still taken, as the FLOOR: whatever the scheduler resolved has to fit or the session can't open. Above that the device decides.
    Separation coexistence moves to A2e, which releases pages on demand -- the mechanism A3-3c provides and A2d had to approximate by never asking for much.
    """
    import torch
    from muscriptor.modules.paged_kv import (
        declare_pool,
        declared_lane_count,
        mapped_pool_bytes,
        page_bytes_for,
    )

    from resonforge.scheduler.device import ledger as device_ledger
    from resonforge.scheduler.device.block_pool import device_block_pool, reserve_bytes

    lm = model_obj._model
    weight = lm.transformer.layers[0].self_attn.in_proj_weight
    # `_kv_geometry` resolves the dtype, the heads and the per-row block count
    # exactly as `MultiHeadAttention.init_state` does, and the floor declaration
    # draws on the same function -- so a lane's floor and its supply can never
    # be sized from two different answers.
    per_row, num_heads, head_dim, dtype = _kv_geometry(model_obj, kv_capacity)
    rows = int(width) * (2 if getattr(lm, "cfg_enabled", False) else 1)
    # One row of headroom above what the arena holds. Not a margin: a rebuilt row needs its whole length back while every resident row keeps what it has,
    # so a supply sized to exactly the resident rows can't fund the one operation preemption exists to make possible. Measured before it was there: 37 pages wanted per layer, 36 available.
    #
    # Plus one row-set at PREFILL length, because prefill now draws from this supply instead of allocating privately.
    # An admission cohort is bounded by the session's free slots, so worst case every slot prefills at once -- but a prefill row is the condition prefix plus one sampled token, not the generation the slot will hold,
    # so it's a fraction of a resident row-set, not a second one.
    # Priced as arithmetic (`_prefill_row_blocks`), not a fitted fraction of `per_row`: the ratio moves with `kv_capacity` and the conditioner vocabulary, exactly the coefficient shape R6 forbids.
    #
    # And no more than that.
    wanted = per_row * (rows + 1) + _prefill_row_blocks(model_obj) * rows
    # And no more than the device can hold. The docstring says a large declaration doesn't occupy a large amount -- true of a reservation, not of this:
    # `KVBlockPool._ensure_storage` is one `torch.empty` over the whole declared extent, so the pool occupies every byte it declares as soon as anything leases from it.
    # Over-declaring once put more on the card than it had, and Windows backs an oversubscribed allocation with system memory instead of failing -- a silent 4.9x on every decode step (R11).
    #
    # `empty_cache` first or the reading isn't the truth: the caching allocator holds a GiB or two per loaded model that `mem_get_info` counts as used and torch would hand straight back.
    #
    # How much of the device this lane may take is `device_ledger`'s answer, not this function's.
    # Three reserves come off one reading, each exactly once, and what's left is divided by ROLE rather than declaration order -- which is what the old arithmetic here divided by,
    # and why the idle recovery lane opened at 81 rows on BAIR while the primary lane with 18 jobs queued opened at 28 and then ran out of pages.
    #   * process reserve = `reserve_bytes`, nothing re-derives it: the measured transient + fragmentation the block pool already holds back. A second fragmentation constant reverted the first attempt at this wiring.
    #   * headroom is for consumers outside the process, which no measurement from inside can see.
    #   * prefill is STATED before it allocates and accumulated per device -- the one consumer that takes memory AFTER the pool took its share.
    #
    # The reading has the other lanes' MAPPED KV added back before dividing. Without that, whichever lane sizes second reads a card already reduced by the first lane's share and takes a share of the remainder
    # -- order dependence wearing a fair split's clothes.
    if weight.device.type == "cuda":
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(weight.device)
        # **The scheduler's key, and it is required for that reason.** The
        # registry is keyed by the string, and a tensor reports `cuda:0` where
        # the scheduler says `cuda`, so re-deriving it here opened a **second
        # pool for the same card** -- one holding every measurement, one
        # holding none, and this side priced its reserve against the empty one.
        # A default that fell back to the tensor would let that return in
        # silence.
        pool_budget = device_block_pool(device)
        # **Prefill is no longer a term in this division at all.** It used to
        # be, and getting its width wrong here cost the whole device: priced
        # against `rows`, a wide arena reserved most of the card at production
        # sequence length, left the pool nothing and put every lane at width 1.
        # The term was zeroed because
        # `reserve_bytes` is a process-wide high-water that already contained
        # prefill's peak, and it is gone for good now that prefill's KV is
        # *inside* the page pool (`_prefill_row_blocks`, priced in `wanted`
        # above) rather than a second allocation this division had to guess at.
        # One lane on the card means no second family to hold a share for, and
        # a share held for nobody is a third of the device nothing may use.
        # With more than one, both families are taken to be present: a lane is
        # a *model*, recovery and bootstrap are the same model, so a device
        # carrying two lanes is carrying one of each. Coarse if that ever stops
        # holding, never unsafe -- the shares still sum to the remainder.
        roles_present = (
            frozenset({role})
            if declared_lane_count(weight.device) <= 1
            else device_ledger.ALL_ROLES
        )
        # **This line is where a width is decided**, and `note_width_decision`
        # is here rather than inside `transient_bytes_for` for that reason: the
        # price is read on every state observation and bound by almost none of
        # them. What the probe has to answer is whether the *decision* saw a
        # measurement or the bootstrap -- see `bootstrap_pricing_report`.
        priced_rows = pool_budget.resident_rows + int(width)
        pool_budget.note_width_decision(priced_rows)
        divisible = free_bytes + mapped_pool_bytes(weight.device, exclude=lane)
        page = page_bytes_for(
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

        def blocks_for(transient_bytes: int) -> int:
            budget = device_ledger.divide(
                divisible,
                # Rows this device will hold once this lane opens: what it already holds + the width being declared.
                # A lane re-declaring during a resize is counted twice, which over-reserves rather than under-reserves -- the right direction, since under-reserving on Windows is a silent 4.9x.
                process_reserve_bytes=reserve_bytes(
                    resident_rows=priced_rows,
                    transient_bytes=transient_bytes,
                    fragmentation_bytes=pool_budget.measured_fragmentation_bytes,
                ),
                roles_present=roles_present,
            )
            spare = device_ledger.lane_bytes(budget, role)
            # Never below one row across every layer: a supply that can't fund the lane's floor is the terminal capacity outcome and belongs to the scheduler's `DeviceCannotHoldOneRow` (model_workers), not a silent narrowing here.
            # That guard exists since 2026-08-29: `_kv_row_capacity` raises when a WHOLE supply funds zero rows (deadlock capture C).
            return max(per_row, min(wanted, spare // page))

        wanted = blocks_for(pool_budget.transient_bytes_for(priced_rows))
        # **The counterfactual, and it is why the division above is a closure.**
        # Every width decision prices a row count nothing has decoded at, so it
        # is always the bootstrap -- knowing that says nothing about whether the
        # bootstrap's error *changed* anything. Re-running the same division
        # against what the measurements imply does, and it is pure arithmetic:
        # no device read, no allocation, one extra divide per declaration.
        from_measurements = pool_budget.extrapolated_transient_bytes(priced_rows)
        if from_measurements is not None:
            pool_budget.note_width_outcome(wanted, blocks_for(from_measurements))
    pool = declare_pool(
        num_blocks=wanted,
        num_heads=num_heads,
        head_dim=head_dim,
        device=weight.device,
        dtype=dtype,
        lane=lane,
        # What one row costs across every layer: the floor below which no
        # amount of preemption lets anybody run. See `block_pool.py`.
        row_floor=per_row,
    )
    return pool.total_blocks * pool.page_bytes


def _create_continuous_session(
    model_obj: object,
    requests: tuple[object, ...],
    width: int,
    capacity: int,
) -> object:
    from resonforge.transcribers.muscriptor.runtime.continuous_generation import (
        create_continuous_generation_session,
    )

    # The Graph ceiling is **not** set here any more. It is set inside
    # `_create_cuda_graph_runtime`, from the session's own width, at every
    # moment the runtime is built -- which is the same guarantee this
    # assignment gave, extended to the resize that this one could not see.
    return create_continuous_generation_session(
        model_obj,
        requests,
        width=width,
        capacity=capacity,
    )


@dataclass(frozen=True)
class _ScheduledWork:
    region_index: int
    key: ModelKey
    session: RegionProducerSession
    generation_job: bool
    claim_token: DependencyClaimToken | None = None
    producer_handles: tuple[object, ...] = ()


# A run collecting training data is archived whole, so the cap is a guard
# against an unbounded process rather than a budget: 256 MiB is the debug
# tracer's, and this keeps orders of magnitude more per song.
ACTIVATION_EXPORT_MAX_MIB = 16 * 1024


def activation_export_key(out_dir: Path, stem: str) -> str:
    """Which activation export a row's output belongs to.

    As fine as the unit that owns an artifact, which is a **stem of a run** and
    not a run: one loaded model serves several songs at once, and
    `--parallelism` decodes one song's stems concurrently through it. Keyed per
    run instead, five collectors overwrote each other in the registry and four
    stems exported nothing -- silently, because each still wrote its own
    artifact and one of them was full.

    Named and tested rather than inlined for exactly that reason: it is the
    kind of decision that looks like string formatting and is not.
    """
    return f"{out_dir}::{stem}"


def _register_activation_export(
    model_workers: object,
    keys: tuple[ModelKey, ...],
    export_key: str,
    collector: object | None,
) -> None:
    """Attach or detach one run's export sink on every worker model.

    Through `run`, because the sink has to be set on the model object itself
    and only its own worker thread may touch it. `None` removes the entry, so
    a finished run stops collecting even though the model stays loaded for the
    next one.
    """
    if collector is None and not keys:
        return

    def attach(model_obj: object) -> None:
        lm = model_obj._model
        registry = getattr(lm, "_activation_export", None)
        if registry is None:
            registry = {}
            lm._activation_export = registry
        if collector is None:
            registry.pop(export_key, None)
        else:
            registry[export_key] = collector

    for key in keys:
        model_workers.run(key, attach)


def output_path(task: StemTask, out_dir: Path) -> Path:
    return out_dir / task.name / "raw.midi"


def _event_to_jsonl(event: object) -> dict[str, object] | None:
    """Convert one event into the CLI-style JSONL shape."""
    from muscriptor.events import NoteEndEvent, NoteStartEvent

    from resonforge.transcribers.muscriptor.quality.generation_anomaly import (
        GenerationAnomalyEvent,
    )
    from resonforge.transcribers.muscriptor.quality.overlap import (
        OverlapDiagnosticsEvent,
    )
    from resonforge.transcribers.muscriptor.quality.recovery import (
        RecoveryDiagnosticsEvent,
    )
    from resonforge.transcribers.muscriptor.quality.recovery_runtime import (
        FirstChunkBootstrapEvent,
    )

    if isinstance(event, NoteStartEvent):
        return {"type": "start", **dataclasses.asdict(event)}
    if isinstance(event, NoteEndEvent):
        return {
            "type": "end",
            "end_time": event.end_time,
            "start_event_index": event.start_event_index,
        }
    if isinstance(event, OverlapDiagnosticsEvent):
        return {"type": "overlap_diagnostics", **dataclasses.asdict(event)}
    if isinstance(event, GenerationAnomalyEvent):
        return {"type": "generation_anomaly", **dataclasses.asdict(event)}
    if isinstance(event, RecoveryDiagnosticsEvent):
        return {"type": "recovery_diagnostics", **dataclasses.asdict(event)}
    if isinstance(event, FirstChunkBootstrapEvent):
        return {"type": "first_chunk_bootstrap", **dataclasses.asdict(event)}
    return None


def load_worker_model(key: ModelKey) -> object:
    """Load the model owned by one explicit scheduler worker."""
    from muscriptor.transcription_model import TranscriptionModel

    loaded = TranscriptionModel.load_model(
        weights_path=key.model,
        device=key.device,
        dtype=key.dtype,
    )
    # No ceiling here: the session sets one from the width it actually opens
    # at, which is the only number a captured Graph may be compared against.
    _configure_loaded_model(loaded, key.execution_profile)
    return loaded


PACKED_PREFILL_VARIABLE = "RESONFORGE_PACKED_PREFILL"


def _configure_loaded_model(
    loaded: object,
    profile: MuscriptorExecutionProfile,
    *,
    graph_max_batch: int | None = None,
) -> None:
    """Apply one already-resolved profile to a loaded MuScriptor model."""
    from resonforge.transcribers.muscriptor.runtime.prefill.prepare import (
        configure_prefill_graphs,
    )

    model = loaded._model
    selection = model.configure_attention_backend(profile.resolved_backend)
    if selection.selected != profile.resolved_backend:
        raise RuntimeError(
            "low-level backend selection changed the resolved profile: "
            f"expected {profile.resolved_backend}, got {selection.selected}"
        )
    # The graph runtime rejects any batch wider than the size it was
    # configured for. The ceiling is set again by `_create_continuous_session`
    # from the width the arena actually opens at; leaving it at the backend
    # default here would cap decode at eight rows for anything that never
    # reaches that call.
    model.configure_continuous_graph_decode(
        profile.cuda_graphs,
        bucket_size=DEFAULT_GRAPH_BUCKET_SIZE,
        max_graphs=DEFAULT_GRAPH_CACHE_SIZE,
        **({} if graph_max_batch is None else {"max_batch_size": graph_max_batch}),
    )
    # The prefill line, on the same object for the same reason: `model` is what
    # `prepare` is handed. No ceiling and no bucket count to configure -- the
    # cache is created on demand and its bucket is a module constant, so this
    # is one boolean and the whole of the axis.
    configure_prefill_graphs(model, profile.prefill_graphs)
    # `configure_packed_prefill` had zero callers, so the ragged mixed-length
    # prefill path had never run in production. This is the caller, and it
    # stays **off by default on a measured result**: it fires every time it is
    # reachable, its whole ceiling is a couple of percent because most prefills
    # are width 1 and have nothing to pack, and it is **measurably slower on
    # exactly the batches it acts on** (permutation test on the median) while
    # identical elsewhere. The inference is that the ragged kernel costs more
    # per token than N dense passes at small N; what is *measured* is that the
    # cost appears only where ragged fires.
    #
    # Do not default it on without a device where that reverses -- and turning
    # it on is an O1 event either way.
    if os.environ.get(PACKED_PREFILL_VARIABLE) == "1":
        model.configure_packed_prefill(True)
    LOGGER.info(
        "attention backend selected=%s packed_prefill=%s prefill_runtime=%s",
        selection.selected,
        model.packed_prefill_enabled,
        profile.prefill_runtime,
    )


def _load_model(
    model: str,
    *,
    profile: MuscriptorExecutionProfile,
    device: str,
    dtype: str | None,
) -> object:
    """Direct loader requiring an already-resolved execution profile."""
    from muscriptor.transcription_model import TranscriptionModel

    loaded = TranscriptionModel.load_model(
        weights_path=model,
        device=device,
        dtype=dtype,
    )
    _configure_loaded_model(loaded, profile)
    return loaded


def _resolve_instruments(spec: str | None) -> list[str] | None:
    """Translate the interface's General MIDI vocabulary into model groups.

    The service names GM programs and nothing else (T9/D9-1); every mapping to
    MuScriptor's 35 groups happens here. GM has no program for percussion, so
    the interface carries the channel-10 sentinel 128 and this is where it
    becomes the model's `drums` group — the backend's own resolver rejects 128,
    which is correct: it is not a program.
    """
    if not spec or not spec.strip():
        return None
    names = [
        DRUM_GROUP if value.strip() == str(DRUM_PROGRAM) else value.strip()
        for value in spec.split(",")
        if value.strip()
    ]
    if not names:
        return None
    from muscriptor.tokenizer.mt3 import resolve_instrument_names

    try:
        return resolve_instrument_names(names)
    except ValueError as error:
        raise RuntimeError(f"Invalid Muscriptor instrument list: {error}") from error


def _region_audio(
    task: StemTask,
    region: AudioRegion,
) -> Path | tuple[torch.Tensor, int]:
    if task.buffer is not None:
        if task.buffer.sample_rate != region.sample_rate:
            raise RuntimeError("activity region sample rate changed during transcription")
        audio = task.buffer.samples[region.start_frame : region.end_frame]
        return torch.from_numpy(audio.T.copy()), task.buffer.sample_rate
    if region.start_frame == 0:
        with sf.SoundFile(task.audio) as source:
            if region.end_frame == len(source):
                return task.audio
    audio, sample_rate = sf.read(
        task.audio,
        start=region.start_frame,
        stop=region.end_frame,
        dtype="float32",
        always_2d=True,
    )
    if sample_rate != region.sample_rate:
        raise RuntimeError("activity region sample rate changed during transcription")
    return torch.from_numpy(audio.T.copy()), sample_rate


# Chunk geometry, frozen at the measured defaults. The two bounds that used
# to be enforced in parse_args hold by construction now:
#   REPLAY + max(MIN_VERIFY, 0.8)                    < 5   model condition window
#   RECOVERY_SHIFT + REPLAY + MIN_VERIFY             <= 5
# Silence floor for region detection. -80 rather than activity_regions'
# own -50 default, because that is the value production has always passed:
# it arrived via the hard-zero gate's threshold, which no longer exists.
ACTIVITY_THRESHOLD_DBFS = -80.0
REPLAY_SECONDS = 0.6
MIN_VERIFY_SECONDS = 0.4
RECOVERY_SHIFT_SECONDS = 1.0
# Recovery candidates are seeded so a rerun reproduces them; the value is
# arbitrary, only its stability matters.
RECOVERY_SEED = 0
# In-memory ceiling for one stem's model trace.
TRACE_MAX_MIB = 256
# The one group General MIDI cannot name, because percussion is a channel.
DRUM_GROUP = "drums"
DRUM_PROGRAM = 128


def _require_options(options: object) -> MuscriptorOptions:
    """Fail loudly rather than silently defaulting a mis-routed request.

    Annotating `args: MuscriptorOptions` was not enough: until 3.4 the service
    config carried the same field names, so passing the wrong type worked by
    accident and only broke once those fields moved into `backend_options`.
    An annotation the caller can ignore is a comment.
    """
    if not isinstance(options, MuscriptorOptions):
        raise TypeError(
            f"muscriptor requires MuscriptorOptions, got {type(options).__name__}"
        )
    return options
# Reproduces the activity_regions defaults, so region boundaries are
# unchanged from every run recorded in the baseline.
SILENCE_SPLIT_MINIMUM_SECONDS = 1.5
SILENCE_SPLIT_PRE_ROLL_SECONDS = 0.75
SILENCE_SPLIT_POST_ROLL_SECONDS = 0.25


def _region_chunk_count(duration: float, overlap_seconds: float) -> int:
    if duration <= 5.0:
        return 1
    return 1 + math.ceil((duration - 5.0) / (5.0 - overlap_seconds))


def _rebase_payloads(
    region_payloads: list[tuple[AudioRegion, list[dict[str, object]]]],
) -> list[dict[str, object]]:
    """Rebase local region events and make note indexes globally unique."""
    combined: list[dict[str, object]] = []
    next_index = 0
    for region, payloads in region_payloads:
        index_map: dict[int, int] = {}
        offset = region.source_offset_seconds
        for payload in payloads:
            rebased = dict(payload)
            event_type = rebased.get("type")
            if event_type == "start":
                local_index = int(rebased["index"])
                index_map[local_index] = next_index
                rebased["index"] = next_index
                rebased["start_time"] = float(rebased["start_time"]) + offset
                next_index += 1
            elif event_type == "end":
                local_index = int(rebased["start_event_index"])
                rebased["start_event_index"] = index_map[local_index]
                rebased["end_time"] = float(rebased["end_time"]) + offset
            else:
                rebased = _rebase_diagnostic_times(rebased, offset)
            combined.append(rebased)
    return combined


def _rebase_diagnostic_times(value: object, offset: float) -> object:
    if isinstance(value, dict):
        return {
            key: (
                float(item) + offset
                if key
                in {
                    "seek_time",
                    "condition_seek_time",
                    "condition_origin",
                    "replay_start_time",
                    "verification_start_time",
                    "verification_end_time",
                    "output_start_time",
                    "output_end_time",
                    "safe_frontier_time",
                    "window_start",
                    "verification_start",
                    "verification_end",
                    "origin",
                    "absolute_time",
                }
                and item is not None
                else _rebase_diagnostic_times(item, offset)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rebase_diagnostic_times(item, offset) for item in value]
    if isinstance(value, tuple):
        return tuple(_rebase_diagnostic_times(item, offset) for item in value)
    return value


def _transcribe_scheduled(
    task: StemTask,
    *,
    model: str,
    args: MuscriptorOptions,
    out_dir: Path,
    log_dir: Path,
    runner: ProcessRunner,
    environment_overrides: dict[str, str] | None = None,
    device: str | None = None,
    model_workers: ModelWorkerRegistry[object],
    summary_sink: list[str] | None = None,
    report_sink: dict[str, object] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    plan: TranscriptionPlan | None = None,
) -> TranscriptionResult:
    """Transcribe one stem and return events, MIDI, and optional artifacts."""
    if plan is None:
        plan = plan_transcription(task, args=args)
    activity = plan.activity
    regions = activity.regions
    total_chunks = plan.total_chunks
    logs = configure_muscriptor_loggers(
        log_dir,
        task.name,
    )
    resolved_device = device
    if resolved_device is None:
        from muscriptor.accelerator import current_accelerator

        resolved_device = str(current_accelerator())
    # Widths and profiles come from the scheduler's solve of the residency the
    # pipeline declared; this backend states what is resident, never how much
    # of the device it costs.
    # A pipeline session owns one GPU. Parallel songs share its device lane;
    # regions never fan one song across devices and duplicate model/KV state.
    region_devices = (resolved_device,)

    primary_keys = {
        worker_device: worker_model_key(model, args=args, device=worker_device)
        for worker_device in region_devices
    }
    dtypes_by_device = {
        worker_device: key.dtype for worker_device, key in primary_keys.items()
    }
    profiles_by_device = {
        worker_device: key.execution_profile
        for worker_device, key in primary_keys.items()
    }
    primary_profile = next(iter(profiles_by_device.values()))
    # Restored in 3.4 as `--debug-capture` members. The collector's cap is a
    # constant rather than a flag: a debug capture that needs tuning from the
    # command line is a capture nobody reads twice.
    trace_level = (
        "final-hidden"
        if "trace-hidden" in args.debug_capture
        else "logits"
        if "trace" in args.debug_capture
        else "off"
    )
    export_key = activation_export_key(out_dir, task.name)
    if args.export_activations:
        from muscriptor.model_trace import ModelTraceCollector
    activation_export = (
        ModelTraceCollector("final-hidden", max_mib=ACTIVATION_EXPORT_MAX_MIB)
        if args.export_activations
        else None
    )
    if activation_export is not None:
        # Registered before any work is submitted: a row decoded before the
        # sink exists is a row silently missing from the export.
        _register_activation_export(
            model_workers, tuple(primary_keys.values()), export_key, activation_export
        )
    trace_collector = None
    if trace_level != "off":
        from muscriptor.model_trace import ModelTraceCollector

        trace_collector = ModelTraceCollector(trace_level, max_mib=TRACE_MAX_MIB)
    LOGGER.info(
        "started stem=%s backend=muscriptor model=%s runtime=%s backend=%s",
        task.name,
        model,
        primary_profile.runtime,
        primary_profile.resolved_backend,
        extra={"console": False},
    )

    instruments = _resolve_instruments(task.instruments)
    # Every deferred generation role, including recovery candidates, now
    # publishes model-independent requests before returning to the producer.
    # Both runtime lines own resident KV slots and hot-replace released rows.
    # CUDA Graphs only changes how one decode quantum is executed.
    prelude_forcing = True
    recovery_model_name = args.muscriptor_recovery_model
    bootstrap_model_name = _resolve_bootstrap_model(model)
    fresh_reanchor_model_name = _resolve_fresh_reanchor_model(
        recovery_model_name,
        bootstrap_model_name,
    )
    from resonforge.transcribers.muscriptor.quality.chunk_quality import (
        DEFAULT_ADAPTIVE_CHUNK_QUALITY_CONFIG,
        ChunkQualityHistory,
    )
    from resonforge.transcribers.muscriptor.quality.generation_guard import (
        DEFAULT_HARD_PITCH_ENVELOPE,
    )

    chunk_quality_history = ChunkQualityHistory(
        DEFAULT_ADAPTIVE_CHUNK_QUALITY_CONFIG.history_size
    )
    completed_chunks = 0
    if progress_callback is not None:
        progress_callback(task.name, 0, total_chunks)

    def create_region_session(
        region: AudioRegion,
        worker_device: str,
        worker_dtype: str,
        region_order: int,
    ) -> RegionProducerSession:
        plan = MuscriptorQualityPlan(
            anomaly_detection=args.muscriptor_anomaly_detection,
            overlap_detection=args.muscriptor_overlap_detection,
            recovery=args.muscriptor_recovery,
            recovery_shift_seconds=RECOVERY_SHIFT_SECONDS,
            recovery_seed=RECOVERY_SEED,
            replay_seconds=REPLAY_SECONDS,
            min_verify_seconds=MIN_VERIFY_SECONDS,
            checkpoint_generation=True,
            fresh_reanchor=args.muscriptor_fresh_reanchor,
            first_chunk_model=bootstrap_model_name,
            fresh_reanchor_model=fresh_reanchor_model_name,
            chunk_quality_history=chunk_quality_history,
            hard_pitch_envelope=(
                None
                if task.name in {"other", "drums"}
                else DEFAULT_HARD_PITCH_ENVELOPE
            ),
            trace_collector=trace_collector,
            trace_context_prefix=(export_key, task.name, region_order),
            overlap_probe="overlap-probe" in args.debug_capture,
        )

        def create_events(model_obj: object) -> Iterator[object]:
            return iter(
                model_obj.transcribe(
                    _region_audio(task, region),
                    use_sampling=False,
                    instruments=instruments,
                    batch_size=1,
                    no_eos_is_ok=True,
                    prelude_forcing=prelude_forcing,
                    plan=plan,
                    stdout_logger=logs.stdout_logger,
                    stderr_logger=logs.stderr_logger,
                )
            )

        return RegionProducerSession(
            create_events=create_events,
            event_to_payload=_event_to_jsonl,
            cancelled=lambda: runner.stop_event.is_set() is True,
            ready_prefix=f"{task.name}:{region_order:06d}",
            trace_collector=trace_collector,
        )

    def report_completed(delta: int) -> None:
        nonlocal completed_chunks
        if delta <= 0:
            return
        completed_chunks += delta
        if progress_callback is not None:
            progress_callback(task.name, completed_chunks, total_chunks)

    preparation = acquire_preparation_service(
        session_id=getattr(model_workers, "session_id", f"direct-{id(model_workers):x}")
    )

    try:
        region_payloads: list[tuple[AudioRegion, list[dict[str, object]]]] = []
        if model_workers is not None:
            scheduled: dict[Future[object], _ScheduledWork] = {}
            payloads_by_region: list[list[dict[str, object]]] = [[] for _ in regions]

            def submit_producer(
                key: ModelKey,
                session: RegionProducerSession,
                *,
                transfer_handles: tuple[object, ...] = (),
                new_handles: tuple[object, ...] = (),
                transfer_claim: DependencyClaimToken | None = None,
                new_claim: DependencyClaimToken | None = None,
            ) -> Future[object]:
                """Advance deferred event production off the model lane."""
                if session.producer_model is None or not session.producer_handoff_ready:
                    return model_workers.submit(
                        key,
                        lambda model_obj, selected=session: selected.handoff(model_obj),
                    )
                queued_at = time.perf_counter()

                def advance() -> RegionStep:
                    started_at = time.perf_counter()
                    model_workers.record_waterfall_span(
                        "producer_queue_wait",
                        queued_at,
                        started_at,
                        resource="cpu",
                        lane=f"stem {task.name}",
                        stem=task.name,
                        target_device=key.device,
                        preparation_lane="critical",
                    )
                    initial_prepare = session.events is None
                    generation_result = session.generation_result
                    results = (
                        generation_result
                        if isinstance(generation_result, tuple)
                        else (generation_result,)
                    )
                    action = (
                        "cpu_region_prepare"
                        if initial_prepare
                        else "cpu_verify"
                        if any(
                            result is not None
                            and "RecoveryCandidate" in type(result).__name__
                            for result in results
                        )
                        else "cpu_producer"
                    )
                    try:
                        return session.advance()
                    finally:
                        model_workers.record_waterfall_span(
                            action,
                            started_at,
                            time.perf_counter(),
                            resource="cpu",
                            lane=f"stem {task.name}",
                            stem=task.name,
                            target_device=key.device,
                        )

                producer_future = preparation.submit("critical", advance)
                deliveries: list[Future[object]] = []
                try:
                    if transfer_handles or transfer_claim is not None:
                        deliveries.append(
                            model_workers.transfer_producer_decision_waits(
                                key,
                                transfer_handles,
                                producer_future,
                                claim_tokens=(
                                    ()
                                    if transfer_claim is None
                                    else (transfer_claim,)
                                ),
                            )
                        )
                    if new_handles or new_claim is not None:
                        deliveries.append(model_workers.bind_producer_decision_waits(
                            key,
                            new_handles,
                            producer_future,
                            claim_tokens=(
                                () if new_claim is None else (new_claim,)
                            ),
                        ))
                except BaseException:
                    producer_future.cancel()
                    raise
                return deliveries[0] if deliveries else producer_future

            for index, region in enumerate(regions):
                worker_device = region_devices[index % len(region_devices)]
                worker_dtype = dtypes_by_device[worker_device]
                key = primary_keys[worker_device]
                session = create_region_session(
                    region,
                    worker_device,
                    worker_dtype,
                    index,
                )
                future = model_workers.submit(
                    key,
                    lambda model_obj, selected=session: selected.handoff(model_obj),
                )
                scheduled[future] = _ScheduledWork(index, key, session, False)
            while scheduled:
                if runner.stop_event.is_set():
                    model_workers.cancel_pending()
                    raise RuntimeError("MuScriptor transcription cancelled")
                done, _pending = model_workers.wait_for_first(scheduled)
                for future in done:
                    scheduled_work = scheduled.pop(future)
                    index = scheduled_work.region_index
                    key = scheduled_work.key
                    session = scheduled_work.session
                    try:
                        result = future.result()
                    except CancelledError as error:
                        raise Cancelled() from error
                    if scheduled_work.generation_job:
                        new_claim = None
                        if isinstance(result, ClaimedBatchResult):
                            new_claim = result.claim_token
                            result = result.value
                        results = result if isinstance(result, tuple) else (result,)
                        new_handles = tuple(
                            handle
                            for item in results
                            if item is not None
                            and (handle := getattr(item, "resident_handle", None))
                            is not None
                        )
                        session.generation_result = result
                        next_future = submit_producer(
                            key,
                            session,
                            transfer_handles=scheduled_work.producer_handles,
                            transfer_claim=scheduled_work.claim_token,
                            new_handles=new_handles,
                            new_claim=new_claim,
                        )
                        scheduled[next_future] = _ScheduledWork(
                            index,
                            key,
                            session,
                            False,
                            new_claim or scheduled_work.claim_token,
                            (*scheduled_work.producer_handles, *new_handles),
                        )
                        continue
                    if isinstance(result, ProducerHandoff):
                        next_future = submit_producer(key, session)
                        scheduled[next_future] = _ScheduledWork(
                            index, key, session, False
                        )
                        continue
                    if not isinstance(result, RegionStep):
                        raise TypeError("region worker returned an invalid step")
                    step = result
                    claim_token = scheduled_work.claim_token
                    producer_handles = scheduled_work.producer_handles
                    payloads_by_region[index].extend(step.payloads)
                    report_completed(step.completed_delta)
                    if step.generation_request is not None:
                        from resonforge.transcribers.muscriptor.quality.generation_batch import (
                            GenerationControlRequest,
                            RecoveryCandidateGroupRequest,
                            RecoveryCandidateSpec,
                        )
                        from resonforge.transcribers.muscriptor.runtime.continuous_generation import (
                            condition_batch_key,
                        )

                        submitted_requests = (
                            step.generation_request.requests
                            if isinstance(
                                step.generation_request,
                                RecoveryCandidateGroupRequest,
                            )
                            else (step.generation_request,)
                        )
                        submissions = []
                        for request in submitted_requests:
                            is_recovery = isinstance(request, RecoveryCandidateSpec)
                            is_control = isinstance(request, GenerationControlRequest)
                            job_type = (
                                "control"
                                if is_control
                                else "recovery"
                                if is_recovery
                                else "primary"
                            )
                            unified_secondary = (
                                is_recovery and request.model_role == "recovery"
                            )
                            generation_key = (
                                ModelKey(
                                    identity=ModelIdentity(
                                        scope="shared",
                                        device=key.device,
                                        model=(
                                            request.target_model or recovery_model_name
                                        ),
                                        dtype=key.dtype,
                                        stem=key.stem,
                                    ),
                                    execution_profile=(
                                        profiles_by_device[key.device]
                                        if unified_secondary
                                        else key.execution_profile
                                    ),
                                )
                                if unified_secondary
                                else key
                            )
                            continuous_factory = (
                                _create_continuous_session
                                if not is_control
                                else None
                            )
                            prepare = None
                            if continuous_factory is not None and is_recovery:
                                prepare = _prepare_recovery_continuous_work
                            submissions.append(
                                BatchSubmission(
                                    key=generation_key,
                                    compatibility_key=(
                                        (
                                            "resident",
                                            request.resident_handle.session_id,
                                        )
                                        if is_control
                                        else request.session_compatibility_key
                                    ),
                                    item=request,
                                    condition_compatibility_key=(
                                        None
                                        if is_control
                                        else condition_batch_key(request)
                                    ),
                                    max_batch_size=arena_width_ceiling(
                                        generation_key.model,
                                        args,
                                    ),
                                    arena_cost=_measure_arena_cost,
                                    kv_pool=_declare_kv_pool,
                                    minimum_width=1,
                                    # No band. A secondary tier is bounded by
                                    # its own demand and the device, exactly
                                    # like the primary one.
                                    maximum_width=None,
                                    priority=10,
                                    job_type=job_type,
                                    batch_dimension=(
                                        0 if is_control else len(request.prompt_ids)
                                    ),
                                    continuous_factory=continuous_factory,
                                    prepare=prepare,
                                    continuous_replacement=True,
                                    ready_order=step.ready_order,
                                    dependency_claim=_transitive_dependency_claim(
                                        request,
                                        primary_key=generation_key,
                                        recovery_model_name=recovery_model_name,
                                        profiles_by_device=profiles_by_device,
                                        configured_batch_size=args.batch_size,
                                    ),
                                )
                            )
                        if isinstance(
                            step.generation_request,
                            RecoveryCandidateGroupRequest,
                        ):
                            if claim_token is None:
                                raise RuntimeError(
                                    "recovery group lacks its dependency claim token"
                                )
                            parent_handle = (
                                step.generation_request.parent_resident_handle
                            )
                            generation_future = model_workers.submit_batch_group(
                                key,
                                claim_token,
                                parent_handle,
                                tuple(submissions),
                            )
                            next_claim_token = None
                            next_handles = producer_handles
                        else:
                            if claim_token is not None and not is_control:
                                model_workers.release_dependency_claim(
                                    key,
                                    claim_token,
                                )
                            next_claim_token = claim_token if is_control else None
                            submission = submissions[0]
                            consumed_handle = getattr(
                                submission.item, "resident_handle", None
                            )
                            next_handles = tuple(
                                handle
                                for handle in producer_handles
                                if handle is not consumed_handle
                            )
                            generation_future = model_workers.submit_prepared(
                                submission,
                            )
                        scheduled[generation_future] = _ScheduledWork(
                            index,
                            key,
                            session,
                            True,
                            next_claim_token,
                            next_handles,
                        )
                    elif not step.done:
                        next_future = submit_producer(
                            key,
                            session,
                            transfer_handles=producer_handles,
                            transfer_claim=claim_token,
                        )
                        scheduled[next_future] = _ScheduledWork(
                            index,
                            key,
                            session,
                            False,
                            claim_token,
                            producer_handles,
                        )
                    else:
                        if claim_token is not None:
                            model_workers.release_dependency_claim(key, claim_token)
                        if producer_handles:
                            # Name the rows. The filter above drops a consumed
                            # handle by object identity while the scheduler
                            # identifies rows by resident_handle_key, so a
                            # handle rebuilt across a preemption or restore survives
                            # the filter and lands here. Printing both keys
                            # separates that from a genuinely unconsumed row.
                            raise RuntimeError(
                                "region completed with unresolved producer "
                                "handles: "
                                f"{[resident_handle_key(h) for h in producer_handles]}"
                            )
            region_payloads = list(zip(regions, payloads_by_region, strict=True))
        payloads = tuple(_rebase_payloads(region_payloads))
    except Cancelled:
        raise
    except Exception as error:
        logs.stderr_logger.exception("MuScriptor transcription failed")
        raise RuntimeError(
            f"MuScriptor transcription failed for {task.name}; see {logs.stderr_path}"
        ) from error
    finally:
        preparation.close(cancel_pending=runner.stop_event.is_set())
        logs.close()

    conversion = convert_events_to_midi(payloads, cleanup=False, allow_empty=True)
    artifacts = []
    if args.publish_files:
        artifacts.append(
            MemoryArtifact(
                name="events.json",
                kind="muscriptor_events",
                media_type="application/json",
                data=(json.dumps(payloads, ensure_ascii=False) + "\n").encode(
                    "utf-8"
                ),
                metadata={"events": len(payloads)},
            )
        )
    if report_sink is not None:
        report_sink.update(
            {
                "events_json": "events.json",
                "raw_midi": "raw.midi",
                "activity_regions": activity.to_dict(),
                "region_devices": list(region_devices),
                "execution_profiles": {
                    device_name: profile.to_dict()
                    for device_name, profile in profiles_by_device.items()
                },
            }
        )
    if activation_export is not None:
        _register_activation_export(
            model_workers, tuple(primary_keys.values()), export_key, None
        )
        export_data, export_manifest = activation_export.finalize_bytes()
        artifacts.append(
            MemoryArtifact(
                name="activations.npz",
                kind="muscriptor_activation_export",
                media_type="application/x-npz",
                data=export_data,
                metadata=export_manifest,
            )
        )
        if report_sink is not None:
            report_sink.setdefault("artifacts", []).append(
                {
                    "kind": "muscriptor_activation_export",
                    "member": "activations.npz",
                    **export_manifest,
                }
            )
    if trace_collector is not None:
        trace_data, artifact = trace_collector.finalize_bytes()
        artifacts.append(
            MemoryArtifact(
                name="model-trace.npz",
                kind="muscriptor_model_trace",
                media_type="application/x-npz",
                data=trace_data,
                metadata=artifact,
            )
        )
        if report_sink is not None:
            report_sink.setdefault("artifacts", []).append(
                {"kind": "muscriptor_model_trace", "member": "model-trace.npz", **artifact}
            )
    if summary_sink is not None:
        summary_sink.append("raw MIDI ready; cleanup deferred")
    return TranscriptionResult(
        stem=task.name,
        events=payloads,
        raw_midi=conversion.document,
        artifacts=tuple(artifacts),
        metrics={
            "activity_regions": activity.to_dict(),
            "region_devices": list(region_devices),
            "execution_profiles": {
                device_name: profile.to_dict()
                for device_name, profile in profiles_by_device.items()
            },
            "conversion": conversion.report,
        },
        summary="raw MIDI ready in memory",
    )


def transcribe(
    task: StemTask,
    *,
    model: str,
    args: MuscriptorOptions,
    out_dir: Path,
    log_dir: Path,
    runner: ProcessRunner,
    environment_overrides: dict[str, str] | None = None,
    device: str | None = None,
    model_workers: ModelWorkerRegistry[object] | None = None,
    summary_sink: list[str] | None = None,
    report_sink: dict[str, object] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    plan: TranscriptionPlan | None = None,
) -> TranscriptionResult:
    """Transcribe through the unified scheduler, owning one when needed."""
    kwargs = dict(
        model=model,
        args=args,
        out_dir=out_dir,
        log_dir=log_dir,
        runner=runner,
        environment_overrides=environment_overrides,
        device=device,
        summary_sink=summary_sink,
        report_sink=report_sink,
        progress_callback=progress_callback,
        plan=plan,
    )
    if model_workers is not None:
        return _transcribe_scheduled(task, model_workers=model_workers, **kwargs)
    with ModelWorkerRegistry(
        load_worker_model,
        stop_event=runner.stop_event,
    ) as owned_workers:
        return _transcribe_scheduled(
            task,
            model_workers=owned_workers,
            **kwargs,
        )


class MuScriptorTranscriber:
    name = "muscriptor"
    models = frozenset({"small", "medium", "large"})
    supported_stems = None
    supports_default = True
    self_scheduling = True
    # Declared, so a request naming something this model cannot emit is
    # answerable before the run rather than silently narrowed inside it. The
    # model's 35 groups reach these General MIDI programs and no others.
    supported_programs = frozenset(GM.values()) | {DRUM_PROGRAM}

    def options_from(self, source: object) -> MuscriptorOptions:
        return MuscriptorOptions.from_source(source)

    def describe(
        self,
        model: str | None,
        options: BackendOptions,
        device: str,
    ) -> ModelDescriptor | None:
        """Recovery and bootstrap tiers stay resident beside the primary."""
        if model is None:
            return None
        resolved = _require_options(options)
        return ModelDescriptor(
            model=model,
            lanes=tuple(
                ModelLane(
                    model=candidate,
                    key=worker_model_key(candidate, args=resolved, device=device),
                    arena_cost=_measure_arena_cost,
                    # Every tier needs one resident row before it can serve a
                    # request, and the secondary tiers need theirs held back
                    # while the primary sizes itself against the device.
                    minimum_width=1,
                    # A forced `--batch-size` for this model, or none. The band
                    # that used to separate primary from secondary is gone with
                    # the greedy opening width it defended against.
                    maximum_width=arena_width_ceiling(candidate, resolved) or None,
                )
                for candidate in initial_allocation_models(model, resolved)
            ),
        )

    def plan(self, task: StemTask, options: BackendOptions) -> TranscriptionPlan:
        return plan_transcription(task, args=_require_options(options))

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        if request.model is None:
            raise RuntimeError("MuScriptor transcriber requires a model size")
        return transcribe(
            request.task,
            model=request.model,
            args=_require_options(request.options),
            out_dir=request.out_dir,
            log_dir=request.log_dir,
            runner=request.runner,
            environment_overrides=request.environment_overrides,
            device=request.device,
            model_workers=request.model_workers,
            summary_sink=request.summary_sink,
            report_sink=request.report_sink,
            progress_callback=request.progress_callback,
            plan=(
                request.plan if isinstance(request.plan, TranscriptionPlan) else None
            ),
        )

    def output_path(self, task: StemTask, out_dir: Path, model: str | None) -> Path:
        return output_path(task, out_dir)


BACKEND = MuScriptorTranscriber()
