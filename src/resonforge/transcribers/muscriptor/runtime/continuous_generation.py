"""Fixed-slot continuous greedy generation with private row prefill."""

from __future__ import annotations

import dataclasses
import logging
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import count

import torch
from muscriptor.modules.cuda_contiguous_attention import check_block_table_fault
from muscriptor.modules.paged_kv import (
    KVPagesExhausted,
    blocks_for_tokens,
)
from muscriptor.modules.streaming import (
    bind_continuous_decode_metadata,
    block_tables,
    grow_state_rows,
    increment_state_rows,
    init_states,
    release_state_blocks,
    reset_state_rows,
    select_state_rows,
)

from resonforge.transcribers.muscriptor.quality.generation_batch import (
    GenerationControlResult,
    GenerationRequest,
    GenerationResult,
)
from resonforge.transcribers.muscriptor.quality.guard_protocol import (
    GuardFinding,
)
from resonforge.transcribers.muscriptor.runtime.decode.graphs import (
    ContinuousDecodeGraphRuntime,
)
from resonforge.transcribers.muscriptor.runtime.decode.lines import (
    DecodeLine,
    EagerLine,
    GraphLine,
)
from resonforge.transcribers.muscriptor.runtime.instrumentation import (
    _CudaPhaseRecorder,
)
from resonforge.transcribers.muscriptor.runtime.prefill.prepare import (
    prepare_generation_conditions,
    prepare_generation_row,
    prepare_generation_rows,
)
from resonforge.transcribers.muscriptor.runtime.rows import (
    ContinuousGenerationStats,
    GenerationHandle,
    GenerationPhase,
    PreparedConditionBatch,
    PreparedGenerationRow,
    condition_batch_key,
    condition_span_headroom,
    split_prepared_condition_batch,
)

# Re-exported, not used here. This module was one file until the prefill half
# moved out; every caller that imported these from it keeps working, and the
# split stays an internal fact rather than a rename every importer pays for.
__all__ = [
    "ContinuousGenerationBatch",
    "ContinuousGenerationStats",
    "GenerationHandle",
    "GenerationPhase",
    "PreparedConditionBatch",
    "PreparedGenerationRow",
    "condition_batch_key",
    "condition_span_headroom",
    "create_continuous_generation_session",
    "prepare_generation_conditions",
    "prepare_generation_row",
    "prepare_generation_rows",
    "split_prepared_condition_batch",
]

from resonforge.transcribers.muscriptor.runtime.generation_state import (
    allocate_condition_slots,
    copy_prepared_slot_state,
)

_LOGGER = logging.getLogger("muscriptor.stderr")












def _export_key(context: tuple[object, ...] | None) -> object | None:
    """Which activation export a row's output belongs to.

    The first element of a row's trace context is its run. One model serves
    several concurrently, so a single sink would file one song's activations
    under another's -- worse than collecting nothing, because the labels would
    be wrong and nothing would say so.

    A dict lookup per row-step, against a decode step; the alternative was
    threading a sink through six `GenerationRequest` construction sites.
    """
    return context[0] if context else None


def _collector_allows_graph(collector: object) -> bool:
    """Whether a row carrying this collector may decode inside a graph.

    Recovery rows carry a margin collector, and excluding every row with ANY collector put the whole recovery lane on the per-token path -- every eager step of a run came from that rule.
    A margin collector needs only the chosen-token margin, which the capture now computes on device, so it's fed after the replay instead of during it.

    A debug tracer is different and still forces eager decode: it wants hidden
    states and full per-step logits, which a replay does not surface. A margin
    collector with a tracer chained behind it reports itself the same way.
    """
    if collector is None:
        return True
    return bool(getattr(collector, "graph_capturable", False))
















@dataclass
class _PreemptedGeneration:
    """One row whose KV was thrown away, and everything needed to rebuild it.

    Distinct from the spill it replaced, which MOVED a row's KV somewhere it still occupied memory -- now deleted outright, store and all, because nothing had written to it since A3.
    A preempted row keeps only what can't be recomputed: emitted tokens plus the sampling and grammar state saying where in its generation it was. Pages go back to the arena.

    Not bit-exact, deliberately (A3-3d). A decoded KV is written one token at a time by the Tq=1 kernel, a re-prefilled one in a single dense pass; they differ ~1e-4 relative.
    Greedy decoding tolerates that. Nothing here promises token identity the way `resume` does.
    """

    row: PreparedGenerationRow
    forbidden: torch.Tensor
    sampling_mask: torch.Tensor
    sampling_seed: torch.Tensor
    sampling_position: torch.Tensor | None
    temporal_floor: torch.Tensor
    temporal_checkpoint_floor: torch.Tensor
    phase: GenerationPhase
    checkpoint_tokens: frozenset[int]
    checkpoint_emitted: bool
    capacity: int
    # Whether it was parked when it was displaced. A parked row is the case
    # preemption exists *for* here: it is doing nothing, its KV is rebuildable
    # bit-for-bit, and it is holding a slot its own downstream members need.
    # Restored with it, so a handle the scheduler still holds resumes exactly
    # as it would have.
    paused: bool = False




















@dataclass
class _Slot:
    capacity: int
    row: PreparedGenerationRow | None = None
    # Two fields because one number carried two requirements pointing opposite ways.
    # `generation` = who occupies this slot NOW. `_resolve_handle` rejects a handle whose generation differs, and a rebuilt row must come back under the generation the scheduler still holds -> a restore writes it BACKWARDS, deliberately.
    # `issued` = what this slot has EVER handed out. The scheduler keys `_producer_waits`, `_handle_claim_tokens` and `resident_handles_by_key` on `(session_id, slot, generation)`, so a number may never be issued twice.
    # With one field it was: nine backwards moves in a single run (40 -> 37), after which install and pause re-minted 38 and 39 -- and 39 was the parent checkpoint's own number. The R9 collision.
    # Monotonic; restores don't touch it.
    generation: int = 0
    issued: int = 0
    phase: GenerationPhase = GenerationPhase.CONTINUE
    paused: bool = False
    checkpoint_tokens: frozenset[int] = frozenset()
    checkpoint_emitted: bool = False
    pending_tokens: tuple[int, ...] = ()
    guard_checkpoint_pending: bool = False


_SESSION_IDS = count(1)


def _unique_tensor_storage_bytes(value: object) -> int:
    """Count unique tensor storages in nested runtime state."""
    seen: set[tuple[str, int]] = set()

    def visit(candidate: object) -> int:
        if isinstance(candidate, torch.Tensor):
            storage = candidate.untyped_storage()
            key = (str(candidate.device), storage.data_ptr())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(candidate, dict):
            return sum(visit(item) for item in candidate.values())
        if isinstance(candidate, (list, tuple)):
            return sum(visit(item) for item in candidate)
        return 0

    return visit(value)


def _snapshot_to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _snapshot_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(_snapshot_to_cpu(item) for item in value)
    return value


def _snapshot_to_device(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {
            key: _snapshot_to_device(item, device) for key, item in value.items()
        }
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(_snapshot_to_device(item, device) for item in value)
    return value




class _InFlightQuantum:
    """A launched, unconsumed graph replay (S5 host-async decode).

    `tokens` is device-side; nothing has synchronized on it. `rows` pins each active slot's row identity at launch, so `consume_quantum` can refuse a slot mutation that slipped in between -- while a quantum is in flight the slots must not change."""

    __slots__ = ("active", "rows", "tokens", "quantum")

    def __init__(
        self,
        active: tuple[int, ...],
        rows: tuple[int, ...],
        tokens: torch.Tensor,
        quantum: int,
    ) -> None:
        self.active = active
        self.rows = rows
        self.tokens = tokens
        self.quantum = quantum


class ContinuousGenerationBatch:
    """Keep a fixed physical batch while logical requests enter and leave."""

    def __init__(
        self,
        lm,
        requests: tuple[GenerationRequest, ...],
        *,
        width: int | None = None,
        capacity: int | None = None,
    ) -> None:
        init_started = time.perf_counter()
        self._init_wall_spans: list[tuple[str, float, float]] = []
        if not requests:
            raise ValueError("continuous batch requires at least one request")
        key = requests[0].session_compatibility_key
        if any(request.session_compatibility_key != key for request in requests):
            raise ValueError("continuous generation rows are incompatible")
        self.lm = lm
        self.session_id = next(_SESSION_IDS)
        self.compatibility_key = key
        self.width = len(requests) if width is None else width
        if self.width < len(requests):
            raise ValueError("session width cannot be smaller than initial rows")
        requested_capacity = max(request.max_gen_len for request in requests)
        if capacity is not None:
            requested_capacity = max(requested_capacity, capacity)
        self.cfg_enabled = requests[0].cfg_coef != 1.0
        self.cfg_coef = requests[0].cfg_coef
        self.temperature = requests[0].temperature
        self.eos_id = requests[0].eos_id
        self.device = lm.emb.weight.device
        if any(
            request.use_sampling and request.sampling_seed is None
            for request in requests
        ):
            raise ValueError("continuous sampling rows require a stable seed")
        self.sampling_mask = torch.zeros(
            self.width,
            device=self.device,
            dtype=torch.bool,
        )
        self.sampling_mask[: len(requests)] = torch.tensor(
            [request.use_sampling for request in requests],
            device=self.device,
            dtype=torch.bool,
        )
        self.sampling_seeds = torch.zeros(
            self.width,
            device=self.device,
            dtype=torch.long,
        )
        self.sampling_seeds[: len(requests)] = torch.tensor(
            [request.sampling_seed or 0 for request in requests],
            device=self.device,
            dtype=torch.long,
        )
        self.sampling_positions = torch.ones(
            self.width,
            device=self.device,
            dtype=torch.long,
        )
        configured_grammar = next(
            (
                request.temporal_grammar
                for request in requests
                if request.temporal_grammar is not None
            ),
            None,
        )
        self._temporal_shift_values_key = (
            () if configured_grammar is None else configured_grammar.shift_values
        )
        if any(
            (
                ()
                if request.temporal_grammar is None
                else request.temporal_grammar.shift_values
            )
            != self._temporal_shift_values_key
            for request in requests
        ):
            raise ValueError("continuous rows require one temporal token mapping")
        if (
            configured_grammar is not None
            and len(configured_grammar.shift_values) > lm.card
        ):
            raise ValueError("temporal grammar vocabulary exceeds model card")
        self.temporal_shift_values = torch.tensor(
            (
                (-1,) * lm.card
                if configured_grammar is None
                else (
                    *configured_grammar.shift_values,
                    *((-1,) * (lm.card - len(configured_grammar.shift_values))),
                )
            ),
            device=self.device,
            dtype=torch.long,
        )
        self.temporal_floors = torch.full(
            (self.width,),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self.temporal_checkpoint_floors = torch.full(
            (self.width,),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self.sequence = torch.full(
            (self.width, 1),
            lm.initial_token_id,
            device=self.device,
            dtype=torch.long,
        )
        self._state_rows: dict[
            tuple[int, ...], tuple[torch.Tensor, tuple[int, ...]]
        ] = {}
        # `active` as a device tensor. Separate from `_state_rows` only because a CFG session's state rows are the active set PLUS its mirror,
        # so the two tuples differ; with CFG off they are equal and this cache defers to that one rather than making a second copy of it.
        self._active_rows: dict[tuple[int, ...], torch.Tensor] = {}
        # Graphs bind tensor addresses, but the physical slot selected for a
        # compact active cohort may change after pause/resume or replacement.
        # Keep one graph-owned row-index tensor per active width and update its
        # contents in place before capture/replay instead of keying the Graph
        # cache by every slot permutation.
        self._graph_state_rows: dict[int, torch.Tensor] = {}
        self._graph_state_row_values: dict[int, tuple[int, ...]] = {}
        self._prefill_batches_by_width: Counter[int] = Counter()
        self._condition_batches_by_width: Counter[int] = Counter()
        self._first_token_batches_by_width: Counter[int] = Counter()
        self._packed_prefill_batches_by_width: Counter[int] = Counter()
        self._phase_recorder = _CudaPhaseRecorder(self.device)
        prefill_started = time.perf_counter()
        self._init_wall_spans.append(
            ("session_setup", init_started, prefill_started)
        )
        with self._phase_recorder.measure("prefill"):
            prepared = prepare_generation_rows(
                lm,
                requests,
                phase_recorder=self._phase_recorder,
                _record_condition_batch=(
                    lambda width: self._condition_batches_by_width.__setitem__(
                        width,
                        self._condition_batches_by_width[width] + 1,
                    )
                ),
                _record_first_token_batch=(
                    lambda width: self._first_token_batches_by_width.__setitem__(
                        width,
                        self._first_token_batches_by_width[width] + 1,
                    )
                ),
                _record_packed_prefill_batch=(
                    lambda width: self._packed_prefill_batches_by_width.__setitem__(
                        width,
                        self._packed_prefill_batches_by_width[width] + 1,
                    )
                ),
            )
        prefill_finished = time.perf_counter()
        self._init_wall_spans.append(
            ("session_prefill", prefill_started, prefill_finished)
        )
        self._record_prefill_widths(requests)
        first = prepared[0]
        # ``max_gen_len`` excludes the condition and forced prompt already
        # written into the private KV cache. Reusing only that value here can
        # shrink a valid prepared row into a smaller persistent cache, then
        # fail later Graph decode at ``kv-capacity``. Preserve the true prepared
        # cache length while retaining the scheduler's requested minimum.
        #
        # From the row's own ``decode_span`` rather than from the size of its
        # prefill allocation. The two agreed only while prefill allocated for
        # the whole generation it never writes; see ``_prefill_states``.
        prepared_capacity = max(row.decode_span for row in prepared)
        # Sizing the session to exactly what the *first* batch prepared makes
        # the fit a coincidence: slots are hot-replaced for the session's whole
        # life, and a replacement legitimately carries a longer condition prefix
        # — one class-conditioned token per requested instrument group, so a
        # stem admitted into a session an unconditioned stem created needs more
        # positions than it was built with. Until the guards above, those rows
        # reached the KV and position tensors as an out-of-range index rather
        # than as an error. Size for the declared maximum prefix instead of
        # detecting the overflow afterwards.
        session_capacity = max(
            requested_capacity, prepared_capacity
        ) + condition_span_headroom(
            first.conditions,
            getattr(lm, "condition_span_bounds", None),
        )
        self.model_state = init_states(
            lm,
            batch_size=self.width * (2 if self.cfg_enabled else 1),
            sequence_length=session_capacity,
        )
        bind_continuous_decode_metadata(self.model_state)
        self.conditions = allocate_condition_slots(
            first.conditions,
            self.width,
            cfg_enabled=self.cfg_enabled,
        )
        self.forbidden = torch.zeros(
            (self.width, lm.card), device=self.device, dtype=torch.bool
        )
        self.slots = [_Slot(session_capacity) for _ in range(self.width)]
        self._preempted: dict[tuple[int, int], _PreemptedGeneration] = {}
        self._preemption_order: deque[GenerationHandle] = deque()
        # Preemption's own ledger. Monotone within a session and read as a
        # delta by the scheduler, the same way graph telemetry is: these are
        # the numbers the "drop the KV and re-prefill, or move it to host
        # memory" decision was deferred until they existed.
        self._preemptions: Counter[str] = Counter()
        self._preemption_quantum: dict[tuple[int, int], int] = {}
        self._quanta = 0
        # S5 host-async decode: one launched, unconsumed graph replay, or None. See `launch_quantum`.
        self._in_flight_quantum: _InFlightQuantum | None = None
        self._cuda_graph_runtime = self._create_cuda_graph_runtime()
        # Decided once, here, and never revisited. A session that runs graph
        # decode runs *only* graph decode: reaching the per-token path below is
        # a broken invariant, not a slower route. Interleaving the two let each
        # mask defects in the other: a few percent of production decode ran
        # eager and nothing failed, in either direction.
        # One value, not a flag beside a guard beside a branch. A graph
        # session holds a line that has no per-token path at all, which is
        # what makes "graph decode never falls back" a property of the
        # object rather than a check inside shared code.
        self._decode_line: DecodeLine = (
            GraphLine() if self._cuda_graph_runtime is not None else EagerLine()
        )
        for index, row in enumerate(prepared):
            self.install(index, row)
        self._init_wall_spans.append(
            ("session_arena_init", prefill_finished, time.perf_counter())
        )

    def drain_init_wall_spans(self) -> tuple[tuple[str, float, float], ...]:
        spans = tuple(self._init_wall_spans)
        self._init_wall_spans.clear()
        return spans

    def close(self) -> None:
        """Release an owner-free arena and its address-bound Graph cache."""
        if self.active_count or self.occupied_count or self.displaced_count:
            raise RuntimeError("cannot close a continuous session with live rows")
        if self._cuda_graph_runtime is not None:
            self._cuda_graph_runtime.dispatcher.clear()
            self._cuda_graph_runtime = None
        # Before the dict is cleared: the pages are a range inside a pool that
        # outlives this session, and dropping the reference frees nothing.
        release_state_blocks(self.model_state)
        self.model_state.clear()
        self.conditions.clear()
        self._state_rows.clear()
        self._active_rows.clear()
        self._graph_state_rows.clear()
        self._graph_state_row_values.clear()
        self._preempted.clear()
        self._preemption_order.clear()
        self.slots.clear()

    def _create_cuda_graph_runtime(self) -> ContinuousDecodeGraphRuntime | None:
        if not getattr(self.lm, "_decode_graph_enabled", False):
            return None
        if self.device.type != "cuda":
            return None
        # Graph ceiling set HERE, from this session's own width. It used to be assigned by the caller before construction, which held while a width was decided once;
        # a session that can now widen would keep the ceiling it opened at and reject its own decode with `batch-too-wide`. One number, one place, every time the runtime is built.
        # Only ever RISES, because `lm` is shared by every session of this model -- a narrow session lowering it would reject a wide sibling's decode instead of its own.
        ceiling = max(
            int(getattr(self.lm, "_decode_graph_max_batch_size", 0)),
            int(self.width),
        )
        self.lm._decode_graph_max_batch_size = ceiling
        return ContinuousDecodeGraphRuntime(
            self.lm,
            self.model_state,
            bucket_size=int(getattr(self.lm, "_decode_graph_bucket_size", 64)),
            capture_bucket_size=int(
                getattr(self.lm, "_decode_graph_capture_bucket_size", 256)
            ),
            max_batch_size=ceiling,
            max_graphs=int(getattr(self.lm, "_decode_graph_max_graphs", 16)),
        )

    def install(self, slot: int, row: PreparedGenerationRow) -> None:
        target = self.slots[slot]
        if row.request.session_compatibility_key != self.compatibility_key:
            raise ValueError("replacement row is incompatible")
        if target.row is not None:
            raise ValueError("cannot replace an active slot")
        if row.request.max_gen_len > target.capacity:
            raise ValueError("replacement row exceeds slot capacity")
        # `max_gen_len` excludes the condition and forced prompt, so it's the wrong quantity to admit a replacement by:
        # the row decodes from `prepend_length + max_gen_len` and the session's shared position table was sized once at construction.
        # A row clearing the check above on `max_gen_len` alone can still index past `position_capacity` -- not a Python error but a device-side assert that poisons the CUDA context for every session on the device.
        #
        # The row states it (`decode_span`). Used to be read off the prefill allocation's `capacity`, which carried the number only because prefill allocated for a generation it never writes.
        prepared_span = row.decode_span
        if prepared_span > target.capacity:
            raise ValueError(
                "replacement row exceeds slot capacity: prepared span "
                f"{prepared_span} > {target.capacity} "
                f"(max_gen_len={row.request.max_gen_len})"
            )
        grammar = row.request.temporal_grammar
        mapping = () if grammar is None else grammar.shift_values
        if mapping != self._temporal_shift_values_key:
            raise ValueError("replacement row changed temporal token semantics")
        copy_prepared_slot_state(
            self.lm,
            self.model_state,
            row.model_state,
            slot=slot,
            width=self.width,
            cfg_enabled=self.cfg_enabled,
            device=self.device,
            source_slot=row.source_slot,
            source_width=row.source_width,
        )
        # Conditions have already been consumed into the private KV cache by
        # first_step=True. Shared fixed-slot decoding is always first_step=False,
        # so encoded condition shapes may differ across stems without padding.
        self.forbidden[slot].zero_()
        if row.request.forbidden_token_ids:
            self.forbidden[slot, list(row.request.forbidden_token_ids)] = True
        row.release_prefill()
        target.row = row
        target.issued += 1
        target.generation = target.issued
        target.phase = GenerationPhase.CONTINUE
        target.paused = False
        target.checkpoint_tokens = frozenset(row.request.checkpoint_token_ids)
        target.checkpoint_emitted = False
        target.pending_tokens = ()
        target.guard_checkpoint_pending = row.guard_action in {
            "interrupt_to_recovery",
            "reject_candidate",
        }
        if row.request.use_sampling and row.request.sampling_seed is None:
            raise ValueError("sampling replacement row is missing its seed")
        self.sampling_mask[slot] = row.request.use_sampling
        self.sampling_seeds[slot] = row.request.sampling_seed or 0
        self.sampling_positions[slot] = 1
        self.temporal_floors[slot] = row.temporal_floor
        self.temporal_checkpoint_floors[slot] = row.checkpoint_floor
        self.sequence[slot, 0] = row.last_token

    @property
    def active_count(self) -> int:
        """Number of runnable rows; paused rows remain resident but inactive.

        Preempted rows count -- UNLESS they were parked when displaced.
        A running one occupies no slot and holds no KV, but it's unfinished work this session owes, and every driver loop is `while session.active_count`.
        Leaving those out let a session whose last live row finished stop with a preempted row still holding its tokens: caller sees two results where it asked for three, no error. Counting them is also what runs the quantum that restores them.

        A PARKED one is the opposite. Nothing this loop does can resume it -- only the handle its holder carries can -- so counting it spins the driver forever.
        A hang, not a stall, and it looks exactly like the one a silent `return` in `_restore_preempted_rows` caused once.
        """
        return sum(
            slot.row is not None and not slot.paused for slot in self.slots
        ) + sum(
            1 for preempted in self._preempted.values() if not preempted.paused
        )

    @property
    def displaced_count(self) -> int:
        """Rows whose external handles still depend on this session.

        A preempted PARKED row is one of them, and leaving it out breaks the scheduler's `resident_handles == paused + displaced` invariant:
        its slot is empty so it isn't paused, it was never copied so it isn't displaced, and the handle its holder carries counts against a location neither term names.
        Surfaced as `SchedulerInvariantError: resident handle location is not unique`, which reads like a duplicate and is the opposite -- a handle whose location nothing claims.

        Only the parked ones. A row preempted mid-decode has no resident handle: it was running, not waiting to be resumed.
        """
        return sum(
            1 for preempted in self._preempted.values() if preempted.paused
        )

    @property
    def occupied_count(self) -> int:
        """Number of slots retaining row state, including paused rows."""
        return sum(slot.row is not None for slot in self.slots)

    @property
    def available_count(self) -> int:
        return self.width - self.occupied_count

    @property
    def remaining_decode_token_budget(self) -> int:
        """Estimated runnable work for deterministic scheduler scoring."""
        return sum(
            max(
                0,
                slot.row.request.remaining_decode_token_budget
                - max(0, slot.row.steps - len(slot.row.request.prompt_ids)),
            )
            for slot in self.slots
            if slot.row is not None and not slot.paused
        )

    @property
    def next_completion_token_budget(self) -> int | None:
        """Conservative token horizon for the nearest active row release."""
        budgets = [
            max(
                0,
                slot.row.request.remaining_decode_token_budget
                - max(0, slot.row.steps - len(slot.row.request.prompt_ids)),
            )
            for slot in self.slots
            if slot.row is not None and not slot.paused
        ]
        return min(budgets) if budgets else None

    def _resolve_handle(self, handle: GenerationHandle) -> _Slot:
        if handle.session_id != self.session_id:
            raise ValueError("generation handle belongs to another session")
        if not 0 <= handle.slot < self.width:
            raise ValueError("generation handle has an invalid slot")
        slot = self.slots[handle.slot]
        if slot.row is None or slot.generation != handle.generation:
            raise ValueError("generation handle is stale")
        return slot

    @staticmethod
    def _handle_key(handle: GenerationHandle) -> tuple[int, int]:
        return handle.slot, handle.generation

    def _free_slot_for(self, row: PreparedGenerationRow) -> int | None:
        """The first slot that can hold this row, or `None`.

        ANY free slot, not the one it left. A parked row is out of its slot by definition -- that's what parking was for -- so the slot it names belongs to whoever took it.
        Coming back only to that slot waits forever the moment anything else does, and `active_count` counts a preempted row, so the driver loop never ends either.

        One search, because there were three: `can_resume` asked "is there any slot", the spill restore relocated, and `restore_preempted` installed into `handle.slot` regardless.
        The first two agreed and the third contradicted both -- `can_resume` said yes and the rebuild then raised `cannot replace an active slot` on a real run.
        A fourth survived that collapse, inline in `admit_prepared`, spelled identically and never noticed -- which is the whole reason this docstring exists.
        """
        return next(
            (
                index
                for index, slot in enumerate(self.slots)
                if slot.row is None and row.request.max_gen_len <= slot.capacity
            ),
            None,
        )

    def can_resume(self, handle: GenerationHandle) -> bool:
        """Whether a resident or displaced checkpoint can resume immediately."""
        if handle.session_id != self.session_id:
            raise ValueError("generation handle belongs to another session")
        key = self._handle_key(handle)
        parked = self._preempted.get(key)
        if parked is not None:
            # Out of its slot either way, same question: is there a slot to come back to.
            # A preempted row also needs pages and asks for them when `resume` rebuilds it -- the same exhaustion every other growth reports, at the same place.
            return self._free_slot_for(parked.row) is not None
        self._resolve_handle(handle)
        return True

    def is_displaced(self, handle: GenerationHandle) -> bool:
        """Whether this row is out of its slot. The name is the protocol's.

        True for a preempted row too -- every caller is asking "does resuming this need a slot back", and the answer is the same whether the row was copied elsewhere or thrown away and rebuilt.
        """
        key = self._handle_key(handle)
        return key in self._preempted

    def can_preempt_for_admission(self) -> bool:
        """Whether one paused resident row can be moved out of a fixed slot."""
        return any(
            slot.row is not None and slot.paused for slot in self.slots
        )

    def preempt_for_admission(self) -> bool:
        """Free one slot by preempting a parked row, not by moving it.

        This used to copy the row's KV into a second arena, freeing a slot and no memory -- and under paging freeing nothing at all, because both arenas draw the same pool.
        That allocation is why the whole path sat behind an environment switch.

        Preemption is the same slot for none of the cost: pages go back to the supply, tokens are kept, and `resume` rebuilds by re-running what wrote the KV -- bit-identical, measured 0.0e+00 across every position.
        A parked row is the ideal victim: doing nothing, and its replay is paid at the moment it was going to run anyway.

        This removes the hold-and-wait the dependency-credit reservation exists to prevent -- a parent parked at a checkpoint stops holding the slot its own downstream members need.
        """
        index = next(
            (
                index
                for index, slot in enumerate(self.slots)
                if slot.row is not None and slot.paused
            ),
            None,
        )
        if index is None:
            return False
        with self._phase_recorder.measure("checkpoint_preempt"):
            self._preemption_order.append(self.preempt(index))
        return True

    def preempt(self, slot_index: int) -> GenerationHandle:
        """Throw one row's KV away and give its pages back.

        The row keeps its tokens, so nothing it produced is lost; what is lost
        is the work of having computed the KV, which `restore_preempted` pays
        again. That is the trade preemption makes: memory now against compute
        later, on a row chosen because it is the cheapest to recompute.
        """
        slot = self.slots[slot_index]
        row = slot.row
        if row is None:
            raise ValueError("no row occupies this slot")
        self._commit_pending(slot_index, slot)
        # The two halves of what a preemption costs, counted where they are
        # incurred: one row displaced, and the tokens its rebuild will have to
        # re-run. Frequency alone cannot say whether preemption is cheap --
        # displacing a row that has emitted three tokens and one that has
        # emitted three hundred are the same event and a hundredfold different
        # bill.
        self._preemptions["preemptions"] += 1
        self._preemptions["preempted_tokens"] += len(row.tokens)
        handle = GenerationHandle(
            session_id=id(self),
            slot=slot_index,
            generation=slot.generation,
            phase=slot.phase,
            steps=row.steps,
            token_count=len(row.tokens),
            logical_bucket=row.request.initial_kv_capacity_bucket,
            remaining_token_budget=max(
                0, row.request.max_gen_len - len(row.tokens)
            ),
        )
        # When it was displaced, so the restore can report how long it waited.
        # Quanta rather than seconds: a preempted row is not slow, it is
        # *absent*, and what matters is how many turns of everyone else's work
        # went past while it was.
        self._preemption_quantum[self._handle_key(handle)] = self._quanta
        self._preempted[self._handle_key(handle)] = _PreemptedGeneration(
            row=row,
            forbidden=self.forbidden[slot_index].clone(),
            sampling_mask=self.sampling_mask[slot_index].clone(),
            sampling_seed=self.sampling_seeds[slot_index].clone(),
            sampling_position=(
                self.sampling_positions[slot_index].clone()
                if self.sampling_positions is not None
                else None
            ),
            temporal_floor=self.temporal_floors[slot_index].clone(),
            temporal_checkpoint_floor=(
                self.temporal_checkpoint_floors[slot_index].clone()
            ),
            phase=slot.phase,
            checkpoint_tokens=slot.checkpoint_tokens,
            checkpoint_emitted=slot.checkpoint_emitted,
            capacity=slot.capacity,
            paused=slot.paused,
        )
        state_rows, state_rows_host = self._active_state_rows((slot_index,))
        # the reclaim: pages go back to the arena's reserve, where another row can grow into them
        reset_state_rows(
            self.lm, self.model_state, state_rows, host_rows=state_rows_host
        )
        slot.row = None
        self.temporal_floors[slot_index] = -1
        self.temporal_checkpoint_floors[slot_index] = -1
        slot.paused = False
        slot.phase = GenerationPhase.CONTINUE
        slot.checkpoint_tokens = frozenset()
        slot.checkpoint_emitted = False
        slot.pending_tokens = ()
        slot.guard_checkpoint_pending = False
        return handle

    def restore_preempted(self, handle: GenerationHandle) -> int:
        """Rebuild a preempted row by re-running what wrote its KV.

        NOT by prefilling its output as a prompt. Tried and measured: a prefilled KV differs from a decoded one by 2.3 absolute against a mean magnitude of 0.44 on this model family
        -- five times the signal, not fp drift -- and the continuation diverges within a few tokens.
        The prompt is laid out by the delay pattern, the decode path writes one token at a time. Same tokens, different tensor.

        So the rebuild re-runs what wrote it: the row's own prefill with its original request unchanged, then one Tq=1 step per emitted token, each discarding what it samples and feeding back the token the row actually produced.
        Same kernels, same order, same inputs -- identical by construction, not by tolerance.

        The row object survives: guard, findings and temporal frontier are accumulated state a fresh prefill would silently reset.

        Returns the slot it rebuilt into, NOT necessarily the one the handle names (see `_free_slot_for`). The handle is spent either way -- it addresses a parked row, and this row is about to run.
        """
        key = self._handle_key(handle)
        preempted = self._preempted.pop(key, None)
        if preempted is None:
            raise ValueError("generation handle was not preempted")
        slot_index = self._free_slot_for(preempted.row)
        if slot_index is None:
            # `can_resume` asks exactly this and must match -- reaching here means the caller resumed a row it was told could not resume
            self._preempted[key] = preempted
            raise ValueError("no free slot can hold the preempted row")
        displaced_at = self._preemption_quantum.pop(key, None)
        if displaced_at is not None:
            self._preemptions["preemption_wait_quanta"] += self._quanta - displaced_at
        row = preempted.row
        prompt_length = len(row.request.prompt_ids)
        if len(row.tokens) <= prompt_length:
            raise ValueError("a row with no generated tokens has nothing to rebuild")
        # This re-encodes conditions in a batch of one. It used to be R10's known defect: a batched encode padded `instrument_group` to the batch max, so the lone re-encode yielded a SHORTER prepend and the rebuilt row sat at a different offset.
        # Fixed 2026-08-29 at the conditioner: `ClassConditioner.minimum_length` floors every encode at the group vocabulary, so alone and batched produce the same prepend length by construction.
        # (Reusing `row.conditions` was NOT a fix and was tried: `release_prefill` clears them before a row can be preempted.)
        rebuilt = prepare_generation_row(
            self.lm, row.request, phase_recorder=self._phase_recorder
        )
        if rebuilt.last_token != row.tokens[prompt_length]:
            # Counted, not fatal. This used to raise, reasoning that a prefill is deterministic in its own inputs. It isn't, across WIDTH: a row prefilled as one of N shares one state and the rebuild prefills it alone, so the reductions differ.
            # With only width moving, the KV diverges at the shortest prompts and is exactly zero by prompt 32 -- orders below what killed prompt-replay in A3-3d, and production divergences were all in the short range (R10).
            # And the token it compares is DISCARDED: `self.sequence` below is loaded with the row's own token whatever the rebuild sampled, so this was always a proxy for KV equality, not a correctness gate.
            # Real quantity is restored KV vs live KV, gated by `test_a_row_prefilled_beside_another_survives_preemption`.
            self._preemptions["rebuild_token_divergences"] += 1
        grafted = dataclasses.replace(
            row,
            model_state=rebuilt.model_state,
            conditions=rebuilt.conditions,
            source_slot=rebuilt.source_slot,
            source_width=rebuilt.source_width,
            prefill_arena=rebuilt.prefill_arena,
        )
        self.install(slot_index, grafted)
        # `install` leaves the slot where the *original* install left it, which
        # is where the replay has to start: the prefill's own token loaded and
        # the sampling position at one.
        self.sequence[slot_index, 0] = row.tokens[prompt_length]
        self.sampling_positions[slot_index] = 1
        self.forbidden[slot_index] = preempted.forbidden
        self.sampling_mask[slot_index] = preempted.sampling_mask
        self.sampling_seeds[slot_index] = preempted.sampling_seed
        self.temporal_floors[slot_index] = row.temporal_floor
        replayed = row.tokens[prompt_length + 1 :]
        self._preemptions["restores"] += 1
        # What the rebuild actually cost, as opposed to what `preempted_tokens`
        # said it would owe: the prompt is prefilled in one pass and only the
        # generated tail is re-run a token at a time.
        self._preemptions["replayed_tokens"] += len(replayed)
        with self._phase_recorder.measure("preemption_replay"):
            self._replay_tokens(slot_index, replayed)
        if (
            self.sampling_positions is not None
            and preempted.sampling_position is not None
        ):
            self.sampling_positions[slot_index] = preempted.sampling_position
        self.temporal_floors[slot_index] = preempted.temporal_floor
        self.temporal_checkpoint_floors[slot_index] = (
            preempted.temporal_checkpoint_floor
        )
        slot = self.slots[slot_index]
        # `install` advances the generation because installing is normally a NEW occupant taking the slot.
        # A rebuild is the same row: advancing makes every handle the scheduler still holds stale, and the one it holds is precisely the displaced row.
        # This is the one write that moves `generation` backwards, and why `issued` exists -- the slot's identity supply must not follow it back, or the next mint re-issues a live checkpoint's number.
        slot.generation = handle.generation
        slot.phase = preempted.phase
        slot.checkpoint_tokens = preempted.checkpoint_tokens
        slot.checkpoint_emitted = preempted.checkpoint_emitted
        # `install` unparks -- a fresh row is runnable by definition. A row parked when displaced must come back parked, or the handle the scheduler holds resumes something already running.
        slot.paused = preempted.paused
        return slot_index

    def _preempt_for_pages(self, active: tuple[int, ...]) -> bool:
        """Free pages by preempting the cheapest row to rebuild. False if none.

        Cheapest, because a preempted row pays one decode step per emitted token -- fewest tokens = least work thrown away. vLLM preempts the newest sequence for the same reason.

        The LAST row too. Refusing there looked right (preempting the only row that could use the pages seems to make no progress) and is wrong, because progress is a property of the DEVICE, not this session.
        Preemption is session-local while the supply is shared, so a session down to one row is exactly where its pages are worth more to somebody else: it gives them up, the other session finishes, the row is rebuilt from what comes back.

        The genuinely terminal case is still terminal one step later, with a better message: `_restore_preempted_rows` raises when no row anywhere is left to release a page.

        Starvation is possible and NOT addressed -- a row that keeps being youngest keeps being chosen.
        Bounded only by a restored row being older than the one that replaced it. Worth measuring before it's worth solving.
        """
        tables = block_tables(self.model_state)
        if tables is None:
            return False
        # Parked rows first, not as a tie-break: a parked row runs nothing, so displacing it costs no progress now, and its replay is paid at the moment it was going to resume anyway.
        parked = [
            index
            for index, slot in enumerate(self.slots)
            if slot.row is not None
            and slot.paused
            and tables.mapped_blocks(index) > 0
        ]
        candidates = parked or [
            index
            for index in active
            if self.slots[index].row is not None
            and tables.mapped_blocks(index) > 0
        ]
        if not candidates:
            # Nothing here holds a page, so the shortage is somebody else's to
            # answer -- and the caller re-raises. Counted separately from a
            # preemption, because "we could not help" and "we displaced a row"
            # are opposite outcomes that a single frequency would merge.
            self._preemptions["preemption_refusals"] += 1
            return False
        victim = min(candidates, key=lambda index: len(self.slots[index].row.tokens))
        self._preemption_order.append(self.preempt(victim))
        return True

    def _restore_preempted_rows(self) -> None:
        """Rebuild preempted rows, oldest first, while the pages are there.

        Oldest first, not cheapest: cheapest-first chose the victim, and using it here too lets one row be preempted and restored repeatedly while an older one never comes back.

        A restore must NOT consume the last page. Restoring the moment the row's own pages are available re-creates the shortage that preempted it and the next quantum preempts it again
        -- measured as two rows preempted and restored alternately while a third made all the progress.
        Headroom required is one block per live row, exactly what the next boundary crossing costs, so it's derived from the batch rather than chosen.
        """
        tables = block_tables(self.model_state)
        if tables is None:
            return
        while self._preemption_order:
            handle = self._preemption_order[0]
            preempted = self._preempted.get(self._handle_key(handle))
            if preempted is None:
                self._preemption_order.popleft()
                continue
            if preempted.paused:
                # Off the queue but still in `_preempted`: a parked row comes back when its holder asks, via `resume`.
                # Leaving it at the head blocks every row behind it on an event this loop can't cause.
                self._preemption_order.popleft()
                continue
            if self._free_slot_for(preempted.row) is None:
                # `restore_preempted` would raise; there's a later turn to retry on, so leave the queue alone
                return
            # `+ 1`: the replay's last step writes one position past the last
            # token it feeds, exactly as a decode quantum does.
            needed = blocks_for_tokens(len(preempted.row.tokens) + 1)
            live = sum(1 for slot in self.slots if slot.row is not None)
            if tables.spare_pages < needed + live:
                if live:
                    # Someone can still make progress and hand pages back.
                    return
                # Nobody can. Waiting here is the livelock this whole path
                # exists to avoid: `active_count` counts preempted rows, so a
                # silent `return` with nothing running spins the driver's loop
                # forever -- a hang that burns host memory and GPU and never
                # ends. Raise instead of returning.
                raise KVPagesExhausted(
                    f"restoring a preempted row needs {needed} pages per "
                    f"layer, {tables.spare_pages} are free, and no row is left "
                    "to release any"
                )
            self._preemption_order.popleft()
            self.restore_preempted(handle)

    @torch.inference_mode()
    def _replay_tokens(self, slot_index: int, tokens: Sequence[int]) -> None:
        """Re-run the decode steps that wrote one row's KV, forcing its tokens.

        Each step writes the KV for the currently loaded token then loads the next known one, discarding what the model sampled.
        The sample is computed and thrown away rather than skipped, because what's being reproduced is the WRITE, and the write happens inside the same forward that produces the sample.

        One row, one token at a time. A graph quantum decides its own following tokens inside the capture, so it can't be handed a sequence to follow
        -- and this isn't decode: no output, no row state advanced. The mode a session chose still governs every step that does.

        Trace collectors are deliberately not passed. A replay would record a
        second copy of steps the collector already has.
        """
        if not tokens:
            return
        state_rows, state_rows_host = self._active_state_rows((slot_index,))
        rows = self._active_rows_tensor((slot_index,))
        # Hoisted: one row, and none of these change across the replay. `EagerLine.launch` hoists exactly this set for the same reason; this path was still re-gathering them per token.
        # `sequence` and `sampling_positions` stay inside -- both are written every step.
        forbidden = self.forbidden.index_select(0, rows)
        sample_mask = self.sampling_mask.index_select(0, rows)
        seeds = self.sampling_seeds.index_select(0, rows)
        floors = self.temporal_floors.index_select(0, rows)
        for token in tokens:
            grow_state_rows(self.model_state, state_rows_host, ahead=1)
            select_state_rows(
                self.model_state, state_rows, host_rows=state_rows_host
            )
            with self.lm.autocast:
                self.lm._sample_next_token(
                    self.sequence.index_select(0, rows),
                    self.conditions,
                    self.model_state,
                    first_step=False,
                    use_sampling=True,
                    temp=self.temperature,
                    top_k=0,
                    top_p=0.0,
                    cfg_coef=self.cfg_coef,
                    forbidden_tokens=forbidden,
                    sample_mask=sample_mask,
                    generator=None,
                    sampling_seeds=seeds,
                    sampling_positions=self.sampling_positions.index_select(
                        0, rows
                    ),
                    trace_collectors=(None,),
                    trace_contexts=((),),
                    temporal_shift_values=self.temporal_shift_values,
                    temporal_floors=floors,
                )
            self.sequence[slot_index, 0] = token
            self.sampling_positions[rows] += sample_mask.long()
            increment_state_rows(
                self.lm.transformer,
                self.model_state,
                state_rows,
                host_rows=state_rows_host,
            )

    def pause(
        self,
        slot_index: int,
        *,
        phase: GenerationPhase = GenerationPhase.PAUSED_PRIMARY,
    ) -> GenerationHandle:
        """Retain an unfinished row and its KV while removing it from decode."""
        if not 0 <= slot_index < self.width:
            raise ValueError("pause slot is outside the session")
        slot = self.slots[slot_index]
        row = slot.row
        if row is None:
            raise ValueError("cannot pause an empty slot")
        if row.finished:
            raise ValueError("cannot pause a finished row")
        if slot.paused:
            raise ValueError("row is already paused")
        return self._pause_slot(slot_index, phase)

    def _pause_slot(
        self,
        slot_index: int,
        phase: GenerationPhase,
    ) -> GenerationHandle:
        slot = self.slots[slot_index]
        row = slot.row
        assert row is not None
        slot.paused = True
        slot.phase = phase
        # A handle names a CHECKPOINT, not an occupancy. It used to advance only in `install`, so pause -> resume -> pause handed out the same `(session_id, slot, generation)` twice
        # -- and that triple keys every index the scheduler keeps (`_producer_waits`, `_handle_claim_tokens`, `resident_handles_by_key`).
        # A second checkpoint then landed on entries the first still owned, reaching the scheduler as `dependency bundle handle also has a producer wait` and `resident handle belongs to multiple dependency bundles`. Two names, one ambiguity.
        # Advancing alone wasn't enough: a restore wrote the counter BACKWARDS, so the next advance re-issued a number the slot had used. Minting from `issued` is what makes it unique -- see `_Slot`.
        slot.issued += 1
        slot.generation = slot.issued
        return GenerationHandle(
            session_id=self.session_id,
            slot=slot_index,
            generation=slot.generation,
            phase=phase,
            steps=row.steps,
            token_count=len(row.tokens),
            logical_bucket=row.request.initial_kv_capacity_bucket,
            remaining_token_budget=max(
                0,
                row.request.remaining_decode_token_budget
                - max(0, row.steps - len(row.request.prompt_ids)),
            ),
        )

    def resume(
        self,
        handle: GenerationHandle,
        *,
        phase: GenerationPhase = GenerationPhase.CONTINUE,
    ) -> GenerationRequest:
        """Make a paused row runnable, rebuilding it first if it was displaced.

        A parked row holds a slot and pages while doing nothing, and its KV is
        rebuildable bit-for-bit, so it is the first thing preemption takes when
        something else needs what it is sitting on. The handle the scheduler
        holds must not know that happened: it asks to resume, and the row is
        there.
        """
        if self._handle_key(handle) in self._preempted:
            # Wherever the rebuild found room, which is not necessarily the
            # slot the handle names. Reading `handle.slot` here is what raised
            # `cannot replace an active slot` on a real run.
            slot_index = self.restore_preempted(handle)
            slot = self.slots[slot_index]
        else:
            slot = self._resolve_handle(handle)
            slot_index = handle.slot
        if not slot.paused:
            raise ValueError("generation handle is not paused")
        self._commit_pending(slot_index, slot)
        slot.phase = phase
        slot.paused = False
        assert slot.row is not None
        return slot.row.request

    def control_outcome(self, *, discarded: bool) -> GenerationControlResult:
        """The result the scheduler publishes for a control action here."""
        return GenerationControlResult(discarded=discarded)

    def discard(self, handle: GenerationHandle) -> GenerationRequest:
        """Release a paused row and invalidate its handle.

        A preempted row is the cheapest case there is: it holds no slot and no
        page, so discarding it is dropping a record. It is also the case that
        would otherwise raise "generation handle is stale", because the slot it
        names belongs to whoever took it.
        """
        key = self._handle_key(handle)
        preempted = self._preempted.pop(key, None)
        if preempted is not None:
            self._preemption_quantum.pop(key, None)
            self._preemption_order = deque(
                entry
                for entry in self._preemption_order
                if self._handle_key(entry) != key
            )
            return preempted.row.request
        slot = self._resolve_handle(handle)
        if not slot.paused:
            raise ValueError("only paused generation can be discarded")
        assert slot.row is not None
        request = slot.row.request
        slot.row = None
        self.temporal_floors[handle.slot] = -1
        self.temporal_checkpoint_floors[handle.slot] = -1
        slot.paused = False
        slot.phase = GenerationPhase.CONTINUE
        slot.checkpoint_tokens = frozenset()
        slot.checkpoint_emitted = False
        slot.pending_tokens = ()
        slot.guard_checkpoint_pending = False
        return request

    @staticmethod
    def _guard_findings(row: PreparedGenerationRow) -> tuple[GuardFinding, ...]:
        guarded = () if row.guard is None else tuple(row.guard.findings)
        return (*guarded, *row.temporal_findings)

    @staticmethod
    def _guard_summary(row: PreparedGenerationRow):
        return (
            None
            if row.guard is None
            else row.guard.summary(row.emitted_eos)
        )

    def _observe_guard_token(self, slot: _Slot, token: int) -> str:
        row = slot.row
        if row is None or row.emitted_eos:
            return "continue"
        action = row.observe_guard_tokens((int(token),))
        if action in {"interrupt_to_recovery", "reject_candidate"}:
            slot.guard_checkpoint_pending = True
        elif action == "reject_terminal":
            row.finished = True
        return action

    def _commit_pending(self, slot_index: int, slot: _Slot) -> None:
        """Commit a graph suffix that was generated past a checkpoint."""
        row = slot.row
        if row is None or not slot.pending_tokens:
            return
        for token in slot.pending_tokens:
            if not row.observe_temporal_token(int(token)):
                break
            row.last_token = int(token)
            row.steps += 1
            row.emitted_eos = row.last_token == self.eos_id
            if not row.emitted_eos:
                row.tokens.append(row.last_token)
                self._observe_guard_token(slot, row.last_token)
            row.finished = row.finished or (
                row.emitted_eos or row.steps >= row.request.max_gen_len
            )
            if row.finished:
                break
        slot.pending_tokens = ()
        slot.checkpoint_emitted = True
        self.sequence[slot_index, 0] = row.last_token

    def collect_checkpoints(
        self,
        checkpointed: Callable[[GenerationRequest, GenerationResult], None],
    ) -> tuple[int, ...]:
        """Pause newly checkpointed rows and publish their resident handles."""
        paused: list[int] = []
        for index, slot in enumerate(self.slots):
            row = slot.row
            if (
                row is None
                or row.finished
                or slot.paused
                or (
                    not slot.guard_checkpoint_pending
                    and (
                        slot.checkpoint_emitted
                        or row.last_token not in slot.checkpoint_tokens
                    )
                )
            ):
                continue
            slot.checkpoint_emitted = True
            if row.last_token in slot.checkpoint_tokens:
                row.temporal_floor = max(
                    row.temporal_floor,
                    row.checkpoint_floor,
                )
                self.temporal_floors[index] = torch.maximum(
                    self.temporal_floors[index],
                    self.temporal_checkpoint_floors[index],
                )
            handle = self._pause_slot(index, GenerationPhase.PAUSED_PRIMARY)
            checkpointed(
                row.request,
                GenerationResult(
                    tuple(row.tokens),
                    False,
                    batch_steps=row.steps,
                    resident_handle=handle,
                    guard_action=row.guard_action,
                    guard_findings=self._guard_findings(row),
                    guard_summary=self._guard_summary(row),
                ),
            )
            slot.guard_checkpoint_pending = False
            paused.append(index)
        return tuple(paused)

    def admit_prepared(self, row: PreparedGenerationRow) -> bool:
        """Install an already-prefilled row into a compatible inactive slot."""
        slot = self._free_slot_for(row)
        if slot is None:
            return False
        self.install(slot, row)
        return True

    def admit_prepared_many(
        self,
        rows: tuple[PreparedGenerationRow, ...],
    ) -> bool:
        """Install one prepared replacement group without partial admission."""
        if len(rows) > self.available_count:
            return False
        if any(
            row.request.session_compatibility_key != self.compatibility_key
            or row.request.max_gen_len > max(slot.capacity for slot in self.slots)
            for row in rows
        ):
            return False
        for row in rows:
            if not self.admit_prepared(row):
                raise RuntimeError("validated prepared row could not be admitted")
        return True

    @staticmethod
    def _release_refused(rows: tuple[PreparedGenerationRow, ...]) -> None:
        """Give back the prefill of rows this session declined to install.

        A refusal happens *after* the prefill has run, so these rows hold real
        pages. The caller drops them either way; without this the pages go with
        them.
        """
        for row in rows:
            row.release_prefill()

    def resize_width(
        self,
        width: int,
        *,
        on_released: Callable[[int], object] | None = None,
    ) -> None:
        """Rebuild this session's fixed-width state at a new row count.

        Legal ONLY with every slot empty. Same rule as `KVBlockPool.resize`, same reason: everything rebuilt here is what CUDA Graphs pinned addresses to, and a row still holding one can't be moved.
        Caller preempts first; rows come back through `restore_preempted`, which re-runs their prefill and replays their tokens, so narrowing costs replay, not output.

        Session identity survives. `session_id` keys every handle the scheduler holds, so rebuilding in place keeps outstanding checkpoints addressable where close+reopen would strand them.

        `issued` carries across at the MAX over every old slot, not per slot -- a slot index that vanishes at a narrower width and returns at a wider one would restart its mint and re-issue a generation. R9, exactly.

        `on_released` runs between dropping the old state and building the new, for exactly one caller: a lane growing wider needs its KV supply to grow first, and `KVBlockPool.resize` is legal only while nothing borrows it.
        The old arena borrows it until dropped, the new one as soon as built => this is the only moment the supply can move.
        A narrowing lane passes nothing -- a supply larger than the arena costs nothing until another lane wants the bytes.

        It returns the width it could FUND, and that width wins. A supply is sized against a live device reading, so it can come back smaller than asked;
        building at the requested width anyway is `KVPagesExhausted` a few rows later with the arena already committed. `None` = no opinion, what a narrowing lane gives.
        """
        if width < 1:
            raise ValueError("a session needs at least one row")
        if any(slot.row is not None for slot in self.slots):
            raise ValueError("cannot resize a session that still holds rows")
        if width == self.width:
            return
        lm = self.lm
        issued = max(slot.issued for slot in self.slots)
        capacity = self.slots[0].capacity
        cfg_width = 2 if self.cfg_enabled else 1
        # Everything the old state owns goes back **before** the new width is
        # decided. The old arena borrows the KV supply, so it must be gone for
        # `on_released` to re-size it -- and the answer that comes back may be
        # smaller than what was asked for, which has to happen before a single
        # fixed-width tensor is built at the wrong count.
        self.model_state = None
        self._state_rows.clear()
        self._active_rows.clear()
        self._cuda_graph_runtime = None
        if on_released is not None:
            funded = on_released(width)
            if funded is not None and int(funded) >= 1:
                width = min(width, int(funded))
        self.width = width
        self.sampling_mask = torch.zeros(
            width, device=self.device, dtype=torch.bool
        )
        self.sampling_seeds = torch.zeros(
            width, device=self.device, dtype=torch.long
        )
        self.sampling_positions = torch.ones(
            width, device=self.device, dtype=torch.long
        )
        self.temporal_floors = torch.full(
            (width,), -1, device=self.device, dtype=torch.long
        )
        self.temporal_checkpoint_floors = torch.full(
            (width,), -1, device=self.device, dtype=torch.long
        )
        self.sequence = torch.full(
            (width, 1), lm.initial_token_id, device=self.device, dtype=torch.long
        )
        self.forbidden = torch.zeros(
            (width, lm.card), device=self.device, dtype=torch.bool
        )
        self.conditions = {
            name: (
                condition.new_zeros((width * cfg_width, *condition.shape[1:])),
                mask.new_zeros((width * cfg_width, *mask.shape[1:])),
            )
            for name, (condition, mask) in self.conditions.items()
        }
        # **The point of the whole operation.** Everything above is bookkeeping
        # the row count has to agree with; this is the memory. A narrower state
        # is a smaller KV arena, which is what the other lane is waiting for.
        #
        # Addresses move here, so every cached row selection and every captured
        # graph named memory this session no longer owns -- all of it was
        # dropped above, before the supply was re-sized.
        self.model_state = init_states(
            lm,
            batch_size=width * cfg_width,
            sequence_length=capacity,
        )
        bind_continuous_decode_metadata(self.model_state)
        self.slots = [_Slot(capacity) for _ in range(width)]
        for slot in self.slots:
            slot.issued = issued
        self._cuda_graph_runtime = self._create_cuda_graph_runtime()

    def admit(self, request: GenerationRequest) -> bool:
        """Prefill and admit one request when this session has capacity."""
        if request.session_compatibility_key != self.compatibility_key:
            return False
        if not any(
            slot.row is None and request.max_gen_len <= slot.capacity
            for slot in self.slots
        ):
            return False
        row = prepare_generation_row(self.lm, request)
        if self.admit_prepared(row):
            return True
        self._release_refused((row,))
        return False

    def prepare_conditions(
        self,
        requests: tuple[GenerationRequest, ...],
    ) -> PreparedConditionBatch:
        """Run the condition phase without creating private KV state yet."""
        if len(requests) > self.available_count:
            raise ValueError("condition batch exceeds available session slots")
        return self.prepare_condition_cohort(requests)

    def prepare_condition_cohort(
        self,
        requests: tuple[GenerationRequest, ...],
    ) -> PreparedConditionBatch:
        """Encode rows that will return to separate compatible sessions."""
        batch = prepare_generation_conditions(
            self.lm,
            requests,
            phase_recorder=self._phase_recorder,
        )
        self._condition_batches_by_width[len(requests)] += 1
        return batch

    def split_condition_cohort(
        self,
        batch: PreparedConditionBatch,
        sizes: tuple[int, ...],
    ) -> tuple[PreparedConditionBatch, ...]:
        """Split an encoded cohort back into ordered session-owned views.

        Exposed on the session so a caller that already holds one never has to
        import this module to undo `prepare_condition_cohort`.
        """
        return split_prepared_condition_batch(batch, sizes)

    def admit_conditioned_many(self, batch: PreparedConditionBatch) -> bool:
        """Consume a completed condition batch through the KV-prefill phase."""
        requests = batch.requests
        if len(requests) > self.available_count:
            return False
        with self._phase_recorder.measure("prefill"):
            prepared = prepare_generation_rows(
                self.lm,
                requests,
                phase_recorder=self._phase_recorder,
                _conditions=batch.conditions,
                _record_first_token_batch=(
                    lambda width: self._first_token_batches_by_width.__setitem__(
                        width,
                        self._first_token_batches_by_width[width] + 1,
                    )
                ),
                _record_packed_prefill_batch=(
                    lambda width: self._packed_prefill_batches_by_width.__setitem__(
                        width,
                        self._packed_prefill_batches_by_width[width] + 1,
                    )
                ),
            )
        with self._phase_recorder.measure("hot_admit"):
            admitted = self.admit_prepared_many(prepared)
        if not admitted:
            self._release_refused(prepared)
            return False
        self._record_prefill_widths(requests)
        return True

    def admit_many(self, requests: tuple[GenerationRequest, ...]) -> bool:
        """Batch-prefill and install compatible requests as one group."""
        if len(requests) > self.available_count:
            return False
        if any(
            request.session_compatibility_key != self.compatibility_key
            or request.max_gen_len > max(slot.capacity for slot in self.slots)
            for request in requests
        ):
            return False
        return self.admit_conditioned_many(self.prepare_conditions(requests))

    def _record_prefill_widths(
        self,
        requests: tuple[GenerationRequest, ...],
    ) -> None:
        prompt_lengths = {len(request.prompt_ids) for request in requests}
        if len(prompt_lengths) == 1 or (
            getattr(self.lm, "packed_prefill_enabled", False)
            and all(
                len(request.prompt_ids) < request.max_gen_len
                for request in requests
            )
        ):
            self._prefill_batches_by_width[len(requests)] += 1
        else:
            self._prefill_batches_by_width[1] += len(requests)

    def drain_prefill_batches_by_width(self) -> dict[int, int]:
        """Return and reset completed private-prefill batch counts."""
        result = dict(self._prefill_batches_by_width)
        self._prefill_batches_by_width.clear()
        return result

    def drain_preemption_telemetry(self) -> dict[str, int]:
        """Return and reset what preemption has cost since the last read.

        Six numbers, because frequency alone answers nothing:
        * `preemptions` -- rows displaced
        * `preempted_tokens` -- what their rebuilds owed when displaced
        * `restores` / `replayed_tokens` -- what the rebuilds actually cost
        * `preemption_wait_quanta` -- turns of everyone else's work that passed while a row was absent. The starvation bound nothing currently proves.
        * `preemption_refusals` -- times this session held no page to give, so the shortage went back to the caller as real exhaustion
        * `rebuild_token_divergences` -- rebuilds whose own prefill sampled a different token than the row originally produced.
          Since the fixed condition padding (2026-08-29) the prepend no longer shifts on a lone rebuild, so only fp16 width-reduction differences remain
          and this should sit at or near zero. A sustained non-zero rate is a signal again -- it used to be fatal (R10).

        Drained, not read: `scheduler_observations` are summed across jobs, so an absolute total gets multiplied by however many jobs looked.
        A monotone counter read as a before/after delta says the same only where the reader brackets EVERY producer -- and preemption's largest producer is the admission path, not decode.
        It was bracketed at decode alone and a complete run recorded 66 preemptions as zero. Draining can't lose them: whoever asks gets everything not yet accounted for, wherever incurred.
        """
        result = dict(self._preemptions)
        self._preemptions.clear()
        return result

    def drain_condition_batches_by_width(self) -> dict[int, int]:
        result = dict(self._condition_batches_by_width)
        self._condition_batches_by_width.clear()
        return result

    def drain_first_token_batches_by_width(self) -> dict[int, int]:
        result = dict(self._first_token_batches_by_width)
        self._first_token_batches_by_width.clear()
        return result

    def drain_packed_prefill_batches_by_width(self) -> dict[int, int]:
        result = dict(self._packed_prefill_batches_by_width)
        self._packed_prefill_batches_by_width.clear()
        return result

    def drain_phase_gpu_ms(self) -> dict[str, float]:
        """Return completed CUDA Event timings without synchronizing the device."""
        return self._phase_recorder.drain()

    def cuda_graph_telemetry(self) -> dict[str, int]:
        """Return graph counters for the scheduler's persistent-run record."""
        if self._cuda_graph_runtime is None:
            return {}
        telemetry = self._cuda_graph_runtime.dispatcher.telemetry
        return {
            "attempts": telemetry.attempts,
            "captures": telemetry.captures,
            "replays": telemetry.replays,
            # Refusals that *raised*, not silent degradation -- `_reject`
            # reports to the caller rather than switching to eager. Every one
            # of its reasons is unreachable by construction today, so a
            # non-zero value here is a broken invariant, not a slow path.
            "graph_refusals": telemetry.graph_refusals,
            "evictions": telemetry.evictions,
        }

    def cuda_graph_variant_telemetry(self) -> dict[str, dict[str, int]]:
        """Return per-variant cache outcomes for churn diagnosis."""
        if self._cuda_graph_runtime is None:
            return {}
        return self._cuda_graph_runtime.dispatcher.variant_telemetry()

    def prefill_graph_telemetry(self) -> dict[str, int]:
        """Return captured-prefill counters for the run record.

        Cumulative per loaded MODEL, not per session -- the cache outlives a session because its entries own the addresses their graphs baked, so read these as gauges, not differenced like the decode counters beside them.

        Reported because nothing else can see this path: it needs CUDA AND a declared KV pool, and the test environment has neither, so the whole suite passes with every prefill refused.
        A cache that never hits and one that always does are the same observation without these numbers.
        """
        cache = getattr(self.lm, "_prefill_graph_cache", None)
        if cache is None:
            return {}
        telemetry = cache.telemetry
        return {
            "captures": telemetry.captures,
            "replays": telemetry.replays,
            "hits": telemetry.hits,
            "misses": telemetry.misses,
            "entries": telemetry.entries,
            "refusals": sum(telemetry.refusals.values()),
        }

    def prefill_graph_refusals(self) -> dict[str, int]:
        """Why prefills were served eagerly, by reason."""
        cache = getattr(self.lm, "_prefill_graph_cache", None)
        return {} if cache is None else dict(cache.telemetry.refusals)

    def cuda_graph_cache_telemetry(self) -> dict[str, int]:
        """Return current cache occupancy and retained static-tensor bytes."""
        if self._cuda_graph_runtime is None:
            return {}
        return self._cuda_graph_runtime.dispatcher.cache_telemetry()

    def capacity_telemetry(self) -> dict[str, int]:
        """Report owned tensor bytes without synchronizing the accelerator."""
        resident_bytes = _unique_tensor_storage_bytes(
            (
                self.model_state,
                self.conditions,
                self.forbidden,
                self.sequence,
                self.sampling_mask,
                self.sampling_seeds,
                self.sampling_positions,
                self.temporal_shift_values,
                self.temporal_floors,
                self.temporal_checkpoint_floors,
            )
        )
        cache = self.cuda_graph_cache_telemetry()
        graph_static_bytes = int(cache.get("static_bytes", 0))
        return {
            "physical_width": self.width,
            "resident_bytes": resident_bytes,
            "graph_static_bytes": graph_static_bytes,
            # Reported beside the owned totals, never folded into them: the
            # private pool belongs to the graph cache, not to any row.
            "graph_pool_bytes": int(cache.get("pool_bytes", 0)),
            "total_bytes": resident_bytes + graph_static_bytes,
            "estimated_row_bytes": (resident_bytes + self.width - 1) // self.width,
        }

    def drain_cuda_graph_capture_gpu_ms(self) -> float:
        """Return completed CUDA Graph capture time without synchronizing."""
        if self._cuda_graph_runtime is None:
            return 0.0
        return self._cuda_graph_runtime.drain_capture_gpu_ms()

    def collect_finished(
        self,
        completed: Callable[[GenerationRequest, GenerationResult], None],
        released: Callable[[int, int], None] | None = None,
    ) -> tuple[int, ...]:
        """Publish finished rows and return their newly inactive slots."""
        inactive: list[int] = []
        for index, slot in enumerate(self.slots):
            row = slot.row
            if row is None or not row.finished:
                continue
            row.observe_guard_completion()
            completed(
                row.request,
                GenerationResult(
                    tuple(row.tokens),
                    row.emitted_eos,
                    batch_steps=row.steps,
                    guard_action=row.guard_action,
                    guard_findings=self._guard_findings(row),
                    guard_summary=self._guard_summary(row),
                ),
            )
            slot.row = None
            # A finished row's pages come back here, not at the next install.
            # Waiting for an install means a slot nothing replaces holds its
            # KV until the session closes -- which is most of A3-3c's reclaim,
            # and it is also what leaves a preempted row unable to return with
            # every other row already done.
            state_rows, state_rows_host = self._active_state_rows((index,))
            reset_state_rows(
                self.lm, self.model_state, state_rows, host_rows=state_rows_host
            )
            self.temporal_floors[index] = -1
            self.temporal_checkpoint_floors[index] = -1
            slot.paused = False
            slot.phase = GenerationPhase.CONTINUE
            slot.checkpoint_tokens = frozenset()
            slot.checkpoint_emitted = False
            slot.pending_tokens = ()
            slot.guard_checkpoint_pending = False
            inactive.append(index)
            if released is not None:
                released(index, slot.capacity)
        return tuple(inactive)

    def _active_state_rows(
        self,
        active: tuple[int, ...],
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        cached = self._state_rows.get(active)
        if cached is not None:
            return cached
        host_rows = active + (
            tuple(index + self.width for index in active)
            if self.cfg_enabled
            else ()
        )
        cached = (
            torch.tensor(host_rows, dtype=torch.long, device=self.device),
            host_rows,
        )
        self._state_rows[active] = cached
        return cached

    def _active_rows_tensor(self, active: tuple[int, ...]) -> torch.Tensor:
        """`active` on the device, cached on the tuple.

        Three sites built this by hand -- the graph launch, the graph consume and the eager launch -- so one quantum paid three H2D copies for one unchanging index set.
        Read-only at every use: nothing writes into the returned tensor.
        """
        if not self.cfg_enabled:
            # No mirror rows => `_active_state_rows` builds exactly this tuple.
            return self._active_state_rows(active)[0]
        cached = self._active_rows.get(active)
        if cached is None:
            cached = torch.tensor(active, dtype=torch.long, device=self.device)
            self._active_rows[active] = cached
        return cached

    def _graph_state_rows_for(
        self,
        state_rows: torch.Tensor,
        state_rows_host: tuple[int, ...],
    ) -> torch.Tensor:
        """Return a stable-address row map for one compact active width."""
        width = state_rows.shape[0]
        graph_rows = self._graph_state_rows.get(width)
        if graph_rows is None:
            graph_rows = torch.empty_like(state_rows)
            self._graph_state_rows[width] = graph_rows
            self._graph_state_row_values[width] = ()
        values = state_rows_host
        if self._graph_state_row_values[width] != values:
            graph_rows.copy_(state_rows)
            self._graph_state_row_values[width] = values
        return graph_rows

    def _try_cuda_graph_quantum(
        self,
        active: tuple[int, ...],
        *,
        quantum: int,
    ) -> torch.Tensor | None:
        runtime = self._cuda_graph_runtime
        if runtime is None:
            return None
        state_rows, state_rows_host = self._active_state_rows(active)
        # A row may have fewer than a full q4/q8 quantum left in its allocated
        # KV cache. The ordinary per-token path below is the correct tail path;
        # do not turn this valid boundary into a Graph-runtime failure.
        if not runtime.can_replay(state_rows_host, quantum):
            return None
        graph_state_rows = self._graph_state_rows_for(
            state_rows,
            state_rows_host,
        )
        select_state_rows(
            self.model_state,
            graph_state_rows,
            host_rows=state_rows_host,
        )
        active_rows = self._active_rows_tensor(active)
        # The only phase in this file whose stream-elapsed time IS device time.
        # Everywhere else a `_CudaPhaseRecorder` pair brackets a region the host participates in, so the stream drains between launches and the reading is the region's wall clock over again -- same interval, measured twice.
        # A graph replay needs no host participation once enqueued, so between these two events the stream runs the graph and nothing else.
        # `decode_replay / decode` is therefore the device fraction of a decode quantum, and the only honest device-time figure available: `torch.profiler` with CUDA activities fails inside a replay, and `nvidia-smi` samples far coarser.
        with self._phase_recorder.measure("decode_replay"):
            graph_tokens = runtime.replay(
                self.sequence.index_select(0, active_rows),
                self.forbidden.index_select(0, active_rows),
                active_rows=active_rows,
                state_rows=graph_state_rows,
                state_rows_host=state_rows_host,
                quantum=quantum,
                cfg_coef=self.cfg_coef,
                temperature=self.temperature,
                sampling_mask=self.sampling_mask.index_select(0, active_rows),
                sampling_seeds=self.sampling_seeds.index_select(0, active_rows),
                sampling_positions=self.sampling_positions.index_select(
                    0, active_rows
                ),
                temporal_shift_values=self.temporal_shift_values,
                temporal_floors=self.temporal_floors.index_select(0, active_rows),
            )
        sampled_shifts = self.temporal_shift_values[graph_tokens]
        self.temporal_floors[active_rows] = torch.maximum(
            self.temporal_floors.index_select(0, active_rows),
            sampled_shifts.amax(dim=0),
        )
        self.sampling_positions[active_rows] += (
            self.sampling_mask.index_select(0, active_rows).long() * quantum
        )
        return graph_tokens

    def _consume_graph_quantum(
        self,
        active: tuple[int, ...],
        graph_tokens: torch.Tensor,
        completed: Callable[[GenerationRequest, GenerationResult], None],
        released: Callable[[int, int], None] | None,
        checkpointed: Callable[[GenerationRequest, GenerationResult], None]
        | None,
    ) -> tuple[int, int, dict[int, int]]:
        """Apply one graph block, retaining suffixes past checkpoints."""
        token_rows = graph_tokens.detach().cpu().tolist()
        # The stream just drained for the tokens, so this is a 4-byte read, and
        # it is the only place a *replayed* kernel's block-table fault can
        # surface -- replay never re-enters the Python wrapper.
        check_block_table_fault(self.device)
        # Margins the capture computed on device, one per step per row. Read
        # once here rather than per row: this is a device-to-host copy, and the
        # per-step `.item()` calls it replaces were two syncs per step.
        runtime = self._cuda_graph_runtime
        margin_rows = (
            None
            if runtime is None or runtime.last_margins is None
            else runtime.last_margins.detach().cpu().tolist()
        )
        traces = None if runtime is None else runtime.last_traces
        exports = getattr(self.lm, "_activation_export", None)
        wasted_token_rows = 0
        next_sequences = []
        for compact_index, index in enumerate(active):
            slot = self.slots[index]
            row = slot.row
            for step, token_row in enumerate(token_rows):
                if row is None or row.finished:
                    wasted_token_rows += len(token_rows) - step
                    break
                token = int(token_row[compact_index])
                collector = row.request.trace_collector
                if margin_rows is not None and collector is not None:
                    collector.observe(token, margin_rows[step][compact_index])
                export = (
                    None
                    if not exports
                    else exports.get(_export_key(row.request.trace_context))
                )
                if traces is not None and export is not None:
                    # One step of one row, already reduced on device. The slices
                    # keep the eager path's shapes -- `[1, ...]` per row -- so
                    # `observe` stores exactly what the eager path would have.
                    export.observe(
                        traces[0][step, compact_index : compact_index + 1],
                        traces[1][step, compact_index : compact_index + 1],
                        traces[2][step, compact_index : compact_index + 1],
                        traces[3][step, compact_index : compact_index + 1],
                        traces[4][step, compact_index : compact_index + 1],
                        graph_tokens[step, compact_index : compact_index + 1],
                        rows=1,
                        context=row.request.trace_context or (),
                    )

                if not row.observe_temporal_token(token):
                    wasted_token_rows += len(token_rows) - step - 1
                    break
                row.last_token = token
                row.steps += 1
                row.emitted_eos = token == self.eos_id
                if not row.emitted_eos:
                    row.tokens.append(token)
                    guard_action = self._observe_guard_token(slot, token)
                else:
                    guard_action = "continue"
                row.finished = row.finished or (
                    row.emitted_eos or row.steps >= row.request.max_gen_len
                )
                if row.finished:
                    wasted_token_rows += len(token_rows) - step - 1
                    break
                if (
                    checkpointed is not None
                    and (
                        (
                            not slot.checkpoint_emitted
                            and token in slot.checkpoint_tokens
                        )
                        or guard_action
                        in {"interrupt_to_recovery", "reject_candidate"}
                    )
                ):
                    slot.pending_tokens = tuple(
                        int(future_row[compact_index])
                        for future_row in token_rows[step + 1 :]
                    )
                    break
            next_sequences.append(row.last_token if row is not None else 0)
        active_rows = self._active_rows_tensor(active)
        self.sequence[active_rows, 0] = torch.tensor(
            next_sequences,
            device=self.device,
            dtype=torch.long,
        )
        if checkpointed is not None:
            self.collect_checkpoints(checkpointed)
        self.collect_finished(completed, released)
        return (
            graph_tokens.shape[0],
            wasted_token_rows,
            {len(active): int(graph_tokens.shape[0])},
        )

    @torch.inference_mode()
    def run_quantum(
        self,
        max_steps: int,
        completed: Callable[[GenerationRequest, GenerationResult], None],
        released: Callable[[int, int], None] | None = None,
        checkpointed: Callable[[GenerationRequest, GenerationResult], None]
        | None = None,
    ) -> ContinuousGenerationStats:
        """One synchronous quantum: launch and consume back to back.

        A convenience composition, not a second decode implementation -- the
        scheduler drives the two halves itself (S5) and this exists for
        callers that want one call per quantum (gates, benchmarks, tests).
        """
        self.collect_finished(completed, released)
        if checkpointed is not None:
            self.collect_checkpoints(checkpointed)
        if not self.launch_quantum(max_steps):
            return ContinuousGenerationStats(
                physical_steps=0,
                wasted_token_rows=0,
                replacements=0,
                active_steps_by_width={},
            )
        return self.consume_quantum(completed, released, checkpointed)

    # -- S5 host-async decode: launch a quantum now, consume its tokens later.
    # One protocol for both lines since 2026-08-30; which launch runs is
    # `self._decode_line`. Contract: every True from `launch_quantum` owes
    # exactly one `consume_quantum` before anything mutates the session's
    # slots -- consume asserts row identity and fails closed on a mutation
    # that slipped in.

    @property
    def quantum_in_flight(self) -> bool:
        return self._in_flight_quantum is not None

    @torch.inference_mode()
    def launch_quantum(self, max_steps: int) -> bool:
        """Launch one quantum without waiting for its tokens.

        Returns False when there are no active rows. Finished rows are NOT
        published here -- launch takes no callbacks, so callers run it only
        after a `consume_quantum` or `run_quantum` has drained completions.
        """
        if max_steps < 1:
            raise ValueError("quantum must be positive")
        if self._in_flight_quantum is not None:
            raise RuntimeError(
                "a quantum is already in flight; consume it before launching"
            )
        # After the finished rows have given their pages back (the caller's
        # consume), so a preempted row is measured against what is free.
        self._restore_preempted_rows()
        with self._phase_recorder.measure("decode"):
            while True:
                active = tuple(
                    index
                    for index, slot in enumerate(self.slots)
                    if (
                        slot.row is not None
                        and not slot.paused
                        and not slot.row.finished
                    )
                )
                if not active:
                    return False
                # Before the launch, not during: a graph replay writes every position of its quantum with no Python in it, so a page it will need must already be in the table; the eager line writes the same positions.
                # One batched update per quantum, no-op for every row that hasn't crossed a block boundary since the last.
                # PHYSICAL rows, not slots: a CFG-doubled state writes each row's twin at `slot + width`, and a twin nobody grew has no page under the position about to be written.
                # Nothing enables CFG on this path today (`cfg_coef` is 1.0 everywhere), which is exactly why it has to be right here rather than found later.
                _, grow_rows = self._active_state_rows(active)
                try:
                    grow_state_rows(self.model_state, grow_rows, ahead=max_steps)
                except KVPagesExhausted:
                    # Release rather than wait: a row that can't be given a page gives one up instead, which is what keeps this from being hold-and-wait.
                    # The victim is rebuilt exactly when there's room again -- see `restore_preempted`.
                    if not self._preempt_for_pages(active):
                        raise
                    continue
                tokens = self._decode_line.launch(
                    self, active, quantum=max_steps
                )
                self._quanta += 1
                self._in_flight_quantum = _InFlightQuantum(
                    active=active,
                    rows=tuple(id(self.slots[index].row) for index in active),
                    tokens=tokens,
                    quantum=max_steps,
                )
                return True

    @torch.inference_mode()
    def consume_quantum(
        self,
        completed: Callable[[GenerationRequest, GenerationResult], None],
        released: Callable[[int, int], None] | None = None,
        checkpointed: Callable[[GenerationRequest, GenerationResult], None]
        | None = None,
    ) -> ContinuousGenerationStats:
        """Synchronize on the in-flight replay and apply its tokens."""
        in_flight = self._in_flight_quantum
        if in_flight is None:
            raise RuntimeError("no quantum is in flight")
        self._in_flight_quantum = None
        for index, row_id in zip(in_flight.active, in_flight.rows, strict=True):
            row = self.slots[index].row
            if row is None or id(row) != row_id:
                raise RuntimeError(
                    f"slot {index} changed while a quantum was in flight; "
                    "every path that mutates rows must consume first"
                )
        with self._phase_recorder.measure("decode"):
            steps, wasted, widths = self._consume_graph_quantum(
                in_flight.active,
                in_flight.tokens,
                completed,
                released,
                checkpointed,
            )
        return ContinuousGenerationStats(
            physical_steps=steps,
            wasted_token_rows=wasted,
            replacements=0,
            active_steps_by_width=widths,
        )

    def _rows_allow_graph(self, active: tuple[int, ...]) -> bool:
        """Whether every active row can be fed by a replay.

        A row carrying a trace collector cannot: a replay runs no Python, so
        there is nothing to call the collector from.
        """
        return all(
            self.slots[index].row is not None
            and _collector_allows_graph(self.slots[index].row.request.trace_collector)
            for index in active
        )

def create_continuous_generation_session(
    model,
    requests: tuple[GenerationRequest, ...],
    *,
    width: int,
    capacity: int,
) -> ContinuousGenerationBatch:
    """Create one resumable session through a TranscriptionModel wrapper."""
    return ContinuousGenerationBatch(
        model._model,
        requests,
        width=width,
        capacity=capacity,
    )
