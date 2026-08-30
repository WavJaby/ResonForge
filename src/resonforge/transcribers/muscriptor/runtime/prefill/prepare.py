"""Turning a request and its conditions into a row decode can install.

Nothing here knows a session exists. The dependency is one-way -- `decode`
calls into this module, never the reverse -- and `PreparedGenerationRow` in
`runtime.rows` is the whole of what crosses.

The device memory this takes, and what it costs before it is taken, is
`prefill.memory`.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext

import torch
from muscriptor.generation_batch import GenerationRequest
from muscriptor.modules.streaming import ModelState, increment_steps

from resonforge.transcribers.muscriptor.runtime.generation_state import (
    slot_state_row_indices,
)
from resonforge.transcribers.muscriptor.runtime.instrumentation import (
    _CudaPhaseRecorder,
)
from resonforge.transcribers.muscriptor.runtime.prefill import (
    memory as prefill_memory,
)
from resonforge.transcribers.muscriptor.runtime.prefill.graphs import (
    PrefillGraphCache,
    PrefillGraphKey,
    _Buffers,
    bucket_for,
)
from resonforge.transcribers.muscriptor.runtime.rows import (
    PreparedConditionBatch,
    PreparedGenerationRow,
    _PrefillArena,
    condition_batch_key,
)


def _prefill_states(
    lm, *, batch_size: int, prepend_length: int, prompt_length: int
) -> ModelState:
    """Allocate a prefill state for exactly the positions prefill writes.

    Kept as a name because three call sites use it; the allocation, the cost
    arithmetic and the calibrated peak all live in `prefill.memory`, so
    "what does prefill cost" and "where does prefill allocate" have one answer
    each. `prefill.memory.demand_bytes` states that cost **before** anything is
    taken, which is the order a ledger needs and the opposite of the high-water
    reading the device reserve is sized from today.

    **It used to allocate `prepend_length + max_gen_len` and write
    `prepend_length + 1 + prompt_length`.** Prefill runs the condition prefix,
    the forced prompt and one sampled token; every position after that is
    written by *decode*, into the session's own arena, after the row has been
    copied out of this one. A fresh row carries no forced prompt at all, so on
    the transcription model the allocation was ~290 positions of use inside
    ~1,790 of allocation.

    The written extent is not an estimate: `increment_steps` advances the state
    by `sequence.shape[1] + prepend_length`, which is this same sum, and has
    done so at every call site all along. The allocation and the cursor now
    come from one arithmetic instead of two.

    `padded_length_for` adds the graph quantum of write headroom on top, so the
    `+ 1` here is the sampled token and nothing is rounding-sensitive.
    """
    return prefill_memory.allocate_state(
        lm,
        rows=batch_size,
        sequence_length=int(prepend_length) + 1 + int(prompt_length),
    )



def _forbidden_mask(lm, request: GenerationRequest) -> torch.Tensor:
    mask = torch.zeros((1, lm.card), device=lm.emb.weight.device, dtype=torch.bool)
    if request.forbidden_token_ids:
        mask[0, list(request.forbidden_token_ids)] = True
    return mask


def _sampling_metadata(
    requests: tuple[GenerationRequest, ...],
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Return per-row mode plus counter state for deterministic sampling."""
    mask = torch.tensor(
        [request.use_sampling for request in requests],
        device=device,
        dtype=torch.bool,
    )
    if not any(request.use_sampling for request in requests):
        return None, None, mask
    if any(
        request.use_sampling and request.sampling_seed is None
        for request in requests
    ):
        raise ValueError("continuous sampling rows require a stable seed")
    return (
        torch.tensor(
            [request.sampling_seed or 0 for request in requests],
            device=device,
            dtype=torch.long,
        ),
        torch.zeros(len(requests), device=device, dtype=torch.long),
        mask,
    )


def _temporal_metadata(
    requests: tuple[GenerationRequest, ...],
    device: torch.device,
    card: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one shared token map plus independent initial row floors."""
    configs = tuple(request.temporal_grammar for request in requests)
    configured = tuple(config for config in configs if config is not None)
    if not configured:
        return (
            torch.full((card,), -1, device=device, dtype=torch.long),
            torch.full((len(requests),), -1, device=device, dtype=torch.long),
        )
    shift_values = configured[0].shift_values
    if len(shift_values) > card:
        raise ValueError("temporal grammar vocabulary exceeds model card")
    if any(config.shift_values != shift_values for config in configured[1:]):
        raise ValueError("continuous rows require one temporal token mapping")
    return (
        torch.tensor(
            (*shift_values, *((-1,) * (card - len(shift_values)))),
            device=device,
            dtype=torch.long,
        ),
        torch.tensor(
            [
                -1 if config is None else config.initial_floor
                for config in configs
            ],
            device=device,
            dtype=torch.long,
        ),
    )


def _condition_rows(
    source: torch.Tensor,
    row_indices: tuple[int, ...],
) -> torch.Tensor:
    """Take one prefill row's slice of a whole-batch condition tensor.

    A single index is a contiguous slice, so this is a view and issues no
    kernel; the CFG pair is not contiguous and is concatenated, which is what
    `split_prepared_condition_batch` does with the same tensors.
    """
    if len(row_indices) == 1:
        return source.narrow(0, row_indices[0], 1)
    return torch.cat(
        [source.narrow(0, index, 1) for index in row_indices],
        dim=0,
    )


def _rows_from_first_token(
    *,
    lm,
    requests,
    prompts,
    next_tokens: torch.Tensor,
    temporal_floors: torch.Tensor,
    conditions,
    state,
    arena,
    batch_size: int,
    cfg_enabled: bool,
    decode_span,
) -> tuple[PreparedGenerationRow, ...]:
    """Turn one prefill batch's first tokens into prepared rows.

    Both prefill paths -- ragged and conditioned -- ended a batch this exact
    way, in two byte-identical loops that had to be kept in agreement by hand.

    The two host reads happen ONCE for the batch, not once per row: they are
    per-row scalars off whole-batch tensors, so the per-row form paid 2N syncs
    for what 2 buy. `phase_occupancy.py --host-reads` measured them as the only
    device reads the prefill phase makes.
    """
    next_tokens_host = next_tokens.tolist()
    temporal_floors_host = temporal_floors.tolist()
    prepared = []
    for row, (request, prompt) in enumerate(zip(requests, prompts, strict=True)):
        # This row is logical slot `row` of a width-`batch_size` prefill, and
        # `copy_prepared_slot_state` reads it there when the row is admitted.
        # Allocating a private per-row state and copying into it doubled the
        # prefill KV footprint -- a second full allocation outside the block
        # pool -- to produce a buffer whose only job was to be copied a second
        # time. `slot_state_row_indices` gives the same (row, row + batch_size)
        # pair the hand-built tensors did, from the one definition of that map.
        #
        # HOST indices, and `narrow` rather than a gather: the fancy-index form
        # built one index tensor per row (an H2D copy each) and issued two
        # gathers per condition per row. A single row is one contiguous slice,
        # so it is a view and costs no kernel at all; only the CFG pair needs a
        # copy, and it is assembled the same way `split_prepared_condition_batch`
        # already assembles it. Rows of one batch are installed and released
        # together (`admit_prepared_many` -> `install` -> `release_prefill`, or
        # `_release_refused`), so a view holds nothing alive past its copy would.
        row_indices = slot_state_row_indices(
            row,
            batch_size,
            cfg_enabled=cfg_enabled,
        )
        private_conditions = {
            name: (
                _condition_rows(condition, row_indices),
                _condition_rows(mask, row_indices),
            )
            for name, (condition, mask) in conditions.items()
        }
        next_token = int(next_tokens_host[row])
        emitted_eos = next_token == request.eos_id
        if not emitted_eos:
            prompt.append(next_token)
        steps = len(request.prompt_ids) + 1
        temporal_floor = _advanced_temporal_floor(
            request,
            int(temporal_floors_host[row]),
            next_token,
        )
        prepared.append(
            PreparedGenerationRow(
                request=request,
                model_state=state,
                prefill_arena=arena,
                source_slot=row,
                source_width=batch_size,
                conditions=private_conditions,
                tokens=prompt,
                last_token=next_token,
                steps=steps,
                finished=emitted_eos or steps >= request.max_gen_len,
                emitted_eos=emitted_eos,
                decode_span=decode_span,
                temporal_floor=temporal_floor,
                checkpoint_floor=(
                    -1
                    if request.temporal_grammar is None
                    or request.temporal_grammar.checkpoint_floor is None
                    else request.temporal_grammar.checkpoint_floor
                ),
            )
        )
    return tuple(prepared)


def _advanced_temporal_floor(
    request: GenerationRequest,
    floor: int,
    token: int,
) -> int:
    grammar = request.temporal_grammar
    if grammar is None or not 0 <= token < len(grammar.shift_values):
        return floor
    shift = grammar.shift_values[token]
    if shift < 0:
        return floor
    if shift < floor:
        raise RuntimeError(
            "temporal grammar emitted a decreasing shift: "
            f"{shift} < {floor}"
        )
    return max(floor, shift)


@torch.inference_mode()
def prepare_generation_row(
    lm,
    request: GenerationRequest,
    *,
    phase_recorder: _CudaPhaseRecorder | None = None,
) -> PreparedGenerationRow:
    """Prefill one request privately and generate its first unforced token."""
    if request.beam_size != 1:
        raise ValueError("continuous generation requires beam_size=1")
    with (
        phase_recorder.measure("condition_encode")
        if phase_recorder is not None
        else nullcontext()
    ):
        conditions = lm.prepare_generation_conditions(
            [request.condition], request.cfg_coef
        )
    return _prepare_generation_row_with_conditions(
        lm,
        request,
        conditions,
        phase_recorder=phase_recorder,
    )


@torch.inference_mode()
def prepare_generation_conditions(
    lm,
    requests: tuple[GenerationRequest, ...],
    *,
    phase_recorder: _CudaPhaseRecorder | None = None,
) -> PreparedConditionBatch:
    """Batch only the model conditioning work for later KV prefill."""
    if not requests:
        raise ValueError("condition batch requires at least one request")
    key = condition_batch_key(requests[0])
    if any(condition_batch_key(request) != key for request in requests):
        raise ValueError("condition generation rows are incompatible")
    measure = (
        phase_recorder.measure("condition_encode")
        if phase_recorder is not None
        else nullcontext()
    )
    with measure:
        conditions = lm.prepare_generation_conditions(
            [request.condition for request in requests], requests[0].cfg_coef
        )
    return PreparedConditionBatch(requests=requests, conditions=conditions)


def _prepare_generation_rows_ragged(
    lm,
    requests: tuple[GenerationRequest, ...],
    conditions: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    phase_recorder: _CudaPhaseRecorder | None = None,
    record_first_token_batch: Callable[[int], None] | None = None,
    record_packed_prefill_batch: Callable[[int], None] | None = None,
) -> tuple[PreparedGenerationRow, ...]:
    """Prefill mixed prompt lengths through the opt-in packed reference path."""
    batch_size = len(requests)
    cfg_enabled = requests[0].cfg_coef != 1.0
    cfg_width = 2 if cfg_enabled else 1
    prepend_length = sum(cond.shape[1] for cond, _ in conditions.values())
    decode_span = prepend_length + max(
        request.max_gen_len for request in requests
    )
    # Before the allocation, because the allocation is sized from them now.
    prompts = [
        list(request.prompt_ids[: request.max_gen_len]) for request in requests
    ]
    state = _prefill_states(
        lm,
        batch_size=batch_size * cfg_width,
        prepend_length=prepend_length,
        prompt_length=max(len(prompt) for prompt in prompts),
    )
    arena = _PrefillArena(state, len(requests))
    token_rows = tuple(
        torch.tensor(
            [lm.initial_token_id, *prompt],
            device=lm.emb.weight.device,
            dtype=torch.long,
        )
        for prompt in prompts
    )
    forbidden = torch.zeros(
        (batch_size, lm.card),
        device=lm.emb.weight.device,
        dtype=torch.bool,
    )
    for row, request in enumerate(requests):
        if request.forbidden_token_ids:
            forbidden[row, list(request.forbidden_token_ids)] = True
    if record_first_token_batch is not None:
        record_first_token_batch(batch_size)
    if record_packed_prefill_batch is not None:
        record_packed_prefill_batch(batch_size)
    first_token_measure = (
        phase_recorder.measure("first_token")
        if phase_recorder is not None
        else nullcontext()
    )
    with first_token_measure, lm.autocast:
        sampling_seeds, sampling_positions, sampling_mask = _sampling_metadata(
            requests,
            lm.emb.weight.device,
        )
        temporal_shift_values, temporal_floors = _temporal_metadata(
            requests,
            lm.emb.weight.device,
            lm.card,
        )
        next_tokens = lm._sample_next_token_ragged(
            token_rows,
            conditions,
            state,
            use_sampling=any(request.use_sampling for request in requests),
            temp=requests[0].temperature,
            top_k=0,
            top_p=0.0,
            cfg_coef=requests[0].cfg_coef,
            forbidden_tokens=forbidden,
            temporal_shift_values=temporal_shift_values,
            temporal_floors=temporal_floors,
            sample_mask=sampling_mask,
            generator=None,
            sampling_seeds=sampling_seeds,
            sampling_positions=sampling_positions,
            trace_collectors=tuple(request.trace_collector for request in requests),
            trace_contexts=tuple(request.trace_context for request in requests),
        )

    return _rows_from_first_token(
        lm=lm,
        requests=requests,
        prompts=prompts,
        next_tokens=next_tokens,
        temporal_floors=temporal_floors,
        conditions=conditions,
        state=state,
        arena=arena,
        batch_size=batch_size,
        cfg_enabled=cfg_enabled,
        decode_span=decode_span,
    )


#: Where `--muscriptor-prefill-runtime` lands on a loaded model. Stamped by
#: `_configure_loaded_model`, next to the decode line's
#: `configure_continuous_graph_decode`, because this is the same object the
#: prefill path is handed and a per-model attribute is what the graph cache
#: already is.
PREFILL_GRAPHS_ATTRIBUTE = "_resonforge_prefill_graphs"


def configure_prefill_graphs(lm, enabled: bool) -> None:
    """Apply one resolved prefill line to a loaded model."""
    setattr(lm, PREFILL_GRAPHS_ATTRIBUTE, bool(enabled))


def prefill_graphs_enabled(lm) -> bool:
    """Whether captured prefill is on for this model.

    **An unstamped model is eager, and the refusal is counted rather than
    silent.** The alternative -- defaulting to on -- means a plumbing break
    that never reaches the model leaves `--muscriptor-prefill-runtime eager`
    capturing graphs, and an A/B that measures the wrong arm is worse than one
    that measures nothing. Standalone MuScriptor and the CPU tests live here
    and would refuse on `not-pooled` regardless.

    The eager path itself is not a mode and stays whatever the flag says:
    `PrefillGraphCache.refuse` counts four ordinary reasons a prefill is served
    eagerly, and every one of them has a non-zero counter on a real run.
    """
    return bool(getattr(lm, PREFILL_GRAPHS_ATTRIBUTE, False))


def prefill_graph_cache(lm) -> PrefillGraphCache:
    """The captured-prefill cache for this loaded model, created on demand.

    Per model rather than per session: the entries own `ModelState`s addressing
    that model's KV pool, and a session is a much shorter thing than a loaded
    model. Attached to `lm` the way `_activation_export` is, for the same
    reason -- there is one of these per model and threading it through six
    construction sites would be a seventh list to keep in step.
    """
    cache = getattr(lm, "_prefill_graph_cache", None)
    if cache is None:
        cache = PrefillGraphCache()
        lm._prefill_graph_cache = cache
    return cache


def _reset_prefill_cursors(state) -> None:
    """Put a reused state back at position zero, in place.

    In place because the transformer's `offsets` tensor is one of the addresses
    a captured graph baked; rebinding it would leave the graph reading a tensor
    nobody writes.
    """
    for layer in state.values():
        if "offset" in layer:
            layer["offset"] = 0
        offsets = layer.get("offsets")
        if offsets is not None:
            offsets.zero_()


def _graph_prefill_row(
    lm,
    request: GenerationRequest,
    conditions: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    prepend_length: int,
    prompt: list[int],
    decode_span: int,
    phase_recorder: _CudaPhaseRecorder | None,
) -> PreparedGenerationRow | None:
    """Prefill through a captured graph, or return None and leave it eager.

    Returning None is ordinary, not a failure: a cold bucket, a traced request
    and a full cache all take the eager path, which is correct and always
    available. Every one of them is counted -- see `PrefillGraphCache.refuse`.
    """
    if not prefill_graphs_enabled(lm):
        prefill_graph_cache(lm).refuse("eager-line")
        return None
    cache = prefill_graph_cache(lm)
    device = lm.emb.weight.device
    if device.type != "cuda":
        cache.refuse("not-cuda")
        return None
    if request.trace_collector is not None or getattr(lm, "_activation_export", None):
        # Both do per-call host work -- appending to a collector, reducing and
        # storing per row -- and a replay runs no Python at all.
        cache.refuse("traced")
        return None
    cfg_enabled = request.cfg_coef != 1.0
    rows = 2 if cfg_enabled else 1
    tokens = 1 + len(prompt)
    positions = bucket_for(prepend_length + tokens)
    key = PrefillGraphKey(
        rows=rows,
        positions=positions,
        prepend_length=prepend_length,
        cfg_coef=float(request.cfg_coef),
        temperature=float(request.temperature),
        sampling=bool(request.use_sampling),
    )
    entry = cache.free_entry(key)
    if entry is None and not cache.has_room():
        cache.refuse("graph-cache-full")
        return None

    padded_tokens = positions - prepend_length
    # One **logical** row, whatever `rows` says: `_compute_logits` doubles the
    # sequence itself under CFG, and the state's `rows` counts the physical
    # halves. Handing it the doubled shape makes the transformer see twice the
    # rows its offsets table has.
    sequence = torch.full(
        (1, padded_tokens), lm.zero_token_id, device=device, dtype=torch.long
    )
    # **Padding sits behind the real tokens and is embedded as zero.** Causal
    # attention means the true final position cannot see it, and `last_index`
    # is what stops the sample being taken from it.
    sequence[0, :tokens] = torch.tensor(
        [lm.initial_token_id, *prompt], device=device, dtype=torch.long
    )
    seeds, positions_counter, sample_mask = _sampling_metadata((request,), device)
    shift_values, floors = _temporal_metadata((request,), device, lm.card)
    forbidden = _forbidden_mask(lm, request)
    last_index = torch.tensor([tokens - 1], device=device, dtype=torch.long)

    if entry is None:
        state = _prefill_states(
            lm,
            batch_size=rows,
            prepend_length=prepend_length,
            prompt_length=padded_tokens - 1,
        )
        buffers = _Buffers(
            sequence=sequence.clone(),
            forbidden=forbidden.clone(),
            temporal_shift_values=shift_values.clone(),
            temporal_floors=floors.clone(),
            sample_mask=sample_mask.clone(),
            last_index=last_index.clone(),
            conditions={
                name: (cond.clone(), mask.clone())
                for name, (cond, mask) in conditions.items()
            },
            sampling_seeds=None if seeds is None else seeds.clone(),
            sampling_positions=(
                None if positions_counter is None else positions_counter.clone()
            ),
        )
        if not _every_arena_is_pooled(state):
            # No pool was declared, so each layer owns its storage for life and
            # writes through `slab`. There is nothing to relet and nothing whose
            # address would move -- but also no captured entry could ever be
            # reused, because its state cannot change pages. Standalone
            # MuScriptor and the CPU tests live here.
            cache.refuse("not-pooled")
            prefill_memory.release_state(state)
            return None
        forward = _graph_forward(lm, request, state, buffers)
        measure = (
            phase_recorder.measure("prefill_graph_capture")
            if phase_recorder is not None
            else nullcontext()
        )
        # **Without the weight cache**, or the capture bakes the addresses of
        # half-precision weight copies that autocast frees when this block
        # exits -- and every replay after that reads freed memory. See
        # `TorchAutocast.without_weight_cache`.
        with measure, lm.autocast.without_weight_cache():
            # Warm against this exact state before capturing it. The warm-up is
            # a real prefill of the same row, so the capture that follows costs
            # one extra forward rather than a wasted one.
            forward()
            _reset_prefill_cursors(state)
            entry = cache.capture(key, state, buffers, forward)
        if entry is None:
            prefill_memory.release_state(state)
            return None
        next_tokens = entry.output
    else:
        cache.relet(entry)
        _reset_prefill_cursors(entry.state)
        buffers = entry.buffers
        buffers.sequence.copy_(sequence)
        buffers.forbidden.copy_(forbidden)
        buffers.temporal_shift_values.copy_(shift_values)
        buffers.temporal_floors.copy_(floors)
        buffers.sample_mask.copy_(sample_mask)
        buffers.last_index.copy_(last_index)
        for name, (cond, mask) in conditions.items():
            static_cond, static_mask = buffers.conditions[name]
            static_cond.copy_(cond)
            static_mask.copy_(mask)
        if buffers.sampling_seeds is not None:
            buffers.sampling_seeds.copy_(seeds)
            buffers.sampling_positions.copy_(positions_counter)
        measure = (
            phase_recorder.measure("first_token")
            if phase_recorder is not None
            else nullcontext()
        )
        with measure:
            next_tokens = cache.replay(entry)

    increment_steps(lm.transformer, entry.state, increment=prepend_length + tokens)
    return _row_from_prefill(
        request,
        entry.state,
        conditions,
        prompt,
        int(next_tokens[0]),
        decode_span=decode_span,
        temporal_floor_source=floors,
        arena=_PrefillArena(
            entry.state, 1, release=lambda _state, _entry=entry: cache.release(_entry)
        ),
    )


def _every_arena_is_pooled(state) -> bool:
    """Whether every layer of this state draws from a supply it can give back to.

    The capture cache reuses a state by returning its pages and taking new
    ones, so an arena that owns its storage cannot take part -- not because
    replaying would be wrong, but because nothing about it could ever change
    and a second admission would need its own state anyway.
    """
    arenas = [
        layer["kv"] for layer in state.values() if layer.get("kv") is not None
    ]
    return bool(arenas) and all(arena.borrowed for arena in arenas)


def _graph_forward(lm, request: GenerationRequest, state, buffers: _Buffers):
    """The captured region: one prefill forward over caller-owned buffers."""

    def forward() -> torch.Tensor:
        return lm._sample_next_token(
            buffers.sequence,
            buffers.conditions,
            state,
            first_step=True,
            use_sampling=request.use_sampling,
            temp=request.temperature,
            top_k=0,
            top_p=0.0,
            cfg_coef=request.cfg_coef,
            forbidden_tokens=buffers.forbidden,
            temporal_shift_values=buffers.temporal_shift_values,
            temporal_floors=buffers.temporal_floors,
            sample_mask=buffers.sample_mask,
            generator=None,
            sampling_seeds=buffers.sampling_seeds,
            sampling_positions=buffers.sampling_positions,
            trace_collectors=(None,),
            trace_contexts=(None,),
            last_index=buffers.last_index,
        )

    return forward


def _row_from_prefill(
    request: GenerationRequest,
    state,
    conditions: dict[str, tuple[torch.Tensor, torch.Tensor]],
    prompt: list[int],
    next_token: int,
    *,
    decode_span: int,
    temporal_floor_source: torch.Tensor,
    arena: _PrefillArena,
) -> PreparedGenerationRow:
    """The row both width-one prefill paths build, from the same fields."""
    emitted_eos = next_token == request.eos_id
    if not emitted_eos:
        prompt.append(next_token)
    steps = len(request.prompt_ids) + 1
    return PreparedGenerationRow(
        request=request,
        model_state=state,
        prefill_arena=arena,
        conditions=conditions,
        tokens=prompt,
        last_token=next_token,
        steps=steps,
        finished=emitted_eos or steps >= request.max_gen_len,
        emitted_eos=emitted_eos,
        decode_span=decode_span,
        temporal_floor=_advanced_temporal_floor(
            request, int(temporal_floor_source[0].item()), next_token
        ),
        checkpoint_floor=(
            -1
            if request.temporal_grammar is None
            or request.temporal_grammar.checkpoint_floor is None
            else request.temporal_grammar.checkpoint_floor
        ),
    )


def _prepare_generation_row_with_conditions(
    lm,
    request: GenerationRequest,
    conditions: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    phase_recorder: _CudaPhaseRecorder | None = None,
) -> PreparedGenerationRow:
    """Prefill one row from conditions encoded by a larger producer batch."""
    cfg_enabled = request.cfg_coef != 1.0
    prepend_length = sum(cond.shape[1] for cond, _ in conditions.values())
    decode_span = prepend_length + request.max_gen_len
    prompt = list(request.prompt_ids[: request.max_gen_len])
    if len(prompt) < request.max_gen_len:
        row = _graph_prefill_row(
            lm,
            request,
            conditions,
            prepend_length=prepend_length,
            prompt=prompt,
            decode_span=decode_span,
            phase_recorder=phase_recorder,
        )
        if row is not None:
            return row
    state = _prefill_states(
        lm,
        batch_size=2 if cfg_enabled else 1,
        prepend_length=prepend_length,
        prompt_length=len(prompt),
    )
    arena = _PrefillArena(state, 1)
    if len(prompt) >= request.max_gen_len:
        return PreparedGenerationRow(
            request, state, conditions, prompt, lm.initial_token_id,
            len(prompt), True, False, decode_span, prefill_arena=arena,
        )
    sequence = torch.tensor(
        [[lm.initial_token_id, *prompt]],
        device=lm.emb.weight.device,
        dtype=torch.long,
    )
    with (
        phase_recorder.measure("first_token")
        if phase_recorder is not None
        else nullcontext()
    ), lm.autocast:
        sampling_seeds, sampling_positions, sampling_mask = _sampling_metadata(
            (request,),
            lm.emb.weight.device,
        )
        temporal_shift_values, temporal_floors = _temporal_metadata(
            (request,),
            lm.emb.weight.device,
            lm.card,
        )
        next_token = int(
            lm._sample_next_token(
                sequence,
                conditions,
                state,
                first_step=True,
                use_sampling=request.use_sampling,
                temp=request.temperature,
                top_k=0,
                top_p=0.0,
                cfg_coef=request.cfg_coef,
                forbidden_tokens=_forbidden_mask(lm, request),
                temporal_shift_values=temporal_shift_values,
                temporal_floors=temporal_floors,
                sample_mask=sampling_mask,
                generator=None,
                sampling_seeds=sampling_seeds,
                sampling_positions=sampling_positions,
                trace_collectors=(request.trace_collector,),
                trace_contexts=(request.trace_context,),
            )[0]
        )
    increment_steps(
        lm.transformer,
        state,
        increment=sequence.shape[1] + prepend_length,
    )
    emitted_eos = next_token == request.eos_id
    if not emitted_eos:
        prompt.append(next_token)
    steps = len(request.prompt_ids) + 1
    temporal_floor = _advanced_temporal_floor(
        request,
        int(temporal_floors[0].item()),
        next_token,
    )
    return PreparedGenerationRow(
        request=request,
        model_state=state,
        prefill_arena=arena,
        conditions=conditions,
        tokens=prompt,
        last_token=next_token,
        steps=steps,
        finished=emitted_eos or steps >= request.max_gen_len,
        emitted_eos=emitted_eos,
        decode_span=decode_span,
        temporal_floor=temporal_floor,
        checkpoint_floor=(
            -1
            if request.temporal_grammar is None
            or request.temporal_grammar.checkpoint_floor is None
            else request.temporal_grammar.checkpoint_floor
        ),
    )


@torch.inference_mode()
def prepare_generation_rows(
    lm,
    requests: tuple[GenerationRequest, ...],
    *,
    phase_recorder: _CudaPhaseRecorder | None = None,
    _conditions: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    _record_condition_batch: Callable[[int], None] | None = None,
    _record_first_token_batch: Callable[[int], None] | None = None,
    _record_packed_prefill_batch: Callable[[int], None] | None = None,
) -> tuple[PreparedGenerationRow, ...]:
    """Prefill compatible equal-prompt rows in one model call."""
    if not requests:
        return ()
    key = requests[0].session_compatibility_key
    if any(request.session_compatibility_key != key for request in requests):
        raise ValueError("continuous generation rows are incompatible")
    if any(request.beam_size != 1 for request in requests):
        raise ValueError("continuous generation requires beam_size=1")
    prompt_lengths = {len(request.prompt_ids) for request in requests}
    if len(requests) == 1:
        if _record_first_token_batch is not None:
            _record_first_token_batch(1)
        if _conditions is not None:
            return (
                _prepare_generation_row_with_conditions(
                    lm,
                    requests[0],
                    _conditions,
                    phase_recorder=phase_recorder,
                ),
            )
        return tuple(
            prepare_generation_row(
                lm,
                request,
                phase_recorder=phase_recorder,
            )
            for request in requests
        )

    cfg_enabled = requests[0].cfg_coef != 1.0
    cfg_width = 2 if cfg_enabled else 1
    batch_size = len(requests)
    if _conditions is None:
        condition_measure = (
            phase_recorder.measure("condition_encode")
            if phase_recorder is not None
            else nullcontext()
        )
        with condition_measure:
            conditions = lm.prepare_generation_conditions(
                [request.condition for request in requests],
                requests[0].cfg_coef,
            )
        if _record_condition_batch is not None:
            _record_condition_batch(len(requests))
    else:
        conditions = _conditions
    if len(prompt_lengths) != 1:
        if (
            getattr(lm, "packed_prefill_enabled", False)
            and all(
                len(request.prompt_ids) < request.max_gen_len
                for request in requests
            )
        ):
            return _prepare_generation_rows_ragged(
                lm,
                requests,
                conditions,
                phase_recorder=phase_recorder,
                record_first_token_batch=_record_first_token_batch,
                record_packed_prefill_batch=_record_packed_prefill_batch,
            )
        groups: dict[int, list[int]] = {}
        for row, request in enumerate(requests):
            groups.setdefault(len(request.prompt_ids), []).append(row)
        prepared_by_row: dict[int, PreparedGenerationRow] = {}
        for rows in groups.values():
            group_requests = tuple(requests[row] for row in rows)
            source_rows = list(rows)
            if cfg_enabled:
                source_rows += [row + batch_size for row in rows]
            group_conditions = {
                name: (
                    condition[source_rows],
                    mask[source_rows],
                )
                for name, (condition, mask) in conditions.items()
            }
            group_prepared = prepare_generation_rows(
                lm,
                group_requests,
                phase_recorder=phase_recorder,
                _conditions=group_conditions,
                _record_first_token_batch=_record_first_token_batch,
            )
            prepared_by_row.update(
                zip(rows, group_prepared, strict=True)
            )
        return tuple(prepared_by_row[row] for row in range(batch_size))
    prepend_length = sum(cond.shape[1] for cond, _ in conditions.values())
    decode_span = prepend_length + max(
        request.max_gen_len for request in requests
    )
    prompts = [
        list(request.prompt_ids[: request.max_gen_len]) for request in requests
    ]
    if all(len(prompt) >= request.max_gen_len for prompt, request in zip(
        prompts, requests, strict=True
    )):
        return tuple(
            _prepare_generation_row_with_conditions(
                lm,
                request,
                {
                    name: (
                        condition[
                            [
                                row,
                                row + batch_size,
                            ]
                            if cfg_enabled
                            else [row]
                        ],
                        mask[
                            [
                                row,
                                row + batch_size,
                            ]
                            if cfg_enabled
                            else [row]
                        ],
                    )
                    for name, (condition, mask) in conditions.items()
                },
                phase_recorder=phase_recorder,
            )
            for row, request in enumerate(requests)
        )
    if any(len(prompt) >= request.max_gen_len for prompt, request in zip(
        prompts, requests, strict=True
    )):
        return tuple(
            _prepare_generation_row_with_conditions(
                lm,
                request,
                {
                    name: (
                        condition[[row, row + batch_size] if cfg_enabled else [row]],
                        mask[[row, row + batch_size] if cfg_enabled else [row]],
                    )
                    for name, (condition, mask) in conditions.items()
                },
                phase_recorder=phase_recorder,
        )
        for row, request in enumerate(requests)
        )

    # Below both per-row fallbacks on purpose: they prefill each row privately
    # and never touch this state, so allocating it above them left a whole
    # cohort-wide KV allocation whose only exit was the garbage collector --
    # invisible while that was the allocator's problem, a permanent loss of
    # pool pages once the state is borrowed.
    state = _prefill_states(
        lm,
        batch_size=batch_size * cfg_width,
        prepend_length=prepend_length,
        prompt_length=max(len(prompt) for prompt in prompts),
    )
    arena = _PrefillArena(state, len(requests))
    if _record_first_token_batch is not None:
        _record_first_token_batch(batch_size)
    sequence = torch.tensor(
        [[lm.initial_token_id, *prompt] for prompt in prompts],
        device=lm.emb.weight.device,
        dtype=torch.long,
    )
    forbidden = torch.zeros(
        (batch_size, lm.card),
        device=lm.emb.weight.device,
        dtype=torch.bool,
    )
    for row, request in enumerate(requests):
        if request.forbidden_token_ids:
            forbidden[row, list(request.forbidden_token_ids)] = True
    first_token_measure = (
        phase_recorder.measure("first_token")
        if phase_recorder is not None
        else nullcontext()
    )
    with first_token_measure, lm.autocast:
        sampling_seeds, sampling_positions, sampling_mask = _sampling_metadata(
            requests,
            lm.emb.weight.device,
        )
        temporal_shift_values, temporal_floors = _temporal_metadata(
            requests,
            lm.emb.weight.device,
            lm.card,
        )
        next_tokens = lm._sample_next_token(
            sequence,
            conditions,
            state,
            first_step=True,
            use_sampling=any(request.use_sampling for request in requests),
            temp=requests[0].temperature,
            top_k=0,
            top_p=0.0,
            cfg_coef=requests[0].cfg_coef,
            forbidden_tokens=forbidden,
            temporal_shift_values=temporal_shift_values,
            temporal_floors=temporal_floors,
            sample_mask=sampling_mask,
            generator=None,
            sampling_seeds=sampling_seeds,
            sampling_positions=sampling_positions,
            trace_collectors=tuple(request.trace_collector for request in requests),
            trace_contexts=tuple(request.trace_context for request in requests),
        )
    increment_steps(
        lm.transformer,
        state,
        increment=sequence.shape[1] + prepend_length,
    )

    return _rows_from_first_token(
        lm=lm,
        requests=requests,
        prompts=prompts,
        next_tokens=next_tokens,
        temporal_floors=temporal_floors,
        conditions=conditions,
        state=state,
        arena=arena,
        batch_size=batch_size,
        cfg_enabled=cfg_enabled,
        decode_span=decode_span,
    )

