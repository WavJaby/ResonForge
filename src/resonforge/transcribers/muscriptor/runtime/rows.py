"""What prefill hands decode, and the shapes both sides name it by.

These types are the boundary. Prefill produces a `PreparedGenerationRow` and
never looks at a session; decode installs one and never runs a model forward.
They live here rather than on either side so the dependency stays one-way --
`decode` -> `prefill` -- with no module importing the other back.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import torch
from muscriptor.generation_batch import GenerationRequest
from muscriptor.generation_guard import (
    GenerationCompleted,
    GenerationGuard,
    GuardFinding,
    TokenBatchObserved,
)
from muscriptor.generation_position import GenerationPosition
from muscriptor.modules.streaming import ModelState

from resonforge.transcribers.muscriptor.runtime.prefill import (
    memory as prefill_memory,
)


def condition_span_headroom(
    conditions: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    span_bounds: Mapping[str, int] | None,
) -> int:
    """Positions a later replacement row's longer condition prefix can need.

    A session's KV and position tensors are allocated once and then serve every
    hot-replaced row for the session's lifetime, so their length must bound the
    whole family of rows the scheduler may admit, not the batch that happened to
    create the session. `max_gen_len` is part of the compatibility key, so the
    only term that varies inside a family is the condition prefix, and the model
    declares the longest span each of its conditioners can produce
    (`condition_span_bounds`). That makes this an exact ceiling — on the
    transcription model the whole instrument-group vocabulary, tens of tokens —
    rather than a percentage band sized to cover an unexplained handful.

    A model that declares nothing gets no headroom: the session stays sized to
    its own batch and `install` refuses an oversized replacement by its span
    check, which is a contained error rather than an out-of-range index.
    """

    if not span_bounds:
        return 0
    return sum(
        max(0, int(bound) - _condition_span(conditions.get(name)))
        for name, bound in span_bounds.items()
    )


def _condition_span(entry: tuple[torch.Tensor, torch.Tensor] | None) -> int:
    return 0 if entry is None else int(entry[0].shape[1])


@dataclass
class _PrefillArena:
    """One prefill allocation, and the rows that still read out of it.

    A batched prefill writes a whole cohort into one state and each row is
    copied out separately when it is installed, so the KV it borrowed can only
    go back after the last row has left. Counted rather than released at the
    highest index: rows are installed in whatever order slots free up, an
    admission can be refused after the prefill has already run, and a row that
    finished during prefill is never installed at all.

    Before a pool was declared this was invisible -- the state was garbage and
    the allocator took it back. A borrowed range has no such owner: nothing
    frees it but this, and a page never returned is a permanently smaller
    device.
    """

    model_state: ModelState
    outstanding: int
    #: What to do once the last row has left. The default retires the state --
    #: its blocks go back and the dict is emptied, because nothing will use it
    #: again. A state a captured prefill graph was built against **is** used
    #: again: its arenas and their table tensors are the addresses the graph
    #: baked, and only its pages turn over. Such a state is handed in with a
    #: release of its own, which returns the pages and gives the arena back to
    #: the cache that owns it.
    release: Callable[[ModelState], None] = prefill_memory.release_state

    def release_row(self) -> None:
        if self.outstanding < 1:
            raise RuntimeError("prefill arena released more rows than it holds")
        self.outstanding -= 1
        if self.outstanding:
            return
        self.release(self.model_state)


@dataclass
class PreparedGenerationRow:
    request: GenerationRequest
    # The prefill state this row lives in, and where in it. A batched prefill
    # shares one state across its rows rather than copying each into a private
    # buffer, so the row is identified by (state, source_slot, source_width),
    # not by owning a batch-one state of its own.
    model_state: ModelState
    conditions: dict[str, tuple[torch.Tensor, torch.Tensor]]
    tokens: list[int]
    last_token: int
    steps: int
    finished: bool
    emitted_eos: bool
    # The furthest position a slot holding this row may ever address:
    # `prepend_length + max_gen_len`. **Stated, because it used to be read off
    # the prefill allocation's `capacity`** -- by `install`, to refuse a row
    # that would index past the session's shared position table, and by the
    # session constructor, to size that table in the first place. That worked
    # only while prefill allocated for the whole generation it would never
    # write, so the allocation happened to carry the number. Sizing prefill to
    # what prefill writes takes that coincidence away, and a quantity two
    # callers depend on may not be inferred from an allocation size.
    decode_span: int
    source_slot: int = 0
    source_width: int = 1
    temporal_floor: int = -1
    checkpoint_floor: int = -1
    guard: GenerationGuard | None = None
    guard_action: str = "continue"
    temporal_findings: list[GuardFinding] = field(default_factory=list)
    temporal_safe_frontier: GenerationPosition | None = None
    # The prefill allocation `model_state` points into. `None` once the row has
    # given it back, and for rows built by tests that hand in a state directly.
    prefill_arena: _PrefillArena | None = None

    def release_prefill(self) -> None:
        """Drop this row's claim on the prefill state, once it is copied out.

        Idempotent: an admission path may refuse a row this one already
        released, and double-counting would hand a live range back to the pool.
        """
        arena = self.prefill_arena
        self.prefill_arena = None
        self.model_state = {}
        self.conditions = {}
        if arena is not None:
            arena.release_row()

    def __post_init__(self) -> None:
        grammar = self.request.temporal_grammar
        if grammar is not None:
            generated = self.tokens[len(self.request.prompt_ids) :]
            for index, token in enumerate(generated):
                shift = grammar.shift_values[token]
                if shift >= 0:
                    self.temporal_safe_frontier = GenerationPosition(index, shift)
        config = self.request.guard_config
        if config is None or config.mode == "off" or not self.request.guard_vocab:
            return
        self.guard = GenerationGuard(config)
        generated = self.tokens[len(self.request.prompt_ids) :]
        if generated:
            self.observe_guard_tokens(tuple(generated))

    def observe_guard_tokens(self, tokens: tuple[int, ...]) -> str:
        if self.guard is None:
            return "continue"
        vocab = self.request.guard_vocab
        events = tuple(vocab[token] for token in tokens if 0 <= token < len(vocab))
        if not events:
            return "continue"
        decision = self.guard.dispatch(TokenBatchObserved(events))
        if decision.action != "continue" and self.guard_action == "continue":
            self.guard_action = decision.action
        return decision.action

    def observe_temporal_token(self, token: int) -> bool:
        """Accept one legal token or terminate before an illegal suffix."""
        grammar = self.request.temporal_grammar
        if grammar is None:
            return True
        generated_index = len(self.tokens) - len(self.request.prompt_ids)
        if not 0 <= token < len(grammar.shift_values):
            shift = None
        else:
            value = grammar.shift_values[token]
            shift = value if value >= 0 else None
        if shift is not None and shift < self.temporal_floor:
            position = GenerationPosition(generated_index, shift)
            self.temporal_findings.append(
                GuardFinding(
                    detector="temporal_grammar",
                    status="critical",
                    reason="musical_time_regression",
                    metrics={
                        "observed_shift": shift,
                        "required_floor": self.temporal_floor,
                    },
                    observed_at=position,
                    suspected_start=position,
                    last_safe_frontier=self.temporal_safe_frontier,
                )
            )
            role = (
                "primary"
                if self.request.guard_config is None
                else self.request.guard_config.role
            )
            self.guard_action = (
                "interrupt_to_recovery"
                if role == "primary"
                else "reject_candidate"
            )
            self.finished = True
            return False
        if shift is not None:
            self.temporal_floor = max(self.temporal_floor, shift)
            self.temporal_safe_frontier = GenerationPosition(
                generated_index,
                shift,
            )
        return True

    def observe_guard_completion(self) -> None:
        if self.guard is None:
            return
        decision = self.guard.dispatch(
            GenerationCompleted(
                emitted_eos=self.emitted_eos,
                reached_max_length=(
                    not self.emitted_eos
                    and self.steps >= self.request.max_gen_len
                ),
            )
        )
        if decision.action != "continue" and self.guard_action == "continue":
            self.guard_action = decision.action


@dataclass(frozen=True)
class PreparedConditionBatch:
    """Immutable condition tensors awaiting prompt/KV prefill."""

    requests: tuple[GenerationRequest, ...]
    conditions: dict[str, tuple[torch.Tensor, torch.Tensor]]


def condition_batch_key(request: GenerationRequest) -> tuple[object, ...]:
    """Return only constraints used by model condition encoding."""
    condition = request.condition
    wav_schema = tuple(
        sorted(
            (
                name,
                int(value.wav.shape[1]),
                str(value.wav.dtype),
                str(value.wav.device),
                tuple(int(rate) for rate in value.sample_rate),
            )
            for name, value in condition.wav.items()
        )
    )
    return (
        request.cfg_coef,
        tuple(sorted(condition.text)),
        wav_schema,
        tuple(sorted(condition.joint_embed)),
        tuple(sorted(condition.symbolic)),
    )


def split_prepared_condition_batch(
    batch: PreparedConditionBatch,
    sizes: tuple[int, ...],
) -> tuple[PreparedConditionBatch, ...]:
    """Split one encoded cohort back into ordered session-owned views."""
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("condition split sizes must be positive")
    total = sum(sizes)
    if total != len(batch.requests):
        raise ValueError("condition split sizes do not cover the batch")
    cfg_enabled = batch.requests[0].cfg_coef != 1.0
    results = []
    start = 0
    for size in sizes:
        requests = batch.requests[start : start + size]
        conditions = {}
        for name, (condition, mask) in batch.conditions.items():
            condition_rows = condition.narrow(0, start, size)
            mask_rows = mask.narrow(0, start, size)
            if cfg_enabled:
                condition_rows = torch.cat(
                    (
                        condition_rows,
                        condition.narrow(0, total + start, size),
                    ),
                    dim=0,
                )
                mask_rows = torch.cat(
                    (mask_rows, mask.narrow(0, total + start, size)),
                    dim=0,
                )
            conditions[name] = (condition_rows, mask_rows)
        results.append(PreparedConditionBatch(requests, conditions))
        start += size
    return tuple(results)


class GenerationPhase(StrEnum):
    """Scheduler-visible phase without changing Transformer compatibility."""

    VERIFY = "verify"
    CONTINUE = "continue"
    RECOVERY_VERIFY = "recovery_verify"
    PAUSED_PRIMARY = "paused_primary"


@dataclass(frozen=True)
class GenerationHandle:
    """Opaque ownership token for one paused row and its resident KV state."""

    session_id: int
    slot: int
    generation: int
    phase: GenerationPhase
    steps: int
    token_count: int
    logical_bucket: int
    remaining_token_budget: int


@dataclass(frozen=True)
class ContinuousGenerationStats:
    physical_steps: int
    wasted_token_rows: int
    replacements: int
    replacement_attempts: int = 0
    replacement_misses: int = 0
    active_steps_by_width: dict[int, int] = field(default_factory=dict)

