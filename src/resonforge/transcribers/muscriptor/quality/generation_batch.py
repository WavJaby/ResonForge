"""Independent stateful chunk rows sharing one model generation call."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import torch
from muscriptor.diagnostics import route_iterator
from muscriptor.model_trace import ModelTraceCollector
from muscriptor.modules.conditioners import ConditioningAttributes
from muscriptor.tokenizer.notes import Event

from resonforge.transcribers.muscriptor.quality.generation_anomaly import MonitorSummary
from resonforge.transcribers.muscriptor.quality.generation_guard import (
    GenerationGuardConfig,
    GuardAction,
    GuardFinding,
)
from resonforge.transcribers.muscriptor.quality.generation_position import (
    GenerationPosition,
)


@dataclass(frozen=True)
class TemporalGrammarConfig:
    """Token semantics and immutable boundaries for monotonic musical time."""

    shift_values: tuple[int, ...]
    initial_floor: int = -1
    checkpoint_floor: int | None = None

    def __post_init__(self) -> None:
        if not self.shift_values:
            raise ValueError("temporal grammar requires a token vocabulary")
        if any(value < -1 for value in self.shift_values):
            raise ValueError("temporal shift values must use -1 for non-shifts")
        if self.initial_floor < -1:
            raise ValueError("initial temporal floor must be at least -1")
        if self.checkpoint_floor is not None and self.checkpoint_floor < 0:
            raise ValueError("checkpoint temporal floor must be non-negative")

    @classmethod
    def from_vocab(
        cls,
        vocab: tuple[Event, ...],
        prompt_ids: tuple[int, ...],
        checkpoint_token_ids: tuple[int, ...] = (),
    ) -> TemporalGrammarConfig:
        shift_values = tuple(
            event.value if event.type == "shift" and event.value >= 0 else -1
            for event in vocab
        )
        floor = -1
        for token in prompt_ids:
            if not 0 <= token < len(shift_values):
                raise ValueError("temporal prompt contains an invalid token")
            value = shift_values[token]
            if value >= 0:
                if value < floor:
                    raise ValueError("temporal prompt contains decreasing shifts")
                floor = value
        checkpoint_values = tuple(
            shift_values[token]
            for token in checkpoint_token_ids
            if 0 <= token < len(shift_values) and shift_values[token] >= 0
        )
        return cls(
            shift_values,
            initial_floor=floor,
            checkpoint_floor=(
                min(checkpoint_values) if checkpoint_values else None
            ),
        )


@dataclass(frozen=True)
class RecoveryClaimMember:
    """One maximum downstream member required before a parent can release."""

    model_role: Literal["primary", "recovery"]
    target_model: str | None = None


@dataclass(frozen=True)
class RecoveryGroupClaim:
    """Typed transitive resource claim declared before parent admission."""

    members: tuple[RecoveryClaimMember, ...]

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise ValueError("recovery group claim requires at least two members")


# The KV span every session opens at. A hard cap, not an estimate: a request
# asking for more is asking for something no arena is built to address.
MAX_KV_CAPACITY = 2048


@dataclass(frozen=True)
class GenerationRequest:
    """One ready chunk whose dependency state has already been resolved."""

    condition: ConditioningAttributes
    prompt_ids: tuple[int, ...]
    forbidden_token_ids: tuple[int, ...]
    max_gen_len: int
    use_sampling: bool
    temperature: float
    cfg_coef: float
    eos_id: int
    beam_size: int = 1
    prompt_bucket: int | None = None
    length_bucket: int | None = None
    kv_capacity_hint: int | None = None
    decode_token_estimate: int | None = None
    checkpoint_token_ids: tuple[int, ...] = ()
    trace_collector: ModelTraceCollector | None = None
    trace_context: tuple[object, ...] = ()
    sampling_seed: int | None = None
    guard_config: GenerationGuardConfig | None = None
    guard_vocab: tuple[Event, ...] = ()
    temporal_grammar: TemporalGrammarConfig | None = None
    recovery_group_claim: RecoveryGroupClaim | None = None

    def __post_init__(self) -> None:
        if self.max_gen_len < 1:
            raise ValueError("max_gen_len must be positive")
        if self.length_bucket is not None and self.length_bucket not in {
            512,
            1024,
            2048,
        }:
            raise ValueError("length_bucket must be 512, 1024, or 2048")
        if self.kv_capacity_hint is not None and self.kv_capacity_hint < 0:
            raise ValueError("kv_capacity_hint cannot be negative")
        if self.decode_token_estimate is not None and self.decode_token_estimate < 1:
            raise ValueError("decode_token_estimate must be positive")
        if self.sampling_seed is not None and self.sampling_seed < 0:
            raise ValueError("sampling_seed must be non-negative")

    @property
    def padded_prompt_length(self) -> int:
        return self.prompt_bucket or len(self.prompt_ids)

    @property
    def prefill_token_budget(self) -> int:
        """Estimated condition plus prompt work used only for scheduling."""
        return 384 + self.padded_prompt_length

    @property
    def remaining_decode_token_budget(self) -> int:
        """Estimated initial decode work used only for scheduling."""
        estimate = (
            self.max_gen_len
            if self.decode_token_estimate is None
            else min(self.max_gen_len, self.decode_token_estimate)
        )
        return max(0, estimate - len(self.prompt_ids))

    @property
    def requires_dependency_credit(self) -> bool:
        """Whether admission can create a resident checkpoint dependency."""
        return bool(self.checkpoint_token_ids) or (
            self.guard_config is not None
            and self.guard_config.mode == "recovery"
            and self.guard_config.role != "fresh"
        ) or self.recovery_group_claim is not None

    @property
    def initial_kv_capacity_bucket(self) -> int:
        """The one KV span every session is opened at.

        It used to pick from 512 / 1024 / 2048 by what the request estimated it
        would need, and that tiering was **memory rationing**: rows sharing an
        arena shared its `capacity`, so a short row admitted into a long
        session paid the long session's price for its whole life.

        Since A3 a row draws pages as it reaches them and gives them back when
        it ends, so a bucket buys nothing and costs the thing it was blocking:
        rows of different estimated lengths went into *different sessions*,
        competing for one device instead of sharing one batch. That is the
        batching freedom `DESIGN.md` names as the one vLLM motivation that
        applies here.

        What a single span still costs is the block *table* -- `rows x
        blocks_per_row` int64 per layer, about 289 KiB at the shipped width --
        and the `sequence` tensor. Both are four orders below one row's KV.
        """
        return MAX_KV_CAPACITY

    @property
    def compatibility_key(self) -> tuple[object, ...]:
        """Dense-batch key; generation length remains a shared tensor shape."""
        return (
            self.max_gen_len,
            *self.session_compatibility_key,
        )

    @property
    def session_compatibility_key(self) -> tuple[object, ...]:
        """Tensor semantics shared by one physical mixed-mode session."""
        return (
            self.temperature,
            self.cfg_coef,
            self.eos_id,
            self.beam_size,
        )


@dataclass(frozen=True)
class GenerationResult:
    """Completed tokens for one row, excluding EOS and post-EOS padding."""

    tokens: tuple[int, ...]
    emitted_eos: bool
    batch_steps: int = 0
    wasted_token_rows: int = 0
    resident_handle: object | None = None
    guard_action: GuardAction = "continue"
    guard_findings: tuple[GuardFinding, ...] = ()
    guard_summary: MonitorSummary | None = None

    @property
    def checkpointed(self) -> bool:
        return self.resident_handle is not None


@dataclass(frozen=True)
class GenerationControlRequest:
    """Resume or discard one worker-resident generation row."""

    resident_handle: object
    action: Literal["resume", "discard"]


@dataclass(frozen=True)
class GenerationControlResult:
    """Acknowledgement for a generation control action without token output."""

    discarded: bool


def run_generation_batch(
    model: object,
    requests: tuple[GenerationRequest, ...],
    *,
    stdout_logger=None,
) -> tuple[GenerationResult, ...]:
    """Run compatible independent rows through one LM ``generate`` call."""
    if len(requests) > 1:
        prompt_groups: dict[int, list[int]] = {}
        for row, request in enumerate(requests):
            prompt_groups.setdefault(len(request.prompt_ids), []).append(row)
        if len(prompt_groups) > 1:
            # ``LMModel.generate`` uses one global start offset.  Passing
            # shorter prompts padded with ``ungenerated_token_id`` would feed
            # those sentinels back into the model and change its output.  Keep
            # the dense path exact by batching equal-length prompts; the
            # continuous packed-prefill path remains the ragged fast path.
            grouped_results: dict[int, GenerationResult] = {}
            for rows in prompt_groups.values():
                group = tuple(requests[row] for row in rows)
                for row, result in zip(
                    rows,
                    run_generation_batch(
                        model,
                        group,
                        stdout_logger=stdout_logger,
                    ),
                    strict=True,
                ):
                    grouped_results[row] = result
            return tuple(grouped_results[row] for row in range(len(requests)))
    try:
        return _run_generation_batch_once(
            model,
            requests,
            stdout_logger=stdout_logger,
        )
    except torch.cuda.OutOfMemoryError:
        if len(requests) <= 1:
            raise
        torch.cuda.empty_cache()
        midpoint = len(requests) // 2
        return run_generation_batch(
            model,
            requests[:midpoint],
            stdout_logger=stdout_logger,
        ) + run_generation_batch(
            model,
            requests[midpoint:],
            stdout_logger=stdout_logger,
        )


def _run_generation_batch_once(
    model: object,
    requests: tuple[GenerationRequest, ...],
    *,
    stdout_logger=None,
) -> tuple[GenerationResult, ...]:
    if not requests:
        return ()
    compatibility_key = requests[0].compatibility_key
    if any(request.compatibility_key != compatibility_key for request in requests):
        raise ValueError("generation batch rows are incompatible")
    first = requests[0]
    if first.beam_size != 1:
        raise ValueError("independent row batching currently requires beam_size=1")

    device = model._device
    prompt_lengths = {len(request.prompt_ids) for request in requests}
    if len(prompt_lengths) > 1:
        raise ValueError("generation batch rows require equal prompt lengths")
    prompt_length = prompt_lengths.pop()
    prompt = None
    if prompt_length:
        prompt = torch.tensor(
            [request.prompt_ids for request in requests],
            device=device,
            dtype=torch.long,
        )
    card = getattr(model._model, "card", len(model._tokenizer._vocab))
    forbidden_mask = torch.zeros(
        (len(requests), card),
        device=device,
        dtype=torch.bool,
    )
    for row, request in enumerate(requests):
        if request.forbidden_token_ids:
            forbidden_mask[row, list(request.forbidden_token_ids)] = True

    temporal_configs = tuple(request.temporal_grammar for request in requests)
    configured_grammars = tuple(
        config for config in temporal_configs if config is not None
    )
    temporal_shift_values = None
    temporal_floors = None
    if configured_grammars:
        shift_values = configured_grammars[0].shift_values
        if len(shift_values) > card:
            raise ValueError("temporal grammar vocabulary exceeds model card")
        if any(
            config is None or config.shift_values != shift_values
            for config in temporal_configs
        ):
            raise ValueError("generation batch requires one temporal token mapping")
        temporal_shift_values = torch.tensor(
            (*shift_values, *((-1,) * (card - len(shift_values)))),
            device=device,
            dtype=torch.long,
        )
        temporal_floors = torch.tensor(
            [config.initial_floor for config in configured_grammars],
            device=device,
            dtype=torch.long,
        )

    generator = None
    if first.sampling_seed is not None:
        from resonforge.transcribers.muscriptor.quality.recovery import (
            make_recovery_generator,
        )

        generator = make_recovery_generator(device, first.sampling_seed)
    steps = list(
        route_iterator(
            model._model.generate(
                prompt=prompt,
                conditions=[request.condition for request in requests],
                max_gen_len=first.max_gen_len,
                use_sampling=first.use_sampling,
                temp=first.temperature,
                top_k=0,
                top_p=0.0,
                cfg_coef=first.cfg_coef,
                early_stop_on_token=first.eos_id,
                beam_size=first.beam_size,
                forbidden_tokens=forbidden_mask,
                generator=generator,
                early_stop_check_interval=8,
                trace_collectors=tuple(r.trace_collector for r in requests),
                trace_contexts=tuple(r.trace_context for r in requests),
                trace_prompt_lengths=tuple(len(r.prompt_ids) for r in requests),
                temporal_shift_values=temporal_shift_values,
                temporal_floors=temporal_floors,
            ),
            stdout_logger,
        )
    )
    if not steps:
        return tuple(GenerationResult((), False) for _ in requests)
    generated = torch.stack(steps, dim=1).cpu()
    results = []
    for row in range(len(requests)):
        request = requests[row]
        row_tokens = generated[row].tolist()
        try:
            eos_index = row_tokens.index(first.eos_id)
        except ValueError:
            eos_index = len(row_tokens)
            emitted_eos = False
        else:
            emitted_eos = True
        temporal_finding = None
        grammar = request.temporal_grammar
        if grammar is not None:
            floor = grammar.initial_floor
            safe_frontier = None
            for token_index, token in enumerate(
                row_tokens[len(request.prompt_ids) : eos_index]
            ):
                shift = grammar.shift_values[token]
                if shift < 0:
                    continue
                if shift < floor:
                    position = GenerationPosition(token_index, shift)
                    temporal_finding = GuardFinding(
                        detector="temporal_grammar",
                        status="critical",
                        reason="musical_time_regression",
                        metrics={
                            "observed_shift": shift,
                            "required_floor": floor,
                        },
                        observed_at=position,
                        suspected_start=position,
                        last_safe_frontier=safe_frontier,
                    )
                    eos_index = len(request.prompt_ids) + token_index
                    emitted_eos = False
                    break
                floor = max(floor, shift)
                safe_frontier = GenerationPosition(token_index, shift)
        wasted = (
            len(row_tokens) - eos_index
            if temporal_finding is not None
            else (len(row_tokens) - eos_index - 1 if emitted_eos else 0)
        )
        role = "primary" if request.guard_config is None else request.guard_config.role
        results.append(
            GenerationResult(
                tuple(row_tokens[:eos_index]),
                emitted_eos,
                batch_steps=len(row_tokens),
                wasted_token_rows=max(0, wasted),
                guard_action=(
                    "continue"
                    if temporal_finding is None
                    else (
                        "interrupt_to_recovery"
                        if role == "primary"
                        else "reject_candidate"
                    )
                ),
                guard_findings=(
                    () if temporal_finding is None else (temporal_finding,)
                ),
            )
        )
    return tuple(results)


def resolve_generation_requests(
    model: object,
    stream: Iterator[object],
    *,
    stdout_logger=None,
) -> Iterator[object]:
    """Drive a request-yielding token stream with serial one-row batches."""
    response: GenerationResult | None = None
    while True:
        try:
            item = stream.send(response) if response is not None else next(stream)
        except StopIteration:
            return
        response = None
        if isinstance(item, GenerationRequest):
            response = run_generation_batch(
                model,
                (item,),
                stdout_logger=stdout_logger,
            )[0]
        else:
            yield item
