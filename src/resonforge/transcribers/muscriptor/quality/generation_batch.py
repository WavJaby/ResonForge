"""Independent stateful chunk rows sharing one model generation call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from muscriptor.model_trace import ModelTraceCollector
from muscriptor.modules.conditioners import ConditioningAttributes
from muscriptor.tokenizer.notes import Event

from resonforge.transcribers.muscriptor.quality.generation_anomaly import MonitorSummary
from resonforge.transcribers.muscriptor.quality.generation_guard import (
    GenerationGuardConfig,
    GuardAction,
    GuardFinding,
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


@dataclass(frozen=True)
class RecoveryCandidateSpec:
    """One recovery candidate: a plain `GenerationRequest` plus its envelope.

    Generation semantics live only on `request` -- the same type, checkpoint
    derivation (`checkpoint_shift_tokens`) and scheduler protocol the primary
    chunk uses. The envelope is what actually differs between a candidate and
    a primary chunk: where it runs (`target_model` / `model_role`) and how the
    completed row is validated (`candidate_name`, the frame-rate check, margin
    collection). The scheduler-facing properties delegate, so a spec submits
    exactly like the request it carries.
    Replaces `RecoveryCandidateRequest`, which duplicated fourteen request
    fields and six property implementations under second names
    (`expected_vocab` = `guard_vocab`, `expected_eos_id` = `eos_id`,
    `verify_shift_value` -> `checkpoint_token_ids`).
    """

    candidate_name: Literal[
        "shifted_replay",
        "new_seed",
        "secondary_model",
        "bootstrap",
        "fresh_reanchor",
    ]
    request: GenerationRequest
    expected_frame_rate: int
    target_model: str | None = None
    model_role: Literal["primary", "recovery"] = "recovery"
    collect_shift_margins: bool = False

    # -- BatchSubmission.item protocol, delegated ------------------------
    @property
    def condition(self) -> ConditioningAttributes:
        return self.request.condition

    @property
    def cfg_coef(self) -> float:
        return self.request.cfg_coef

    @property
    def prompt_ids(self) -> tuple[int, ...]:
        return self.request.prompt_ids

    @property
    def use_sampling(self) -> bool:
        return self.request.use_sampling

    @property
    def recovery_group_claim(self) -> RecoveryGroupClaim | None:
        return self.request.recovery_group_claim

    @property
    def prefill_token_budget(self) -> int:
        return self.request.prefill_token_budget

    @property
    def remaining_decode_token_budget(self) -> int:
        return self.request.remaining_decode_token_budget

    @property
    def requires_dependency_credit(self) -> bool:
        return self.request.requires_dependency_credit

    @property
    def initial_kv_capacity_bucket(self) -> int:
        return self.request.initial_kv_capacity_bucket

    @property
    def session_compatibility_key(self) -> tuple[object, ...]:
        return self.request.session_compatibility_key

    @property
    def compatibility_key(self) -> tuple[object, ...]:
        return self.request.compatibility_key


@dataclass(frozen=True)
class RecoveryCandidateGroupRequest:
    """Independent recovery candidates submitted together to the scheduler."""

    requests: tuple[RecoveryCandidateSpec, ...]
    parent_resident_handle: object | None = None

    def __post_init__(self) -> None:
        if len(self.requests) < 2:
            raise ValueError("recovery candidate group requires at least two rows")


@dataclass(frozen=True)
class RecoveryCandidateResult:
    """Plain completed candidate data returned by a model worker."""

    candidate_name: Literal[
        "shifted_replay",
        "new_seed",
        "secondary_model",
        "bootstrap",
        "fresh_reanchor",
    ]
    tokens: tuple[int, ...]
    emitted_eos: bool
    model_name: str
    shift_margins: tuple[float, ...] = ()
    resident_handle: object | None = None
    guard_action: GuardAction = "continue"
    guard_findings: tuple[GuardFinding, ...] = ()
    guard_summary: MonitorSummary | None = None
