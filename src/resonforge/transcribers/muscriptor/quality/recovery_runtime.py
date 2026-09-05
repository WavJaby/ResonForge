"""Sequential overlap forcing and anomaly-recovery orchestration."""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from muscriptor.events import ChunkBoundary, OpenNoteTracker, ProgressEvent
from muscriptor.model_trace import ModelTraceCollector
from muscriptor.modules.conditioners import ConditioningAttributes
from muscriptor.tokenizer.notes import Event
from muscriptor.utils.sampling import chosen_token_margins

from resonforge.transcribers.muscriptor.quality import overlap_runtime
from resonforge.transcribers.muscriptor.quality.chunk_quality import (
    AdaptiveChunkQualityAssessment,
    ChunkQualityHistory,
    ChunkQualityMetrics,
    ChunkQualityReference,
    assess_adaptive_chunk_quality,
    locate_restart_chain_collapse,
    measure_chunk_quality,
)
from resonforge.transcribers.muscriptor.quality.generation_anomaly import (
    AnomalyMonitorConfig,
    GenerationAnomalyEvent,
    MonitorSummary,
)
from resonforge.transcribers.muscriptor.quality.generation_batch import (
    GenerationControlRequest,
    GenerationControlResult,
    GenerationRequest,
    GenerationResult,
    RecoveryCandidateGroupRequest,
    RecoveryCandidateResult,
    RecoveryCandidateSpec,
    RecoveryClaimMember,
    RecoveryGroupClaim,
    TemporalGrammarConfig,
)
from resonforge.transcribers.muscriptor.quality.generation_guard import (
    ChunkQualityReady,
    GenerationGuard,
    GenerationGuardConfig,
    GuardAction,
    GuardFinding,
    HardPitchEnvelopeConfig,
    OverlapVerificationDetector,
    VerifyReady,
)
from resonforge.transcribers.muscriptor.quality.model_protocol import (
    ModelProtocol,
    TokenizerProtocol,
)
from resonforge.transcribers.muscriptor.quality.monitored_generation import (
    MonitoredGeneration,
)
from resonforge.transcribers.muscriptor.quality.overlap import (
    OverlapDiagnosticsEvent,
    OverlapMatch,
    OverlapWindow,
    match_note_events,
)
from resonforge.transcribers.muscriptor.quality.recovery import (
    OverlapProvenanceProbe,
    RecoveryDiagnosticsEvent,
    build_overlap_probe_stream,
    candidate_safe_frontier,
    derive_recovery_seed,
    evaluate_candidate,
    recovery_candidate_order,
    select_checkpoint_candidate,
)

SecondaryCandidate = Literal["secondary_model", "new_seed"]
RecoverySelection = Literal["checkpoint", "completed_quality"]

if TYPE_CHECKING:
    pass

_SAMPLE_RATE = 16000
_SEGMENT_DURATION = 5.0


def checkpoint_shift_tokens(
    vocab: tuple[Event, ...] | list[Event],
    minimum_shift: int | None,
) -> tuple[int, ...]:
    """Token ids of every shift event at or past `minimum_shift`.

    The one derivation of a checkpoint boundary, shared by the primary chunk
    and every recovery candidate. None = no checkpoint."""
    if minimum_shift is None:
        return ()
    return tuple(
        token
        for token, event in enumerate(vocab)
        if event.type == "shift" and event.value >= minimum_shift
    )


@dataclass
class _VerificationCandidateState:
    """One verification candidate from generation through final ownership."""

    request: RecoveryCandidateSpec
    origin: float
    tokens: list[int]
    evaluation_tokens: list[int]
    prompt_length: int
    ended: bool
    reached_checkpoint: bool
    handle: object | None
    diagnostic: object
    abort_reason: str | None = None
    guard_summary: MonitorSummary | None = None
    guard_findings: tuple[GuardFinding, ...] = ()
    chunk_quality_safe: bool = False


def _critical_guard_reason(findings: tuple[GuardFinding, ...]) -> str | None:
    return next(
        (
            finding.reason
            for finding in findings
            if finding.status == "critical"
        ),
        None,
    )


def _guard_action_reason(findings: tuple[GuardFinding, ...]) -> str | None:
    return _critical_guard_reason(findings) or next(
        (finding.reason for finding in findings),
        None,
    )


def _evaluate_chunk_quality(
    tokenizer: TokenizerProtocol,
    tokens: list[int],
    source_seek: float,
    ownership_start: float,
    ownership_end: float,
    generated_tokens: int,
    guard_config: GenerationGuardConfig,
    reference: ChunkQualityReference,
    rolling_summary: MonitorSummary | None = None,
) -> tuple[
    ChunkQualityMetrics,
    GuardAction,
    tuple[GuardFinding, ...],
    AdaptiveChunkQualityAssessment,
]:
    metrics = measure_chunk_quality(
        tokenizer,
        tokens,
        source_seek,
        ownership_start,
        ownership_end,
        generated_tokens=generated_tokens,
    )
    assessment = assess_adaptive_chunk_quality(
        metrics,
        reference,
        guard_config.chunk_quality,
    )
    restart_chain_collapse = locate_restart_chain_collapse(
        tokenizer,
        tokens,
        source_seek,
        ownership_start,
        ownership_end,
        prompt_tokens=max(0, len(tokens) - generated_tokens),
    )
    decision = GenerationGuard(guard_config).dispatch(
        ChunkQualityReady(
            metrics,
            reference,
            rolling_summary,
            assessment,
            restart_chain_collapse,
        )
    )
    return metrics, decision.action, decision.findings, assessment


def _guard_requests_recovery(match: OverlapMatch) -> bool:
    config = GenerationGuardConfig(mode="off")
    guard = GenerationGuard(
        config,
        detectors=(OverlapVerificationDetector(config),),
    )
    decision = guard.dispatch(
        VerifyReady(
            score=match.score,
            informative=match.informative,
            comparable_evidence=match.comparable_evidence,
        )
    )
    return any(
        finding.reason == "low_overlap_verification"
        for finding in decision.findings
    )


def _resume_resident_generation(
    handle: object,
) -> Iterator[GenerationControlRequest, object, GenerationResult]:
    """Resume any verified candidate through the shared scheduler contract."""
    result = yield GenerationControlRequest(handle, "resume")
    if not isinstance(result, GenerationResult):
        raise TypeError("resume requires a GenerationResult response")
    return result


def _discard_resident_generation(
    handle: object,
) -> Iterator[GenerationControlRequest, object, GenerationControlResult]:
    """Discard any rejected candidate through the shared scheduler contract."""
    result = yield GenerationControlRequest(handle, "discard")
    if not isinstance(result, GenerationControlResult):
        raise TypeError("discard requires a GenerationControlResult response")
    return result


def _candidate_checkpoint_prefix(
    tokens: list[int],
    vocab: list[Event],
    checkpoint_shift: int,
) -> tuple[list[int], bool]:
    """Return the prefix used for verification and whether it reached it.

    A candidate normally pauses AT the checkpoint (its request carries
    `checkpoint_shift_tokens`), so this is usually the whole row. It still
    reconstructs the boundary from the tokens rather than trusting the pause:
    a row can end before its checkpoint (EOS inside the verification window),
    and scoring must stay independent of anything past the evidence window
    either way.
    """
    for index, token in enumerate(tokens):
        if 0 <= token < len(vocab):
            event = vocab[token]
            if event.type == "shift" and event.value >= checkpoint_shift:
                return tokens[: index + 1], True
    return tokens, False


@dataclass(frozen=True)
class PreparedRecoveryCandidate:
    """Session-ready recovery work and bounded role-specific evidence."""

    request: GenerationRequest
    candidate_name: Literal[
        "shifted_replay",
        "new_seed",
        "secondary_model",
        "bootstrap",
        "fresh_reanchor",
    ]
    model_name: str
    margin_collector: _ShiftMarginCollector | None


@dataclass(frozen=True)
class FirstChunkBootstrapEvent:
    """Compact quality telemetry for an alternate-model first chunk."""

    model_name: str
    shift_decisions: int
    first_shift_margin: float | None
    minimum_shift_margin: float | None


class _ShiftMarginCollector:
    """Retain only bounded chosen-shift logit margins."""

    def __init__(self, vocab: tuple[Event, ...], delegate=None, limit: int = 32):
        self._shift_ids = tuple(
            index for index, event in enumerate(vocab) if event.type == "shift"
        )
        self._delegate = delegate
        self._limit = limit
        self.margins: list[float] = []

    @property
    def graph_capturable(self) -> bool:
        """Whether this collector can be fed after a CUDA Graph replay.

        True when nothing is chained behind it. A delegate is a debug tracer
        wanting hidden states and per-step logits, which a replay does not
        surface -- that row has to decode eagerly.
        """
        return self._delegate is None

    def append(self, hidden, logits, chosen, *, context=()) -> None:
        if self._delegate is not None:
            self._delegate.append(hidden, logits, chosen, context=context)
        if len(self.margins) >= self._limit or not self._shift_ids:
            return
        chosen_id = int(chosen[0].item())
        # One implementation of the margin, shared with the graph path. Two
        # would have to be kept equal by hand, and `torch.equal` in
        # `test_shift_margins` is only a test of what exists today.
        self.observe(chosen_id, float(chosen_token_margins(logits, chosen)[0]))

    def observe(self, chosen_id: int, margin: float) -> None:
        """Record a margin computed elsewhere -- see `chosen_token_margins`.

        The filter and the cap live here so the eager and graph paths cannot
        disagree about *which* steps count.
        """
        if len(self.margins) >= self._limit or not self._shift_ids:
            return
        if chosen_id not in self._shift_ids:
            return
        self.margins.append(margin)


def prepare_recovery_candidates(
    model: ModelProtocol,
    requests: tuple[RecoveryCandidateSpec, ...],
) -> tuple[PreparedRecoveryCandidate, ...]:
    """Validate specs against the executing model and bind margin collection.

    The generation request arrives fully built (the spec's whole point); what
    happens here is what genuinely needs the executing model: the tokenizer
    compatibility gate -- a recovery model must share the primary's vocab,
    eos and frame rate -- and the name of the model that actually ran it."""
    if not requests:
        return ()
    tokenizer = model._tokenizer
    first = requests[0].request
    compatibility = (
        first.max_gen_len,
        first.temperature,
        first.cfg_coef,
    )
    for spec in requests:
        request = spec.request
        if (
            tokenizer.eos_id != request.eos_id
            or tokenizer.frame_rate != spec.expected_frame_rate
            or tuple(tokenizer._vocab) != request.guard_vocab
        ):
            raise ValueError("recovery tokenizer is incompatible with primary")
        if (
            request.max_gen_len,
            request.temperature,
            request.cfg_coef,
        ) != compatibility:
            raise ValueError(
                "recovery batch rows require matching generation settings"
            )
    model_name = getattr(model, "_model_name", "secondary")
    prepared = []
    for spec in requests:
        collector = (
            _ShiftMarginCollector(
                spec.request.guard_vocab,
                spec.request.trace_collector,
            )
            if spec.collect_shift_margins
            else None
        )
        prepared.append(
            PreparedRecoveryCandidate(
                request=(
                    replace(spec.request, trace_collector=collector)
                    if collector is not None
                    else spec.request
                ),
                candidate_name=spec.candidate_name,
                model_name=model_name,
                margin_collector=collector,
            )
        )
    return tuple(prepared)


def complete_recovery_candidate(
    prepared: PreparedRecoveryCandidate,
    result: GenerationResult,
) -> RecoveryCandidateResult:
    """Adapt one shared-runtime result to the quality orchestration contract."""
    return RecoveryCandidateResult(
        candidate_name=prepared.candidate_name,
        tokens=result.tokens,
        emitted_eos=result.emitted_eos,
        model_name=prepared.model_name,
        shift_margins=(
            tuple(prepared.margin_collector.margins)
            if prepared.margin_collector is not None
            else ()
        ),
        resident_handle=result.resident_handle,
        guard_action=result.guard_action,
        guard_findings=result.guard_findings,
        guard_summary=result.guard_summary,
    )


def forcing_stream(
    model: ModelProtocol,
    all_conditions: list[ConditioningAttributes] | None,
    seek_times: list[float],
    wav: torch.Tensor,
    instrument_group: str | None,
    max_gen_len: int,
    use_sampling: bool,
    temperature: float,
    cfg_coef: float,
    no_eos_is_ok: bool,
    beam_size: int,
    forbidden_tokens: torch.Tensor | None,
    overlap_window: OverlapWindow | None,
    anomaly_detection: bool,
    overlap_detection: bool,
    recovery: bool,
    recovery_shift_seconds: float,
    recovery_selection: RecoverySelection,
    secondary_recovery_shifted: bool,
    anomaly_config: AnomalyMonitorConfig,
    recovery_seed: int,
    stdout_logger: logging.Logger | None,
    stderr_logger: logging.Logger | None,
    checkpoint_generation: bool = False,
    trace_collector: ModelTraceCollector | None = None,
    trace_context_prefix: tuple[object, ...] = (),
    first_chunk_model: str | None = None,
    fresh_reanchor_model: str | None = None,
    chunk_quality_history: ChunkQualityHistory | None = None,
    hard_pitch_envelope: HardPitchEnvelopeConfig | None = None,
    overlap_probe_enabled: bool = False,
    fresh_reanchor: bool = True,
    # Which second attempt a triggered chunk gets. Both keep the shifted
    # replay beside them; they differ in what the *other* candidate varies:
    #   secondary_model -- a larger model, same sampling
    #   new_seed        -- the same model, resampled from a derived seed
    # The two are not interchangeable. Every other candidate here (bootstrap,
    # secondary_model, fresh_reanchor) escalates model size, so `new_seed` is
    # the only one that answers "this decode was unlucky" rather than "this
    # model is not strong enough", and it costs no second model.
    secondary_candidate: SecondaryCandidate = "secondary_model",
) -> Iterator[
    int
    | ChunkBoundary
    | ProgressEvent
    | OverlapDiagnosticsEvent
    | GenerationAnomalyEvent
    | RecoveryDiagnosticsEvent
    | GenerationRequest
    | GenerationControlRequest
    | RecoveryCandidateSpec
]:
    """Generate sequential chunks with tie/overlap teacher forcing."""
    if recovery_selection not in {"checkpoint", "completed_quality"}:
        raise ValueError("invalid recovery selection policy")
    recovery_enabled = recovery
    eos_id = model._tokenizer.eos_id
    primary_guard = GenerationGuardConfig(
        mode="recovery" if anomaly_detection else "off",
        role="primary",
        anomaly=anomaly_config,
        allow_periodic_recovery=False,
        hard_pitch_envelope=hard_pitch_envelope,
        enforce_completed_quality=(recovery_selection == "completed_quality"),
    )
    recovery_guard = GenerationGuardConfig(
        mode="recovery" if anomaly_detection else "off",
        role="recovery",
        anomaly=anomaly_config,
        hard_pitch_envelope=hard_pitch_envelope,
        enforce_completed_quality=(recovery_selection == "completed_quality"),
    )
    fresh_guard = replace(recovery_guard, role="fresh")
    # Live rolling detection follows the user switch. Terminal hard validity
    # remains an unconditional output contract. Both use the same detectors
    # and role transitions; adaptive findings stay advisory until the explicit
    # completed-quality policy adds a composite continuity rule.
    primary_quality_guard = replace(primary_guard, mode="recovery")
    recovery_quality_guard = replace(recovery_guard, mode="recovery")
    if chunk_quality_history is None:
        chunk_quality_history = ChunkQualityHistory(
            primary_guard.chunk_quality.history_size
        )
    tracker = OpenNoteTracker(model._tokenizer._vocab, model._tokenizer.frame_rate)
    previous_tokens: list[int] = []
    previous_seek = 0.0
    total = len(seek_times)
    audio_duration = wav.shape[-1] / _SAMPLE_RATE
    stride = seek_times[1] - seek_times[0] if total > 1 else _SEGMENT_DURATION
    # condition_origin and ownership_start normally coincide. A reflow chunk is
    # the sole exception: its condition starts one overlap before ownership.
    schedule: list[tuple[ConditioningAttributes | None, float, float]] = [
        (None, seek, seek) for seek in seek_times
    ]
    if all_conditions is not None:
        schedule = [
            (condition, seek, seek)
            for condition, seek in zip(all_conditions, seek_times, strict=True)
        ]
    chunk_index = 0
    segment_samples = int(_SEGMENT_DURATION * _SAMPLE_RATE)
    progress_completed = 0
    while chunk_index < len(schedule):
        condition, condition_origin, ownership_start = schedule[chunk_index]
        selected_reference_tokens: list[int] | None = None
        selected_reference_origin: float | None = None
        selected_reference_end: float | None = None
        force_reflow = False
        safe_frontier_source = None
        safe_frontier_time: float | None = None
        next_seek = (
            schedule[chunk_index + 1][2]
            if chunk_index + 1 < len(schedule)
            else None
        )
        boundary = ChunkBoundary(ownership_start, next_seek)

        resolved_window = overlap_window
        if chunk_index > 0 and overlap_window is not None and overlap_detection:
            if not overlap_runtime.has_continuity_evidence(
                model._tokenizer,
                previous_tokens,
                previous_seek,
                condition_origin,
                overlap_window,
            ):
                resolved_window = None
            else:
                resolved_window = overlap_runtime.resolve_overlap_window(
                    model._tokenizer,
                    previous_tokens,
                    previous_seek,
                    condition_origin,
                    overlap_window,
                )
                extension = (
                    resolved_window.verification_seconds
                    - overlap_window.verification_seconds
                )
                desired_origin = max(0.0, condition_origin - extension)
                if abs(desired_origin - condition_origin) > 1e-9:
                    condition_origin = desired_origin
                    condition = None
                    replacement: list[
                        tuple[ConditioningAttributes | None, float, float]
                    ] = []
                    owner = condition_origin + stride
                    while owner < audio_duration:
                        replacement.append((None, owner, owner))
                        owner += stride
                    schedule[chunk_index + 1 :] = replacement
                    next_seek = replacement[0][2] if replacement else None
                    boundary = ChunkBoundary(ownership_start, next_seek)

        tracker.feed(boundary)

        seek = condition_origin
        if condition is None:
            segment_samples = int(_SEGMENT_DURATION * _SAMPLE_RATE)
            start = round(condition_origin * _SAMPLE_RATE)
            chunk = wav[:, start : start + segment_samples]
            if chunk.shape[-1] < segment_samples:
                chunk = F.pad(chunk, (0, segment_samples - chunk.shape[-1]))
            condition = model._build_conditions(chunk, instrument_group)[0]

        prompt_plan = overlap_runtime.PromptPlan((), 0, 0, 0, 0)
        prompt_ids: list[int] = []
        if chunk_index > 0:
            prompt_plan = overlap_runtime.forcing_prompt_plan(
                model._tokenizer,
                tracker.open_keys(),
                previous_tokens,
                previous_seek,
                seek,
                resolved_window,
                max_gen_len,
                chunk_index,
                stderr_logger,
            )
            prompt_ids = list(prompt_plan.prompt_ids)
        verification_start = (
            seek + resolved_window.replay_seconds
            if resolved_window is not None
            else seek
        )
        verification_end = (
            seek + resolved_window.overlap_seconds
            if resolved_window is not None
            else seek
        )
        verification_end_shift = round(
            (verification_end - seek) * model._tokenizer.frame_rate
        )

        chunk_tokens: list[int] = []
        ended = False
        primary_match: OverlapMatch | None = None
        trigger_reason: str | None = None
        primary_result: GenerationResult | None = None
        primary_guard_summary: MonitorSummary | None = None
        primary_guard_findings: tuple[GuardFinding, ...] = ()
        deferred_primary_handle: object | None = None
        if chunk_index == 0 and first_chunk_model is not None:
            vocab = tuple(model._tokenizer._vocab)
            bootstrap_result = yield RecoveryCandidateSpec(
                candidate_name="bootstrap",
                request=GenerationRequest(
                    condition=condition,
                    prompt_ids=tuple(prompt_ids),
                    forbidden_token_ids=(
                        tuple(int(token) for token in forbidden_tokens.tolist())
                        if forbidden_tokens is not None
                        else ()
                    ),
                    max_gen_len=max_gen_len,
                    use_sampling=False,
                    temperature=temperature,
                    cfg_coef=cfg_coef,
                    eos_id=eos_id,
                    prompt_bucket=prompt_plan.padded_tokens,
                    trace_collector=trace_collector,
                    trace_context=(
                        *trace_context_prefix, "bootstrap", chunk_index
                    ),
                    guard_config=primary_guard,
                    guard_vocab=vocab,
                    temporal_grammar=TemporalGrammarConfig.from_vocab(
                        vocab, tuple(prompt_ids), ()
                    ),
                    recovery_group_claim=(
                        RecoveryGroupClaim(
                            (
                                RecoveryClaimMember(
                                    "primary",
                                    getattr(model, "_model_name", None),
                                ),
                                RecoveryClaimMember("recovery"),
                            )
                        )
                        if recovery_enabled
                        else None
                    ),
                ),
                expected_frame_rate=model._tokenizer.frame_rate,
                target_model=first_chunk_model,
                model_role="recovery",
                collect_shift_margins=True,
            )
            if not isinstance(bootstrap_result, RecoveryCandidateResult):
                raise TypeError(
                    "bootstrap requires a compatible recovery result"
                )
            yield FirstChunkBootstrapEvent(
                model_name=bootstrap_result.model_name,
                shift_decisions=len(bootstrap_result.shift_margins),
                first_shift_margin=(
                    bootstrap_result.shift_margins[0]
                    if bootstrap_result.shift_margins
                    else None
                ),
                minimum_shift_margin=(
                    min(bootstrap_result.shift_margins)
                    if bootstrap_result.shift_margins
                    else None
                ),
            )
            primary_steps = iter(
                [torch.tensor([token]) for token in bootstrap_result.tokens]
                + (
                    [torch.tensor([eos_id])]
                    if bootstrap_result.emitted_eos
                    else []
                )
            )
            primary_guard_summary = bootstrap_result.guard_summary
            primary_guard_findings = bootstrap_result.guard_findings
            deferred_primary_handle = bootstrap_result.resident_handle
        else:
            checkpoint_token_ids = (
                checkpoint_shift_tokens(
                    model._tokenizer._vocab, verification_end_shift
                )
                if checkpoint_generation
                and recovery_enabled
                and overlap_detection
                and chunk_index > 0
                and resolved_window is not None
                else ()
            )
            primary_result = yield GenerationRequest(
                condition=condition,
                prompt_ids=tuple(prompt_ids),
                forbidden_token_ids=(
                    tuple(int(token) for token in forbidden_tokens.tolist())
                    if forbidden_tokens is not None
                    else ()
                ),
                max_gen_len=max_gen_len,
                use_sampling=use_sampling,
                temperature=temperature,
                cfg_coef=cfg_coef,
                eos_id=eos_id,
                beam_size=beam_size,
                prompt_bucket=prompt_plan.padded_tokens,
                decode_token_estimate=(
                    len(previous_tokens) if previous_tokens else None
                ),
                checkpoint_token_ids=checkpoint_token_ids,
                trace_collector=trace_collector,
                trace_context=(*trace_context_prefix, "primary", chunk_index),
                sampling_seed=None,
                guard_config=primary_guard,
                guard_vocab=tuple(model._tokenizer._vocab),
                temporal_grammar=TemporalGrammarConfig.from_vocab(
                    tuple(model._tokenizer._vocab),
                    tuple(prompt_ids),
                    checkpoint_token_ids,
                ),
                recovery_group_claim=(
                    RecoveryGroupClaim(
                        (
                            RecoveryClaimMember("primary"),
                            RecoveryClaimMember("recovery"),
                        )
                    )
                    if recovery_enabled
                    else None
                ),
            )
            if not isinstance(primary_result, GenerationResult):
                raise TypeError(
                    "generation request requires a GenerationResult response"
                )
            primary_steps = iter(
                [torch.tensor([token]) for token in primary_result.tokens]
                + (
                    [torch.tensor([eos_id])]
                    if primary_result.emitted_eos
                    else []
                )
            )
            primary_guard_summary = primary_result.guard_summary
            primary_guard_findings = primary_result.guard_findings
            deferred_primary_handle = primary_result.resident_handle
        primary_generation = MonitoredGeneration(
            primary_steps,
            eos_id=eos_id,
            vocab=model._tokenizer._vocab,
            anomaly_detection=anomaly_detection,
            config=anomaly_config,
            prompt_tokens=len(prompt_ids),
            allow_periodic_recovery=False,
            summary=primary_guard_summary,
        )
        for monitored in primary_generation:
            token = monitored.token
            chunk_tokens.append(token)
            event = monitored.event
            if monitored.decision.action == "interrupt_to_recovery":
                trigger_reason = _critical_guard_reason(
                    monitored.decision.findings
                )
                break
            if (
                overlap_detection
                and recovery_enabled
                and chunk_index > 0
                and resolved_window is not None
                and primary_match is None
                and event.type == "shift"
                and event.value >= verification_end_shift
            ):
                primary_match = overlap_runtime.match_verification(
                    model._tokenizer,
                    previous_tokens,
                    previous_seek,
                    chunk_tokens,
                    seek,
                    verification_start,
                    verification_end,
                )
                if _guard_requests_recovery(primary_match):
                    trigger_reason = "low_overlap_verification"
                    break
        ended = primary_generation.ended
        deferred_guard_reason = _critical_guard_reason(primary_guard_findings)
        if deferred_guard_reason is not None:
            trigger_reason = deferred_guard_reason
        primary_handle = deferred_primary_handle
        if primary_handle is not None and trigger_reason is None:
            continuation_start = len(chunk_tokens)
            continuation = yield from _resume_resident_generation(
                primary_handle
            )
            continuation_tokens = continuation.tokens[continuation_start:]
            primary_generation.set_summary(continuation.guard_summary)
            primary_generation.extend(
                [torch.tensor([token]) for token in continuation_tokens]
                + (
                    [torch.tensor([eos_id])]
                    if continuation.emitted_eos
                    else []
                )
            )
            primary_handle = continuation.resident_handle
            for monitored in primary_generation:
                chunk_tokens.append(monitored.token)
                if monitored.decision.action == "interrupt_to_recovery":
                    trigger_reason = _critical_guard_reason(
                        monitored.decision.findings
                    )
                    break
            ended = primary_generation.ended
            continuation_guard_reason = _critical_guard_reason(
                continuation.guard_findings
            )
            if continuation_guard_reason is not None:
                trigger_reason = continuation_guard_reason
            primary_guard_findings = continuation.guard_findings

        primary_summary = primary_generation.finalize()
        if not ended and trigger_reason is None:
            # A non-EOS primary is incomplete regardless of optional detectors.
            # Recover it when possible; otherwise the safety path discards it.
            trigger_reason = "missing_eos_at_max_length"
        primary_chunk_quality: ChunkQualityMetrics | None = None
        primary_chunk_action: GuardAction = "continue"
        primary_chunk_findings: tuple[GuardFinding, ...] = ()
        primary_chunk_assessment: AdaptiveChunkQualityAssessment | None = None
        chunk_quality_reference_samples = len(
            chunk_quality_history.reference().samples
        )
        primary_quality_end = min(
            next_seek or audio_duration,
            seek + _SEGMENT_DURATION,
            audio_duration,
        )
        (
            primary_chunk_quality,
            primary_chunk_action,
            primary_chunk_findings,
            primary_chunk_assessment,
        ) = _evaluate_chunk_quality(
            model._tokenizer,
            chunk_tokens,
            seek,
            ownership_start,
            primary_quality_end,
            max(0, len(chunk_tokens) - len(prompt_ids)),
            primary_quality_guard,
            chunk_quality_history.reference(),
            primary_summary,
        )
        if trigger_reason is None and primary_chunk_action == "request_recovery":
            trigger_reason = _guard_action_reason(primary_chunk_findings)

        selected_tokens = chunk_tokens
        accepted_chunk_quality = (
            primary_chunk_quality
            if _critical_guard_reason(primary_chunk_findings) is None
            else None
        )
        if primary_match is None:
            primary_match = (
                overlap_runtime.match_verification(
                    model._tokenizer,
                    previous_tokens,
                    previous_seek,
                    chunk_tokens,
                    seek,
                    verification_start,
                    verification_end,
                )
                if chunk_index > 0 and resolved_window is not None
                else match_note_events([], [])
            )
        if (
            trigger_reason is None
            and recovery_enabled
            and overlap_detection
            and chunk_index > 0
            and resolved_window is not None
            and _guard_requests_recovery(primary_match)
        ):
            trigger_reason = "low_overlap_verification"
        primary_diagnostic = None
        if trigger_reason is not None:
            primary_hard_reason = (
                deferred_guard_reason
                or _critical_guard_reason(primary_chunk_findings)
            )
            primary_diagnostic = evaluate_candidate(
                "primary",
                chunk_tokens,
                len(prompt_ids),
                ended,
                model._tokenizer._vocab,
                verification_match=primary_match,
                allow_missing_eos=(
                    primary_handle is not None and primary_hard_reason is None
                ),
                forced_invalid_reason=primary_hard_reason,
                model_name=getattr(model, "_model_name", None),
                sampling=use_sampling,
                condition_seek_time=seek,
                chunk_quality=(
                    None if primary_handle is not None else primary_chunk_quality
                ),
                chunk_guard_reasons=(
                    ()
                    if primary_handle is not None
                    else tuple(
                        finding.reason for finding in primary_chunk_findings
                    )
                ),
                guard_findings=(
                    primary_guard_findings
                    if primary_handle is not None
                    else (*primary_guard_findings, *primary_chunk_findings)
                ),
                chunk_quality_reference_samples=(
                    chunk_quality_reference_samples
                ),
                chunk_quality_assessment=(
                    None if primary_handle is not None else primary_chunk_assessment
                ),
            )
        recovery_event: RecoveryDiagnosticsEvent | None = None
        if recovery_enabled and trigger_reason is not None:
            seed = derive_recovery_seed(recovery_seed, chunk_index, 0)
            shifted_seek = max(0.0, seek - recovery_shift_seconds)
            shifted_start = round(shifted_seek * _SAMPLE_RATE)
            segment_samples = int(_SEGMENT_DURATION * _SAMPLE_RATE)
            shifted_chunk = wav[:, shifted_start : shifted_start + segment_samples]
            if shifted_chunk.shape[-1] < segment_samples:
                shifted_chunk = F.pad(
                    shifted_chunk,
                    (0, segment_samples - shifted_chunk.shape[-1]),
                )
            shifted_condition = model._build_conditions(
                shifted_chunk,
                instrument_group,
            )[0]
            shifted_open = overlap_runtime.open_keys_at(
                model._tokenizer,
                previous_tokens,
                previous_seek,
                shifted_seek,
            )
            extended_replay = overlap_runtime.note_events_in_range(
                model._tokenizer,
                previous_tokens,
                previous_seek,
                shifted_seek,
                verification_start,
            )
            shifted_prompt_plan = overlap_runtime.plan_prompt(
                model._tokenizer,
                shifted_open,
                extended_replay,
                shifted_seek,
                max_gen_len,
            )
            shifted_prompt_ids = list(shifted_prompt_plan.prompt_ids)
            secondary_condition = (
                shifted_condition if secondary_recovery_shifted else condition
            )
            secondary_name = secondary_candidate
            secondary_sampling = secondary_name == "new_seed"
            secondary_seek = shifted_seek if secondary_recovery_shifted else seek
            secondary_prompt_ids = (
                shifted_prompt_ids if secondary_recovery_shifted else prompt_ids
            )
            secondary_prompt_bucket = (
                shifted_prompt_plan.padded_tokens
                if secondary_recovery_shifted
                else prompt_plan.padded_tokens
            )
            vocab = tuple(model._tokenizer._vocab)
            forbidden_ids = (
                tuple(int(token) for token in forbidden_tokens.tolist())
                if forbidden_tokens is not None
                else ()
            )
            secondary_checkpoint = checkpoint_shift_tokens(
                vocab,
                round(
                    (verification_end - secondary_seek)
                    * model._tokenizer.frame_rate
                ),
            )
            secondary_request = RecoveryCandidateSpec(
                candidate_name=secondary_name,
                request=GenerationRequest(
                    condition=secondary_condition,
                    prompt_ids=tuple(secondary_prompt_ids),
                    forbidden_token_ids=forbidden_ids,
                    max_gen_len=max_gen_len,
                    use_sampling=secondary_sampling,
                    temperature=temperature,
                    cfg_coef=cfg_coef,
                    eos_id=eos_id,
                    prompt_bucket=secondary_prompt_bucket,
                    sampling_seed=seed if secondary_sampling else None,
                    trace_collector=trace_collector,
                    trace_context=(
                        *trace_context_prefix, secondary_name, chunk_index
                    ),
                    checkpoint_token_ids=secondary_checkpoint,
                    guard_config=recovery_guard,
                    guard_vocab=vocab,
                    temporal_grammar=TemporalGrammarConfig.from_vocab(
                        vocab, tuple(secondary_prompt_ids), secondary_checkpoint
                    ),
                ),
                expected_frame_rate=model._tokenizer.frame_rate,
                target_model=(
                    getattr(model, "_model_name", None)
                    if secondary_sampling
                    else None
                ),
                model_role="primary" if secondary_sampling else "recovery",
            )
            shifted_checkpoint = checkpoint_shift_tokens(
                vocab,
                round(
                    (verification_end - shifted_seek)
                    * model._tokenizer.frame_rate
                ),
            )
            shifted_request = RecoveryCandidateSpec(
                candidate_name="shifted_replay",
                request=GenerationRequest(
                    condition=shifted_condition,
                    prompt_ids=tuple(shifted_prompt_ids),
                    forbidden_token_ids=forbidden_ids,
                    max_gen_len=max_gen_len,
                    use_sampling=True,
                    temperature=temperature,
                    cfg_coef=cfg_coef,
                    eos_id=eos_id,
                    prompt_bucket=shifted_prompt_plan.padded_tokens,
                    sampling_seed=seed,
                    trace_collector=trace_collector,
                    trace_context=(
                        *trace_context_prefix, "shifted_replay", chunk_index
                    ),
                    checkpoint_token_ids=shifted_checkpoint,
                    guard_config=recovery_guard,
                    guard_vocab=vocab,
                    temporal_grammar=TemporalGrammarConfig.from_vocab(
                        vocab, tuple(shifted_prompt_ids), shifted_checkpoint
                    ),
                ),
                expected_frame_rate=model._tokenizer.frame_rate,
                target_model=getattr(model, "_model_name", None),
                model_role="primary",
            )

            recovery_results = yield RecoveryCandidateGroupRequest(
                (shifted_request, secondary_request),
                parent_resident_handle=primary_handle,
            )
            if (
                not isinstance(recovery_results, tuple)
                or len(recovery_results) != 2
                or not all(
                    isinstance(result, RecoveryCandidateResult)
                    for result in recovery_results
                )
            ):
                raise TypeError(
                    "recovery candidate group requires compatible results"
                )
            shifted_result, external_secondary = recovery_results

            candidate_results = {
                "shifted_replay": shifted_result,
                secondary_name: external_secondary,
            }
            candidate_requests = {
                "shifted_replay": shifted_request,
                secondary_name: secondary_request,
            }
            candidate_states: dict[str, _VerificationCandidateState] = {}
            for name, result in candidate_results.items():
                request = candidate_requests[name]
                full_tokens = list(result.tokens)
                origin = (
                    shifted_seek if name == "shifted_replay" else secondary_seek
                )
                prompt_length = len(request.prompt_ids)
                checkpoint_shift = round(
                    (verification_end - origin) * model._tokenizer.frame_rate
                )
                evaluation_tokens, reached_checkpoint = (
                    _candidate_checkpoint_prefix(
                        full_tokens,
                        model._tokenizer._vocab,
                        checkpoint_shift,
                    )
                )
                abort_reason = _critical_guard_reason(result.guard_findings)
                diagnostic = evaluate_candidate(
                    name,
                    evaluation_tokens,
                    prompt_length,
                    result.emitted_eos if not reached_checkpoint else False,
                    model._tokenizer._vocab,
                    verification_match=overlap_runtime.match_verification(
                        model._tokenizer,
                        previous_tokens,
                        previous_seek,
                        evaluation_tokens,
                        origin,
                        verification_start,
                        verification_end,
                    ),
                    allow_missing_eos=reached_checkpoint,
                    forced_invalid_reason=abort_reason,
                    model_name=result.model_name,
                    sampling=request.use_sampling,
                    condition_seek_time=origin,
                    guard_findings=result.guard_findings,
                )
                candidate_states[name] = _VerificationCandidateState(
                    request=request,
                    origin=origin,
                    tokens=full_tokens,
                    evaluation_tokens=evaluation_tokens,
                    prompt_length=prompt_length,
                    ended=result.emitted_eos,
                    reached_checkpoint=reached_checkpoint,
                    handle=result.resident_handle,
                    diagnostic=diagnostic,
                    abort_reason=abort_reason,
                    guard_summary=result.guard_summary,
                    guard_findings=result.guard_findings,
                )

            assert primary_diagnostic is not None
            checkpoint_selection = select_checkpoint_candidate(
                primary_diagnostic,
                (
                    candidate_states["shifted_replay"].diagnostic,
                    candidate_states[secondary_name].diagnostic,
                ),
            )
            checkpoint_selected_name = (
                checkpoint_selection.name
                if checkpoint_selection is not None
                else None
            )
            # The checkpoint winner resumes first; the losers' paused KV is
            # kept until every attempt is over (the discard block below), so a
            # winner whose CONTINUATION fails terminally can be replaced by
            # the next checkpoint-eligible candidate instead of ending as a
            # frontier prefix or a discarded chunk. Owner-directed 2026-08-29;
            # measured pool: 57 discarded + 23 frontier of 2053 decisions.
            if (
                recovery_selection == "checkpoint"
                and checkpoint_selected_name == "primary"
                and primary_handle is not None
            ):
                continuation = yield from _resume_resident_generation(
                    primary_handle
                )
                chunk_tokens = list(continuation.tokens)
                ended = continuation.emitted_eos
                primary_handle = continuation.resident_handle
                primary_summary = continuation.guard_summary or primary_summary
                primary_hard_reason = _critical_guard_reason(
                    continuation.guard_findings
                )
                (
                    primary_chunk_quality,
                    primary_chunk_action,
                    primary_chunk_findings,
                    primary_chunk_assessment,
                ) = _evaluate_chunk_quality(
                    model._tokenizer,
                    chunk_tokens,
                    seek,
                    ownership_start,
                    primary_quality_end,
                    max(0, len(chunk_tokens) - len(prompt_ids)),
                    primary_quality_guard,
                    chunk_quality_history.reference(),
                    primary_summary,
                )
                primary_diagnostic = evaluate_candidate(
                    "primary",
                    chunk_tokens,
                    len(prompt_ids),
                    ended,
                    model._tokenizer._vocab,
                    verification_match=primary_match,
                    forced_invalid_reason=primary_hard_reason,
                    model_name=getattr(model, "_model_name", None),
                    sampling=use_sampling,
                    condition_seek_time=seek,
                    chunk_quality=primary_chunk_quality,
                    chunk_guard_reasons=tuple(
                        finding.reason for finding in primary_chunk_findings
                    ),
                    guard_findings=(
                        *primary_guard_findings,
                        *primary_chunk_findings,
                    ),
                    chunk_quality_reference_samples=(
                        chunk_quality_reference_samples
                    ),
                    chunk_quality_assessment=primary_chunk_assessment,
                )
            attempted_names: set[str] = set()

            def _fallback_candidates(
                *,
                # Bound at def time, re-bound each chunk iteration -- the
                # closure-over-loop-variable hazard (B023) made explicit.
                _states=candidate_states,
                _attempted=attempted_names,
                _secondary=secondary_name,
            ) -> list[str]:
                """Unattempted candidates still standing, best score first.

                Checkpoint diagnostics only: an attempted candidate's
                diagnostic has been replaced by its (failed) completed one, so
                it can never re-enter."""
                return sorted(
                    (
                        name
                        for name in ("shifted_replay", _secondary)
                        if name not in _attempted
                        and _states[name].diagnostic.structural_valid
                        and _states[name].diagnostic.reference_eligible
                        and _states[name].diagnostic
                        .verification_match.score is not None
                    ),
                    key=lambda name: _states[name]
                    .diagnostic.verification_match.score,
                    reverse=True,
                )

            if recovery_selection == "checkpoint":
                if checkpoint_selected_name in candidate_states:
                    candidate_names = [checkpoint_selected_name]
                elif checkpoint_selected_name == "primary" and not (
                    primary_diagnostic.structural_valid
                    and primary_diagnostic.reference_eligible
                ):
                    # The winner's own continuation failed; the losers are
                    # still paused at their checkpoints.
                    candidate_names = _fallback_candidates()[:1]
                else:
                    candidate_names = []
            else:
                candidate_names = [secondary_name]
            for name in candidate_names:
                checkpoint_candidate = candidate_states[name].diagnostic
                if not checkpoint_candidate.structural_valid:
                    continue
                attempted_names.add(name)
                state = candidate_states[name]
                if state.handle is not None:
                    continuation = yield from _resume_resident_generation(
                        state.handle
                    )
                    state.tokens = list(continuation.tokens)
                    state.ended = continuation.emitted_eos
                    state.handle = continuation.resident_handle
                    state.abort_reason = (
                        _critical_guard_reason(continuation.guard_findings)
                        or state.abort_reason
                    )
                    state.guard_summary = (
                        continuation.guard_summary or state.guard_summary
                    )
                    state.guard_findings = continuation.guard_findings
                final_diagnostic = evaluate_candidate(
                    name,
                    state.tokens,
                    state.prompt_length,
                    state.ended,
                    model._tokenizer._vocab,
                    verification_match=checkpoint_candidate.verification_match,
                    forced_invalid_reason=state.abort_reason,
                    model_name=checkpoint_candidate.model_name,
                    sampling=checkpoint_candidate.sampling,
                    condition_seek_time=checkpoint_candidate.condition_seek_time,
                    guard_findings=state.guard_findings,
                )
                if final_diagnostic.structural_valid:
                    candidate_quality_end = min(
                        next_seek or audio_duration,
                        state.origin + _SEGMENT_DURATION,
                        audio_duration,
                    )
                    (
                        candidate_quality,
                        candidate_action,
                        candidate_findings,
                        candidate_assessment,
                    ) = _evaluate_chunk_quality(
                        model._tokenizer,
                        state.tokens,
                        state.origin,
                        ownership_start,
                        candidate_quality_end,
                        max(0, len(state.tokens) - state.prompt_length),
                        recovery_quality_guard,
                        chunk_quality_history.reference(),
                        state.guard_summary,
                    )
                    semantic_reason = (
                        _critical_guard_reason(candidate_findings)
                        if candidate_action == "reject_candidate"
                        else None
                    )
                    state.chunk_quality_safe = (
                        _critical_guard_reason(candidate_findings) is None
                    )
                    final_diagnostic = evaluate_candidate(
                        name,
                        state.tokens,
                        state.prompt_length,
                        state.ended,
                        model._tokenizer._vocab,
                        verification_match=(
                            checkpoint_candidate.verification_match
                        ),
                        forced_invalid_reason=semantic_reason,
                        model_name=checkpoint_candidate.model_name,
                        sampling=checkpoint_candidate.sampling,
                        condition_seek_time=(
                            checkpoint_candidate.condition_seek_time
                        ),
                        chunk_quality=candidate_quality,
                        chunk_guard_reasons=tuple(
                            finding.reason for finding in candidate_findings
                        ),
                        guard_findings=(
                            *state.guard_findings,
                            *candidate_findings,
                        ),
                        chunk_quality_reference_samples=len(
                            chunk_quality_history.reference().samples
                        ),
                        chunk_quality_assessment=candidate_assessment,
                    )
                state.diagnostic = final_diagnostic
                state.chunk_quality_safe = (
                    final_diagnostic.structural_valid
                    and final_diagnostic.reference_eligible
                )
                if (
                    recovery_selection == "completed_quality"
                    and name == secondary_name
                    and not state.chunk_quality_safe
                ):
                    candidate_names.append("shifted_replay")
                if (
                    recovery_selection == "checkpoint"
                    and not state.chunk_quality_safe
                ):
                    # Same shape as the completed-quality clause above: the
                    # attempted candidate's completed chunk failed, the next
                    # eligible one is still paused at its checkpoint -- try it
                    # before settling for a frontier prefix or a discard.
                    candidate_names.extend(_fallback_candidates()[:1])

            for name in ("shifted_replay", secondary_name):
                state = candidate_states[name]
                if state.handle is not None:
                    yield from _discard_resident_generation(state.handle)
                    state.handle = None
                if (
                    name not in attempted_names
                    and state.reached_checkpoint
                    and state.abort_reason is None
                ):
                    checkpoint = state.diagnostic
                    state.diagnostic = evaluate_candidate(
                        name,
                        state.evaluation_tokens,
                        state.prompt_length,
                        False,
                        model._tokenizer._vocab,
                        verification_match=checkpoint.verification_match,
                        forced_invalid_reason="not_resumed_after_checkpoint",
                        model_name=checkpoint.model_name,
                        sampling=checkpoint.sampling,
                        condition_seek_time=checkpoint.condition_seek_time,
                    )
            if primary_handle is not None:
                yield from _discard_resident_generation(primary_handle)
                primary_handle = None
            primary_generation.close()

            candidate_diagnostics = [
                primary_diagnostic,
                candidate_states["shifted_replay"].diagnostic,
                candidate_states[secondary_name].diagnostic,
            ]
            if recovery_selection == "checkpoint":
                # The winner when its completed chunk survived; otherwise the
                # first fallback whose did. `candidate_names` is attempt order,
                # which is preference order by construction.
                if (
                    checkpoint_selected_name == "primary"
                    and primary_diagnostic.structural_valid
                    and primary_diagnostic.reference_eligible
                ):
                    selected_name = "primary"
                else:
                    selected_name = next(
                        (
                            name
                            for name in candidate_names
                            if name in attempted_names
                            and candidate_states[name].diagnostic.structural_valid
                            and candidate_states[name].diagnostic.reference_eligible
                        ),
                        None,
                    )
            else:
                recovery_order = recovery_candidate_order(
                    primary_diagnostic,
                    candidate_states["shifted_replay"].diagnostic,
                    candidate_states[secondary_name].diagnostic,
                )
                selected_name = recovery_order[0].name if recovery_order else None
            selection_reason = {
                "secondary_model": "secondary_recovery_reference_eligible",
                "new_seed": "secondary_recovery_reference_eligible",
                "shifted_replay": "shifted_replay_reference_eligible",
                "primary": "primary_reference_eligible",
            }.get(selected_name, "fresh_reanchor_required")

            if selected_name is None:
                frontier_names = (
                    # Every completed attempt is frontier material, the winner
                    # first -- a fallback that also failed may still hold a
                    # longer clean prefix than the winner's.
                    list(
                        dict.fromkeys(
                            [checkpoint_selected_name, *candidate_names]
                        )
                    )
                    if recovery_selection == "checkpoint"
                    else list(candidate_names)
                )
                for frontier_name in frontier_names:
                    if frontier_name == "primary":
                        frontier_diagnostic = primary_diagnostic
                        frontier_tokens = chunk_tokens
                        frontier_prompt_length = len(prompt_ids)
                        frontier_origin = seek
                    elif frontier_name in candidate_states:
                        frontier_state = candidate_states[frontier_name]
                        frontier_diagnostic = frontier_state.diagnostic
                        frontier_tokens = frontier_state.tokens
                        frontier_prompt_length = frontier_state.prompt_length
                        frontier_origin = frontier_state.origin
                    else:
                        continue
                    frontier = candidate_safe_frontier(frontier_diagnostic)
                    if frontier is None or frontier.shift_value is None:
                        continue
                    candidate_frontier_time = (
                        frontier_origin
                        + frontier.shift_value / model._tokenizer.frame_rate
                    )
                    candidate_end = min(
                        next_seek or audio_duration,
                        frontier_origin + _SEGMENT_DURATION,
                        audio_duration,
                    )
                    if (
                        candidate_frontier_time <= ownership_start + 1e-9
                        or candidate_frontier_time < verification_end - 1e-9
                    ):
                        continue
                    safe_frontier_time = min(
                        candidate_frontier_time,
                        candidate_end,
                    )
                    prefix_end = min(
                        len(frontier_tokens),
                        frontier_prompt_length + frontier.token_index + 1,
                    )
                    selected_reference_tokens = frontier_tokens[:prefix_end]
                    selected_reference_origin = frontier_origin
                    selected_reference_end = safe_frontier_time
                    selected_tokens = overlap_runtime.canonicalize_tokens(
                        model._tokenizer,
                        selected_reference_tokens,
                        selected_reference_origin,
                        ownership_start,
                        safe_frontier_time,
                    )
                    selected_name = "safe_frontier"
                    selection_reason = "safe_frontier_prefix_retained"
                    safe_frontier_source = frontier_diagnostic.name
                    force_reflow = (
                        safe_frontier_time < audio_duration - 1e-9
                    )
                    ended = True
                    accepted_chunk_quality = None
                    break

            fresh_tokens: list[int] | None = None
            fresh_ended = False
            fresh_quality: ChunkQualityMetrics | None = None
            # The last-resort candidate: primary, both A/B candidates and the
            # safe frontier all failed. It is the only generation on this path
            # with no prompt, so it costs a full chunk on the larger model.
            # Disabling it makes that chunk a discard instead.
            if selected_name is None and fresh_reanchor:
                fresh_vocab = tuple(model._tokenizer._vocab)
                fresh_request = RecoveryCandidateSpec(
                    candidate_name="fresh_reanchor",
                    request=GenerationRequest(
                        condition=condition,
                        prompt_ids=(),
                        forbidden_token_ids=(
                            tuple(
                                int(token)
                                for token in forbidden_tokens.tolist()
                            )
                            if forbidden_tokens is not None
                            else ()
                        ),
                        max_gen_len=max_gen_len,
                        use_sampling=False,
                        temperature=temperature,
                        cfg_coef=cfg_coef,
                        eos_id=eos_id,
                        trace_collector=trace_collector,
                        trace_context=(
                            *trace_context_prefix,
                            "fresh_reanchor",
                            chunk_index,
                        ),
                        # Fresh has no later candidate and never publishes a
                        # resident handle. A live critical finding completes
                        # the row as terminally rejected; final validation
                        # still runs.
                        guard_config=fresh_guard,
                        guard_vocab=fresh_vocab,
                        temporal_grammar=TemporalGrammarConfig.from_vocab(
                            fresh_vocab, (), ()
                        ),
                    ),
                    expected_frame_rate=model._tokenizer.frame_rate,
                    target_model=fresh_reanchor_model,
                    model_role="recovery",
                )
                fresh_result = yield fresh_request
                if not isinstance(fresh_result, RecoveryCandidateResult):
                    raise TypeError(
                        "fresh re-anchor requires a compatible recovery result"
                    )
                if fresh_result.resident_handle is not None:
                    yield from _discard_resident_generation(
                        fresh_result.resident_handle
                    )
                    fresh_forced_reason = "unexpected_fresh_reanchor_checkpoint"
                else:
                    fresh_forced_reason = _critical_guard_reason(
                        fresh_result.guard_findings
                    )
                fresh_tokens = list(fresh_result.tokens)
                fresh_ended = fresh_result.emitted_eos
                fresh_diagnostic = evaluate_candidate(
                    "fresh_reanchor",
                    fresh_tokens,
                    0,
                    fresh_ended,
                    model._tokenizer._vocab,
                    verification_match=match_note_events([], []),
                    forced_invalid_reason=fresh_forced_reason,
                    model_name=fresh_result.model_name,
                    sampling=False,
                        condition_seek_time=ownership_start,
                        guard_findings=fresh_result.guard_findings,
                    )
                if fresh_diagnostic.structural_valid:
                    fresh_quality_end = min(
                        next_seek or audio_duration,
                        ownership_start + _SEGMENT_DURATION,
                        audio_duration,
                    )
                    (
                        fresh_quality,
                        _,
                        fresh_findings,
                        fresh_assessment,
                    ) = _evaluate_chunk_quality(
                        model._tokenizer,
                        fresh_tokens,
                        ownership_start,
                        ownership_start,
                        fresh_quality_end,
                        len(fresh_tokens),
                        recovery_quality_guard,
                        chunk_quality_history.reference(),
                        fresh_result.guard_summary,
                    )
                    fresh_diagnostic = evaluate_candidate(
                        "fresh_reanchor",
                        fresh_tokens,
                        0,
                        fresh_ended,
                        model._tokenizer._vocab,
                        verification_match=match_note_events([], []),
                        forced_invalid_reason=_critical_guard_reason(
                            fresh_findings
                        ),
                        model_name=fresh_result.model_name,
                        sampling=False,
                        condition_seek_time=ownership_start,
                        chunk_quality=fresh_quality,
                        chunk_guard_reasons=tuple(
                            finding.reason for finding in fresh_findings
                        ),
                        guard_findings=(
                            *fresh_result.guard_findings,
                            *fresh_findings,
                        ),
                        chunk_quality_reference_samples=len(
                            chunk_quality_history.reference().samples
                        ),
                        chunk_quality_assessment=fresh_assessment,
                    )
                candidate_diagnostics.append(fresh_diagnostic)
                if (
                    fresh_diagnostic.structural_valid
                    and fresh_diagnostic.reference_eligible
                ):
                    selected_name = "fresh_reanchor"
                    selection_reason = "fresh_reanchor_reference_eligible"

            if selected_name is None:
                selected_name = "discarded"
                selection_reason = (
                    "fresh_reanchor_rejected_chunk_discarded"
                    if fresh_reanchor
                    else "fresh_reanchor_disabled_chunk_discarded"
                )
                selected_tokens = model._tokenizer.tie_section_token_ids([])
                selected_origin = ownership_start
                ended = True
                accepted_chunk_quality = None
            elif selected_name == "primary":
                selected_tokens = chunk_tokens
                selected_origin = seek
                accepted_chunk_quality = primary_chunk_quality
            elif selected_name == "fresh_reanchor":
                assert fresh_tokens is not None
                selected_tokens = fresh_tokens
                selected_origin = ownership_start
                ended = fresh_ended
                accepted_chunk_quality = fresh_quality
            elif selected_name == "safe_frontier":
                assert selected_reference_tokens is not None
                assert selected_reference_origin is not None
                assert selected_reference_end is not None
                accepted_chunk_quality = None
            else:
                selected_state = candidate_states[selected_name]
                selected_tokens = selected_state.tokens
                selected_origin = selected_state.origin
                ended = selected_state.ended
                accepted_chunk_quality = (
                    selected_state.diagnostic.chunk_quality
                    if selected_state.chunk_quality_safe
                    else None
                )

            if selected_name == "safe_frontier":
                assert selected_reference_end is not None
                ownership_end = min(
                    next_seek or audio_duration,
                    selected_reference_end,
                )
            else:
                selected_reference_tokens = selected_tokens
                selected_reference_origin = selected_origin
                selected_reference_end = min(
                    selected_origin + _SEGMENT_DURATION,
                    wav.shape[-1] / _SAMPLE_RATE,
                )
                ownership_end = min(
                    next_seek or audio_duration,
                    selected_reference_end,
                )
                selected_tokens = overlap_runtime.canonicalize_tokens(
                    model._tokenizer,
                    selected_tokens,
                    selected_origin,
                    ownership_start,
                    ownership_end,
                )
            selected_diagnostic = next(
                (
                    candidate
                    for candidate in candidate_diagnostics
                    if candidate.name
                    == (
                        safe_frontier_source
                        if selected_name == "safe_frontier"
                        else selected_name
                    )
                ),
                None,
            )
            overlap_probe = None
            if overlap_probe_enabled:
                primary_checkpoint, _ = _candidate_checkpoint_prefix(
                    chunk_tokens,
                    model._tokenizer._vocab,
                    verification_end_shift,
                )
                probe_candidates = [
                    build_overlap_probe_stream(
                        "primary",
                        chunk_tokens,
                        model._tokenizer._vocab,
                        origin=seek,
                        frame_rate=model._tokenizer.frame_rate,
                        window_start=ownership_start,
                        window_end=verification_end,
                        prompt_tokens=len(prompt_ids),
                        checkpoint_tokens=len(primary_checkpoint),
                    )
                ]
                probe_candidates.extend(
                    build_overlap_probe_stream(
                        name,
                        state.tokens,
                        model._tokenizer._vocab,
                        origin=state.origin,
                        frame_rate=model._tokenizer.frame_rate,
                        window_start=ownership_start,
                        window_end=verification_end,
                        prompt_tokens=state.prompt_length,
                        checkpoint_tokens=len(state.evaluation_tokens),
                    )
                    for name, state in candidate_states.items()
                )
                if fresh_tokens is not None:
                    probe_candidates.append(
                        build_overlap_probe_stream(
                            "fresh_reanchor",
                            fresh_tokens,
                            model._tokenizer._vocab,
                            origin=ownership_start,
                            frame_rate=model._tokenizer.frame_rate,
                            window_start=ownership_start,
                            window_end=verification_end,
                        )
                    )
                overlap_probe = OverlapProvenanceProbe(
                    window_start=ownership_start,
                    verification_start=verification_start,
                    verification_end=verification_end,
                    reference=build_overlap_probe_stream(
                        "reference",
                        previous_tokens,
                        model._tokenizer._vocab,
                        origin=previous_seek,
                        frame_rate=model._tokenizer.frame_rate,
                        window_start=ownership_start,
                        window_end=verification_end,
                    ),
                    candidates=tuple(probe_candidates),
                )
            recovery_event = RecoveryDiagnosticsEvent(
                chunk_index=chunk_index,
                seek_time=ownership_start,
                trigger_reason=trigger_reason,
                trigger_token_index=len(chunk_tokens) - len(prompt_ids),
                base_seed=recovery_seed,
                derived_seed=seed,
                attempt=0,
                execution="sequential",
                trigger_match=primary_match,
                candidates=tuple(candidate_diagnostics),
                selected_candidate=selected_name,
                selection_reason=selection_reason,
                selected_model_name=(
                    None
                    if selected_diagnostic is None
                    else selected_diagnostic.model_name
                ),
                    replay_start_time=(
                        None
                        if selected_name in {"fresh_reanchor", "discarded"}
                        else (
                            selected_reference_origin
                            if selected_name == "safe_frontier"
                            else selected_origin
                        )
                    ),
                verification_start_time=(
                    None
                    if selected_name in {"fresh_reanchor", "discarded"}
                    else verification_start
                ),
                verification_end_time=(
                    None
                    if selected_name in {"fresh_reanchor", "discarded"}
                    else verification_end
                ),
                output_start_time=ownership_start,
                output_end_time=ownership_end,
                selected_reference_eligible=(
                    selected_diagnostic is not None
                    and selected_diagnostic.reference_eligible
                ),
                safe_frontier_source=safe_frontier_source,
                safe_frontier_time=safe_frontier_time,
                overlap_probe=overlap_probe,
            )
        elif trigger_reason is not None:
            assert primary_diagnostic is not None
            primary_generation.close()
            if primary_diagnostic.structural_valid:
                accepted_chunk_quality = primary_chunk_quality
                selected_name = "primary"
                selection_reason = "primary_legal_recovery_disabled"
            else:
                selected_tokens = model._tokenizer.tie_section_token_ids([])
                selected_reference_tokens = selected_tokens
                selected_reference_origin = ownership_start
                selected_reference_end = min(
                    ownership_start + _SEGMENT_DURATION,
                    audio_duration,
                )
                ended = True
                accepted_chunk_quality = None
                selected_name = "discarded"
                selection_reason = "recovery_disabled_hard_invalid_discarded"
            recovery_event = RecoveryDiagnosticsEvent(
                chunk_index=chunk_index,
                seek_time=ownership_start,
                trigger_reason=trigger_reason,
                trigger_token_index=len(chunk_tokens) - len(prompt_ids),
                base_seed=recovery_seed,
                derived_seed=derive_recovery_seed(recovery_seed, chunk_index, 0),
                attempt=0,
                execution="disabled",
                trigger_match=primary_match,
                candidates=(primary_diagnostic,),
                selected_candidate=selected_name,
                selection_reason=selection_reason,
                selected_model_name=primary_diagnostic.model_name,
                replay_start_time=(
                    primary_diagnostic.condition_seek_time
                    if selected_name == "primary"
                    else None
                ),
                verification_start_time=verification_start,
                verification_end_time=verification_end,
                output_start_time=ownership_start,
                output_end_time=min(next_seek or audio_duration, audio_duration),
                selected_reference_eligible=(
                    selected_name == "primary"
                    and primary_diagnostic.reference_eligible
                ),
            )

        if accepted_chunk_quality is not None:
            chunk_quality_history.accept(accepted_chunk_quality)

        # Preserve condition-relative tokens for the next forcing prompt, but
        # emit only the portion owned by this chunk on the canonical timeline.
        if selected_reference_tokens is None:
            selected_reference_tokens = selected_tokens
            selected_reference_origin = condition_origin
            selected_reference_end = min(
                condition_origin + _SEGMENT_DURATION,
                audio_duration,
            )
            selected_tokens = overlap_runtime.canonicalize_tokens(
                model._tokenizer,
                selected_tokens,
                condition_origin,
                ownership_start,
                min(next_seek or audio_duration, selected_reference_end),
            )

        # A shifted winner cannot own beyond its condition. Replace every
        # remaining fixed-grid entry with a new schedule rooted at coverage_end.
        # The first condition reaches one overlap backward for forcing; later
        # entries advance by the normal stride and return to condition==owner.
        if (
            recovery_enabled
            and selected_reference_origin is not None
            and (
                force_reflow
                or (
                    selected_reference_origin != condition_origin
                    and next_seek is not None
                )
            )
            and selected_reference_end is not None
            and selected_reference_end < (next_seek or audio_duration) + (
                overlap_window.overlap_seconds if overlap_window is not None else 0.0
            )
        ):
            overlap_seconds = (
                overlap_window.overlap_seconds if overlap_window is not None else 0.0
            )
            first_owner = selected_reference_end
            first_origin = max(0.0, first_owner - overlap_seconds)
            replacement: list[tuple[ConditioningAttributes | None, float, float]] = [
                (None, first_origin, first_owner)
            ]
            origin = first_origin + stride
            while origin < audio_duration:
                replacement.append((None, origin, origin))
                origin += stride
            schedule[chunk_index + 1 :] = replacement
            next_seek = first_owner
            boundary = ChunkBoundary(ownership_start, next_seek)
            (stderr_logger or logging.getLogger("muscriptor.stderr")).info(
                "[muscriptor] recovery reflow: next condition %.2f-%.2fs; "
                "ownership starts %.2fs",
                first_origin,
                min(first_origin + _SEGMENT_DURATION, audio_duration),
                first_owner,
            )

        yield boundary
        for token in selected_tokens:
            tracker.feed(token)
            yield token

        if not ended:
            message = (
                f"chunk {chunk_index} (seek={seek:.1f}s) did not emit EOS "
                f"within {max_gen_len} tokens"
            )
            if no_eos_is_ok:
                if stderr_logger is None:
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                else:
                    stderr_logger.warning(message)
            else:
                raise RuntimeError(
                    message + " (this is only raised under --strict-eos)"
                )

        if recovery_event is not None:
            yield recovery_event

        yield GenerationAnomalyEvent(
            chunk_index=chunk_index,
            seek_time=ownership_start,
            summary=primary_summary,
            chunk_quality=primary_chunk_quality,
            chunk_guard_action=primary_chunk_action,
            chunk_guard_reasons=tuple(
                finding.reason for finding in primary_chunk_findings
            ),
            chunk_quality_reference_samples=chunk_quality_reference_samples,
            chunk_quality_assessment=primary_chunk_assessment,
        )

        if chunk_index > 0 and resolved_window is not None and overlap_detection:
            yield OverlapDiagnosticsEvent(
                chunk_index=chunk_index,
                seek_time=ownership_start,
                condition_origin=condition_origin,
                overlap_seconds=resolved_window.overlap_seconds,
                replay_seconds=resolved_window.replay_seconds,
                verification_seconds=resolved_window.verification_seconds,
                match=overlap_runtime.match_verification(
                    model._tokenizer,
                    previous_tokens,
                    previous_seek,
                    selected_reference_tokens or selected_tokens,
                    (
                        selected_reference_origin
                        if selected_reference_origin is not None
                        else seek
                    ),
                    verification_start,
                    verification_end,
                ),
            )

        reference_tokens = selected_reference_tokens or selected_tokens
        reference_seek = (
            selected_reference_origin
            if selected_reference_origin is not None
            else seek
        )
        previous_tokens = reference_tokens
        previous_seek = reference_seek
        ownership_end = next_seek or audio_duration
        progress_completed = max(
            progress_completed,
            min(total, math.ceil(ownership_end / stride)),
        )
        yield ProgressEvent(completed=progress_completed, total=total)
        chunk_index += 1
