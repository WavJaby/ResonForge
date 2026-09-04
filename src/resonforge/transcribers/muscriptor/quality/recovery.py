"""Verification-driven recovery candidate validation and selection."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Literal

import torch
from muscriptor.tokenizer.notes import Event

from resonforge.transcribers.muscriptor.quality.chunk_quality import (
    AdaptiveChunkQualityAssessment,
    ChunkQualityMetrics,
)
from resonforge.transcribers.muscriptor.quality.generation_guard import GuardFinding
from resonforge.transcribers.muscriptor.quality.generation_position import (
    GenerationPosition,
)
from resonforge.transcribers.muscriptor.quality.overlap import OverlapMatch

RecoveryCandidateName = Literal[
    "shifted_replay",
    "new_seed",
    "secondary_model",
    "primary",
    "fresh_reanchor",
]
RecoverySelectionName = RecoveryCandidateName | Literal[
    "discarded",
    "safe_frontier",
]

PRIMARY_REPLAY_MIN_VERIFY_SCORE = 0.65


@dataclass(frozen=True)
class RecoveryCandidateDiagnostics:
    name: RecoveryCandidateName
    prompt_tokens: int
    generated_tokens: int
    emitted_eos: bool
    structural_valid: bool
    invalid_reason: str | None
    verification_match: OverlapMatch
    model_name: str | None = None
    sampling: bool | None = None
    condition_seek_time: float | None = None
    chunk_quality: ChunkQualityMetrics | None = None
    chunk_guard_reasons: tuple[str, ...] = ()
    guard_findings: tuple[GuardFinding, ...] = ()
    chunk_quality_reference_samples: int = 0
    chunk_quality_assessment: AdaptiveChunkQualityAssessment | None = None
    reference_eligible: bool = True
    reference_ineligible_reason: str | None = None


@dataclass(frozen=True)
class OverlapProbeToken:
    """One bounded semantic token with generation-phase provenance."""

    token_index: int
    event_type: str
    value: int
    absolute_time: float | None
    phase: Literal["prompt", "checkpoint", "continuation", "reference"]
    time_regression: bool = False


@dataclass(frozen=True)
class OverlapProbeStream:
    """Bounded token evidence for one side of an overlap decision."""

    name: str
    origin: float
    prompt_tokens: int
    checkpoint_tokens: int | None
    total_tokens: int
    captured_tokens: tuple[OverlapProbeToken, ...]
    omitted_tokens: int
    time_regressions: int


@dataclass(frozen=True)
class OverlapProvenanceProbe:
    """Reference/candidate token provenance around one overlap boundary."""

    window_start: float
    verification_start: float
    verification_end: float
    reference: OverlapProbeStream
    candidates: tuple[OverlapProbeStream, ...]


@dataclass(frozen=True)
class RecoveryDiagnosticsEvent:
    chunk_index: int
    seek_time: float
    trigger_reason: str
    trigger_token_index: int
    base_seed: int
    derived_seed: int
    attempt: int
    execution: Literal["sequential"]
    trigger_match: OverlapMatch
    candidates: tuple[RecoveryCandidateDiagnostics, ...]
    selected_candidate: RecoverySelectionName
    selection_reason: str
    selected_model_name: str | None = None
    replay_start_time: float | None = None
    verification_start_time: float | None = None
    verification_end_time: float | None = None
    output_start_time: float | None = None
    output_end_time: float | None = None
    selected_reference_eligible: bool = False
    safe_frontier_source: RecoveryCandidateName | None = None
    safe_frontier_time: float | None = None
    overlap_probe: OverlapProvenanceProbe | None = None


def build_overlap_probe_stream(
    name: str,
    tokens: list[int],
    vocab: list[Event],
    *,
    origin: float,
    frame_rate: int,
    window_start: float,
    window_end: float,
    prompt_tokens: int = 0,
    checkpoint_tokens: int | None = None,
    maximum_tokens: int = 512,
) -> OverlapProbeStream:
    """Capture bounded semantic evidence without retaining model tensors."""
    tick = round(origin * frame_rate)
    maximum_shift = -1
    regressions = 0
    captured: list[OverlapProbeToken] = []
    relevant = 0
    for index, token in enumerate(tokens):
        if token < 0 or token >= len(vocab):
            continue
        event = vocab[token]
        regression = False
        if event.type == "shift" and event.value >= 0:
            regression = maximum_shift >= 0 and event.value < maximum_shift
            regressions += int(regression)
            maximum_shift = max(maximum_shift, event.value)
            tick = round(origin * frame_rate) + event.value
        absolute_time = (
            tick / frame_rate
            if event.type in {"shift", "pitch", "drum", "velocity", "program"}
            else None
        )
        if event.type not in {"tie", "shift", "program", "velocity", "pitch", "drum"}:
            continue
        if absolute_time is not None and not (
            window_start - 1e-9 <= absolute_time <= window_end + 1e-9
        ):
            continue
        relevant += 1
        if len(captured) >= maximum_tokens:
            continue
        phase: Literal["prompt", "checkpoint", "continuation", "reference"]
        if name == "reference":
            phase = "reference"
        elif index < prompt_tokens:
            phase = "prompt"
        elif checkpoint_tokens is not None and index >= checkpoint_tokens:
            phase = "continuation"
        else:
            phase = "checkpoint"
        captured.append(
            OverlapProbeToken(
                token_index=index,
                event_type=event.type,
                value=event.value,
                absolute_time=absolute_time,
                phase=phase,
                time_regression=regression,
            )
        )
    return OverlapProbeStream(
        name=name,
        origin=origin,
        prompt_tokens=prompt_tokens,
        checkpoint_tokens=checkpoint_tokens,
        total_tokens=len(tokens),
        captured_tokens=tuple(captured),
        omitted_tokens=max(0, relevant - len(captured)),
        time_regressions=regressions,
    )


def derive_recovery_seed(base_seed: int, chunk_index: int, attempt: int) -> int:
    """Derive a stable signed-63-bit seed for one recovery attempt."""
    if min(base_seed, chunk_index, attempt) < 0:
        raise ValueError("recovery seed inputs must be non-negative")
    digest = hashlib.blake2b(digest_size=8)
    digest.update(b"muscriptor-recovery-v1")
    digest.update(struct.pack(">QQQ", base_seed, chunk_index, attempt))
    return int.from_bytes(digest.digest(), "big") & ((1 << 63) - 1)


def make_recovery_generator(device: torch.device, seed: int) -> torch.Generator:
    """Create a private generator matching the sampling device where supported."""
    generator_device: torch.device | str = "cpu" if device.type == "mps" else device
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def evaluate_candidate(
    name: RecoveryCandidateName,
    tokens: list[int],
    prompt_tokens: int,
    emitted_eos: bool,
    vocab: list[Event],
    *,
    verification_match: OverlapMatch,
    allow_missing_eos: bool = False,
    forced_invalid_reason: str | None = None,
    model_name: str | None = None,
    sampling: bool | None = None,
    condition_seek_time: float | None = None,
    chunk_quality: ChunkQualityMetrics | None = None,
    chunk_guard_reasons: tuple[str, ...] = (),
    guard_findings: tuple[GuardFinding, ...] = (),
    chunk_quality_reference_samples: int = 0,
    chunk_quality_assessment: AdaptiveChunkQualityAssessment | None = None,
) -> RecoveryCandidateDiagnostics:
    """Validate EOS/grammar and attach the candidate's verification evidence."""
    reason = _structural_invalid_reason(tokens, vocab)
    if forced_invalid_reason is not None:
        reason = forced_invalid_reason
    elif not emitted_eos and not allow_missing_eos:
        reason = "missing_eos"
    reference_ineligible_reason = _reference_ineligible_reason(
        chunk_guard_reasons,
    )
    return RecoveryCandidateDiagnostics(
        name=name,
        prompt_tokens=prompt_tokens,
        generated_tokens=len(tokens[prompt_tokens:]),
        emitted_eos=emitted_eos,
        structural_valid=reason is None,
        invalid_reason=reason,
        verification_match=verification_match,
        model_name=model_name,
        sampling=sampling,
        condition_seek_time=condition_seek_time,
        chunk_quality=chunk_quality,
        chunk_guard_reasons=chunk_guard_reasons,
        guard_findings=guard_findings,
        chunk_quality_reference_samples=chunk_quality_reference_samples,
        chunk_quality_assessment=chunk_quality_assessment,
        reference_eligible=(reason is None and reference_ineligible_reason is None),
        reference_ineligible_reason=reference_ineligible_reason,
    )


def _reference_ineligible_reason(
    guard_reasons: tuple[str, ...],
) -> str | None:
    """Reject intrinsic collapse as continuity state, not output grammar."""
    collapse_reasons = {
        "musical_time_stall",
        "repetition_and_time_stall",
        "periodic_mode_collapse",
        "rolling_musical_time_stall",
        "rolling_repetition_and_time_stall",
        "rolling_periodic_mode_collapse",
        "adaptive_synchronized_restart_burst",
    }
    reason = next((item for item in guard_reasons if item in collapse_reasons), None)
    if reason is not None:
        return reason
    return None


def recovery_candidate_order(
    primary: RecoveryCandidateDiagnostics,
    primary_replay: RecoveryCandidateDiagnostics,
    secondary: RecoveryCandidateDiagnostics,
) -> tuple[RecoveryCandidateDiagnostics, ...]:
    """Order trusted legal candidates under asymmetric recovery policy.

    Independent secondary evidence remains first even when it disagrees with a
    possibly corrupt primary reference. Shifted replay must clear its absolute
    score floor. Primary remains a full candidate rather than an empty-set
    fallback. Intrinsic collapse removes continuity trust without changing hard
    output legality.
    """

    def has_score(candidate: RecoveryCandidateDiagnostics) -> bool:
        return (
            candidate.structural_valid
            and candidate.reference_eligible
            and candidate.verification_match.score is not None
        )

    a_eligible = has_score(primary_replay) and (
        primary_replay.verification_match.score
        > PRIMARY_REPLAY_MIN_VERIFY_SCORE
    )
    secondary_eligible = secondary.structural_valid and secondary.reference_eligible
    ordered: list[RecoveryCandidateDiagnostics] = []
    if secondary.name == "secondary_model" and secondary_eligible:
        ordered.append(secondary)
    if a_eligible:
        ordered.append(primary_replay)
    if secondary.name != "secondary_model" and secondary_eligible:
        ordered.append(secondary)
    if primary.structural_valid and primary.reference_eligible:
        ordered.append(primary)
    return tuple(ordered)


def select_checkpoint_candidate(
    primary: RecoveryCandidateDiagnostics,
    recoveries: tuple[RecoveryCandidateDiagnostics, ...],
) -> RecoveryCandidateDiagnostics | None:
    """Select one checkpoint row only when recovery evidence beats primary."""
    primary_eligible = primary.structural_valid and primary.reference_eligible
    primary_score = (
        primary.verification_match.score if primary_eligible else None
    )
    ranked = sorted(
        (
            candidate
            for candidate in recoveries
            if candidate.structural_valid
            and candidate.reference_eligible
            and candidate.verification_match.score is not None
            and (
                primary_score is None
                or candidate.verification_match.score > primary_score
            )
        ),
        key=lambda candidate: candidate.verification_match.score,
        reverse=True,
    )
    if ranked:
        return ranked[0]
    if primary_eligible:
        return primary
    return None


def candidate_safe_frontier(
    candidate: RecoveryCandidateDiagnostics,
) -> GenerationPosition | None:
    """Return the earliest frontier covering every terminal failure finding."""
    terminal_reasons = {
        reason
        for reason in (
            candidate.invalid_reason,
            candidate.reference_ineligible_reason,
        )
        if reason is not None
    }
    relevant = tuple(
        finding
        for finding in candidate.guard_findings
        if finding.status == "critical" or finding.reason in terminal_reasons
    )
    if not relevant or any(
        finding.last_safe_frontier is None for finding in relevant
    ):
        return None
    return min(
        (
            finding.last_safe_frontier
            for finding in relevant
            if finding.last_safe_frontier is not None
        ),
        key=lambda position: position.token_index,
    )


def _structural_invalid_reason(tokens: list[int], vocab: list[Event]) -> str | None:
    in_prologue = True
    program_seen = False
    for token in tokens:
        if token < 0 or token >= len(vocab):
            return "invalid_token"
        event = vocab[token]
        if not in_prologue:
            continue
        if event.type == "tie":
            in_prologue = False
        elif event.type == "shift":
            return "shift_before_tie"
        elif event.type == "program":
            program_seen = True
        elif event.type == "pitch" and not program_seen:
            return "pitch_without_program_in_tie"
    return "missing_tie" if in_prologue else None
