"""Event-driven generation validation shared by decode and recovery policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from muscriptor.tokenizer.notes import Event

from resonforge.transcribers.muscriptor.quality.chunk_quality import (
    DEFAULT_ADAPTIVE_CHUNK_QUALITY_CONFIG,
    AdaptiveChunkQualityAssessment,
    AdaptiveChunkQualityConfig,
    ChunkQualityMetrics,
    ChunkQualityReference,
    RestartChainCollapseEvidence,
    assess_adaptive_chunk_quality,
)
from resonforge.transcribers.muscriptor.quality.generation_anomaly import (
    DEFAULT_ANOMALY_CONFIG,
    AnomalyMetrics,
    AnomalyMode,
    AnomalyMonitorConfig,
    MonitorSummary,
    RollingAnomalyMonitor,
)
from resonforge.transcribers.muscriptor.quality.generation_position import (
    GenerationPosition,
)

GuardRole = Literal["primary", "recovery", "fresh"]
GuardStatus = Literal["clear", "warning", "critical"]
GuardAction = Literal[
    "continue",
    "pause_for_verify",
    "interrupt_to_recovery",
    "reject_candidate",
    "reject_terminal",
    "accept_candidate",
    "request_recovery",
]


@dataclass(frozen=True)
class HardPitchEnvelopeConfig:
    """Broad physical sanity bounds for known pitched sources."""

    minimum_pitch: int = 12
    maximum_pitch: int = 119

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_pitch <= self.maximum_pitch <= 127:
            raise ValueError("pitch envelope must be within MIDI [0, 127]")


DEFAULT_HARD_PITCH_ENVELOPE = HardPitchEnvelopeConfig()


@dataclass(frozen=True)
class TokenBatchObserved:
    events: tuple[Event, ...]


@dataclass(frozen=True)
class VerifyReady:
    score: float | None
    informative: bool
    comparable_evidence: int
    structural_valid: bool = True


@dataclass(frozen=True)
class GenerationCompleted:
    emitted_eos: bool
    reached_max_length: bool


@dataclass(frozen=True)
class ChunkQualityReady:
    metrics: ChunkQualityMetrics
    reference: ChunkQualityReference = ChunkQualityReference()
    rolling_summary: MonitorSummary | None = None
    assessment: AdaptiveChunkQualityAssessment | None = None
    restart_chain_collapse: RestartChainCollapseEvidence | None = None


GuardEvent = (
    TokenBatchObserved | VerifyReady | GenerationCompleted | ChunkQualityReady
)


@dataclass(frozen=True)
class GuardFinding:
    detector: str
    status: GuardStatus
    reason: str
    metrics: dict[str, float | int | bool | str | None] = field(
        default_factory=dict
    )
    observed_at: GenerationPosition | None = None
    suspected_start: GenerationPosition | None = None
    last_safe_frontier: GenerationPosition | None = None


@dataclass(frozen=True)
class GuardDecision:
    action: GuardAction
    findings: tuple[GuardFinding, ...] = ()


@dataclass(frozen=True)
class GenerationGuardConfig:
    mode: AnomalyMode = "off"
    role: GuardRole = "primary"
    anomaly: AnomalyMonitorConfig = DEFAULT_ANOMALY_CONFIG
    allow_periodic_recovery: bool = True
    verify_threshold: float = 0.80
    min_verify_evidence: int = 4
    chunk_quality: AdaptiveChunkQualityConfig = (
        DEFAULT_ADAPTIVE_CHUNK_QUALITY_CONFIG
    )
    enforce_completed_quality: bool = False
    hard_pitch_envelope: HardPitchEnvelopeConfig | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.verify_threshold <= 1.0:
            raise ValueError("verify_threshold must be in [0, 1]")
        if self.min_verify_evidence < 1:
            raise ValueError("min_verify_evidence must be positive")


class GuardDetector(Protocol):
    detector_id: str

    def on_event(self, event: GuardEvent) -> tuple[GuardFinding, ...]: ...


class RollingAnomalyDetector:
    """Adapter for rolling stall/repetition/GZIP feature detection."""

    detector_id = "rolling_anomaly"

    def __init__(self, config: GenerationGuardConfig) -> None:
        self._mode = config.mode
        self._monitor = RollingAnomalyMonitor(
            config.anomaly,
            allow_periodic_recovery=config.allow_periodic_recovery,
        )
        self._shift_frontiers: list[GenerationPosition] = []

    def _position(self, token_index: int) -> GenerationPosition:
        shift_value = next(
            (
                frontier.shift_value
                for frontier in reversed(self._shift_frontiers)
                if frontier.token_index <= token_index
            ),
            None,
        )
        return GenerationPosition(token_index, shift_value)

    def _finding_positions(
        self,
        reason: str,
        metrics: AnomalyMetrics,
    ) -> tuple[GenerationPosition, GenerationPosition, GenerationPosition | None]:
        observed_index = metrics.generated_tokens - 1
        if reason in {"musical_time_stall", "repetition_and_time_stall"}:
            suspected_index = max(
                0,
                observed_index - metrics.tokens_since_shift + 1,
            )
        else:
            suspected_index = max(
                0,
                observed_index - metrics.window_tokens + 1,
            )
        frontier = next(
            (
                item
                for item in reversed(self._shift_frontiers)
                if item.token_index < suspected_index
            ),
            None,
        )
        return (
            self._position(observed_index),
            self._position(suspected_index),
            frontier,
        )

    def on_event(self, event: GuardEvent) -> tuple[GuardFinding, ...]:
        if not isinstance(event, TokenBatchObserved):
            return ()
        if self._mode == "off":
            return ()
        findings: list[GuardFinding] = []
        for token_event in event.events:
            observation = self._monitor.observe(token_event)
            token_index = self._monitor.generated_tokens - 1
            if token_event.type == "shift" and token_event.value > 0:
                self._shift_frontiers.append(
                    GenerationPosition(token_index, token_event.value)
                )
            if observation.trigger_reason is None:
                continue
            metrics = observation.metrics
            assert metrics is not None
            observed_at, suspected_start, last_safe_frontier = (
                self._finding_positions(observation.trigger_reason, metrics)
            )
            findings.append(
                GuardFinding(
                    detector=self.detector_id,
                    status=(
                        "critical"
                        if self._mode == "recovery"
                        and observation.trigger_reason != "anomaly_score"
                        else "warning"
                    ),
                    reason=observation.trigger_reason,
                    metrics=(
                        {
                            "generated_tokens": metrics.generated_tokens,
                            "tokens_since_shift": metrics.tokens_since_shift,
                            "repeated_8gram_rate": metrics.repeated_8gram_rate,
                            "repeated_16gram_rate": metrics.repeated_16gram_rate,
                            "unique_token_rate": metrics.unique_token_rate,
                            "gzip_ratio": metrics.gzip_ratio,
                            "score": metrics.score,
                        }
                    ),
                    observed_at=observed_at,
                    suspected_start=suspected_start,
                    last_safe_frontier=last_safe_frontier,
                )
            )
        return tuple(findings)

    def finalize(self, emitted_eos: bool) -> MonitorSummary:
        return self._monitor.finalize(emitted_eos)


class OverlapVerificationDetector:
    detector_id = "overlap_verification"

    def __init__(self, config: GenerationGuardConfig) -> None:
        self._config = config

    def on_event(self, event: GuardEvent) -> tuple[GuardFinding, ...]:
        if not isinstance(event, VerifyReady):
            return ()
        if not event.structural_valid:
            return (
                GuardFinding(
                    self.detector_id,
                    "critical",
                    "structural_invalid",
                    {"score": event.score, "informative": event.informative},
                ),
            )
        if (
            event.informative
            and event.score is not None
            and event.comparable_evidence >= self._config.min_verify_evidence
            and event.score < self._config.verify_threshold
        ):
            return (
                GuardFinding(
                    self.detector_id,
                    "warning",
                    "low_overlap_verification",
                    {
                        "score": event.score,
                        "informative": event.informative,
                        "comparable_evidence": event.comparable_evidence,
                    },
                ),
            )
        return ()


class CompletionDetector:
    detector_id = "completion"

    def on_event(self, event: GuardEvent) -> tuple[GuardFinding, ...]:
        if not isinstance(event, GenerationCompleted):
            return ()
        if event.reached_max_length and not event.emitted_eos:
            return (
                GuardFinding(
                    self.detector_id,
                    "warning",
                    "missing_eos_at_max_length",
                ),
            )
        return ()


class ChunkQualityDetector:
    """Compare a completed chunk with accepted history without fixed profiles."""

    detector_id = "chunk_quality"

    def __init__(self, config: AdaptiveChunkQualityConfig) -> None:
        self._config = config

    def on_event(self, event: GuardEvent) -> tuple[GuardFinding, ...]:
        if not isinstance(event, ChunkQualityReady):
            return ()
        metrics = event.metrics
        findings: list[GuardFinding] = []
        if event.rolling_summary is not None:
            rolling_reason = event.rolling_summary.trigger_reason
            if rolling_reason is not None:
                findings.append(
                    GuardFinding(
                        self.detector_id,
                        "warning",
                        f"rolling_{rolling_reason}",
                        {"generated_tokens": metrics.generated_tokens},
                    )
                )
        assessment = event.assessment or assess_adaptive_chunk_quality(
                metrics,
                event.reference,
                self._config,
            )
        for feature in assessment.outliers:
            score = assessment.scores[feature]
            findings.append(
                GuardFinding(
                    self.detector_id,
                    "warning",
                    f"adaptive_{feature}_outlier",
                    {
                        "value": score.value,
                        "history_median": score.history_median,
                        "history_scale": score.history_scale,
                        "robust_z": score.robust_z,
                        "history_samples": len(event.reference.samples),
                    },
                )
            )
        return tuple(findings)


class HardPitchEnvelopeDetector:
    """Reject physically extreme pitches for explicitly pitched sources."""

    detector_id = "hard_pitch_envelope"

    def __init__(self, config: HardPitchEnvelopeConfig | None) -> None:
        self._config = config

    def on_event(self, event: GuardEvent) -> tuple[GuardFinding, ...]:
        if not isinstance(event, ChunkQualityReady) or self._config is None:
            return ()
        metrics = event.metrics
        if metrics.pitch_min is None or metrics.pitch_max is None:
            return ()
        if (
            metrics.pitch_min >= self._config.minimum_pitch
            and metrics.pitch_max <= self._config.maximum_pitch
        ):
            return ()
        return (
            GuardFinding(
                self.detector_id,
                "critical",
                "pitch_outside_hard_envelope",
                {
                    "pitch_min": metrics.pitch_min,
                    "pitch_max": metrics.pitch_max,
                    "minimum_pitch": self._config.minimum_pitch,
                    "maximum_pitch": self._config.maximum_pitch,
                },
            ),
        )


class CompletedQualityCollapseDetector:
    """Reject a synchronized restart burst only with independent rate evidence."""

    detector_id = "completed_quality_collapse"

    def __init__(
        self,
        enabled: bool,
        config: AdaptiveChunkQualityConfig,
    ) -> None:
        self._enabled = enabled
        self._config = config

    def on_event(self, event: GuardEvent) -> tuple[GuardFinding, ...]:
        if not self._enabled or not isinstance(event, ChunkQualityReady):
            return ()
        assessment = event.assessment or assess_adaptive_chunk_quality(
            event.metrics,
            event.reference,
            self._config,
        )
        rate_outliers = all(
            (score := assessment.scores.get(feature)) is not None
            and score.robust_z >= self._config.robust_z_threshold
            for feature in ("event_rate", "generated_token_rate")
        )
        if not rate_outliers or event.metrics.max_synchronized_restarts < 4:
            return ()
        restart = event.restart_chain_collapse
        return (
            GuardFinding(
                self.detector_id,
                "warning",
                "adaptive_synchronized_restart_burst",
                {
                    "max_synchronized_restarts": (
                        event.metrics.max_synchronized_restarts
                    ),
                },
                observed_at=None if restart is None else restart.observed_at,
                suspected_start=(
                    None if restart is None else restart.suspected_start
                ),
                last_safe_frontier=(
                    None if restart is None else restart.last_safe_frontier
                ),
            ),
        )


class RestartChainCollapseDetector:
    """Report one coherent multi-pitch restart episode as continuity-invalid."""

    detector_id = "restart_chain_collapse"

    def on_event(self, event: GuardEvent) -> tuple[GuardFinding, ...]:
        if (
            not isinstance(event, ChunkQualityReady)
            or event.restart_chain_collapse is None
        ):
            return ()
        evidence = event.restart_chain_collapse
        return (
            GuardFinding(
                self.detector_id,
                "warning",
                "adjacent_same_pitch_restart_collapse",
                {
                    "adjacent_restarts": evidence.adjacent_restarts,
                    "maximum_chain": evidence.maximum_chain,
                    "synchronized_restarts": evidence.synchronized_restarts,
                },
                observed_at=evidence.observed_at,
                suspected_start=evidence.suspected_start,
                last_safe_frontier=evidence.last_safe_frontier,
            ),
        )


class GenerationGuard:
    """Fan out events to detectors, then map findings through role policy."""

    def __init__(
        self,
        config: GenerationGuardConfig,
        detectors: tuple[GuardDetector, ...] | None = None,
    ) -> None:
        self.config = config
        self.detectors = detectors or (
            RollingAnomalyDetector(config),
            OverlapVerificationDetector(config),
            CompletionDetector(),
            ChunkQualityDetector(config.chunk_quality),
            CompletedQualityCollapseDetector(
                config.enforce_completed_quality,
                config.chunk_quality,
            ),
            RestartChainCollapseDetector(),
            HardPitchEnvelopeDetector(config.hard_pitch_envelope),
        )
        self.findings: list[GuardFinding] = []

    def dispatch(self, event: GuardEvent) -> GuardDecision:
        fresh = tuple(
            finding
            for detector in self.detectors
            for finding in detector.on_event(event)
        )
        self.findings.extend(fresh)
        return GuardDecision(self._action(event, fresh), fresh)

    def anomaly_summary(self, emitted_eos: bool) -> MonitorSummary | None:
        detector = next(
            (
                detector
                for detector in self.detectors
                if isinstance(detector, RollingAnomalyDetector)
            ),
            None,
        )
        return None if detector is None else detector.finalize(emitted_eos)

    def _action(
        self,
        event: GuardEvent,
        findings: tuple[GuardFinding, ...],
    ) -> GuardAction:
        if isinstance(event, VerifyReady):
            return "pause_for_verify"
        if isinstance(event, ChunkQualityReady) and any(
            finding.status == "critical"
            or finding.reason.startswith("adaptive_")
            for finding in findings
        ):
            if self.config.mode != "recovery":
                return "continue"
            if self.config.role == "primary":
                return "request_recovery"
            if self.config.role == "fresh":
                return "reject_terminal"
            return (
                "reject_candidate"
                if any(finding.status == "critical" for finding in findings)
                else "accept_candidate"
            )
        if isinstance(event, GenerationCompleted) and not findings:
            return "accept_candidate"
        if self.config.mode != "recovery" or not any(
            finding.status == "critical" for finding in findings
        ):
            return "continue"
        return (
            "interrupt_to_recovery"
            if self.config.role == "primary"
            else (
                "reject_terminal"
                if self.config.role == "fresh"
                else "reject_candidate"
            )
        )
