"""Rolling issue-1 generation anomaly diagnostics and policy."""

from __future__ import annotations

import gzip
from collections import Counter, deque
from dataclasses import dataclass
from typing import Literal

from muscriptor.tokenizer.notes import Event

from resonforge.transcribers.muscriptor.quality.chunk_quality import (
    AdaptiveChunkQualityAssessment,
    ChunkQualityMetrics,
)
from resonforge.transcribers.muscriptor.quality.guard_protocol import (
    AnomalyMetrics,
    MonitorSummary,
)

AnomalyMode = Literal["off", "diagnostics", "recovery"]


@dataclass(frozen=True)
class AnomalyMonitorConfig:
    warmup_tokens: int = 128
    window_tokens: int = 128
    poll_tokens: int = 32
    trigger_score: float = 4.0
    consecutive_trigger_windows: int = 2
    periodic_8gram_rate: float = 0.85
    periodic_16gram_rate: float = 0.60
    periodic_unique_token_rate: float = 0.10
    hard_stall_tokens: int = 80

    def __post_init__(self) -> None:
        if min(
            self.warmup_tokens,
            self.window_tokens,
            self.poll_tokens,
            self.hard_stall_tokens,
        ) <= 0:
            raise ValueError("anomaly token counts must be positive")
        if self.consecutive_trigger_windows <= 0:
            raise ValueError("consecutive_trigger_windows must be positive")
        for value in (
            self.periodic_8gram_rate,
            self.periodic_16gram_rate,
            self.periodic_unique_token_rate,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("periodic anomaly thresholds must be in [0, 1]")


DEFAULT_ANOMALY_CONFIG = AnomalyMonitorConfig()


@dataclass(frozen=True)
class AnomalyObservation:
    metrics: AnomalyMetrics | None = None
    trigger_reason: str | None = None


@dataclass(frozen=True)
class GenerationAnomalyEvent:
    """Per-chunk diagnostics; it never mutates decoded note state."""

    chunk_index: int
    seek_time: float
    summary: MonitorSummary
    chunk_quality: ChunkQualityMetrics | None = None
    chunk_guard_action: str = "continue"
    chunk_guard_reasons: tuple[str, ...] = ()
    chunk_quality_reference_samples: int = 0
    chunk_quality_assessment: AdaptiveChunkQualityAssessment | None = None


class RollingAnomalyMonitor:
    """Compute rolling anomaly features without owning runtime policy."""

    def __init__(
        self,
        config: AnomalyMonitorConfig = DEFAULT_ANOMALY_CONFIG,
        *,
        allow_periodic_recovery: bool = True,
        periodic_trigger_windows: int = 1,
        initial_periodic_windows: int = 0,
    ) -> None:
        self.config = config
        self.allow_periodic_recovery = allow_periodic_recovery
        if periodic_trigger_windows <= 0:
            raise ValueError("periodic_trigger_windows must be positive")
        self.periodic_trigger_windows = periodic_trigger_windows
        self._symbols: deque[str] = deque(maxlen=config.window_tokens)
        self._generated_tokens = 0
        self._tokens_since_shift = 0
        self._consecutive_high_scores = 0
        self._trigger_reason: str | None = None
        self._last_metrics: AnomalyMetrics | None = None
        self._poll_metrics: list[AnomalyMetrics] = []
        self._hard_triggered = False
        self._periodic_window_streak = initial_periodic_windows
        self._polled_here = False

    @property
    def generated_tokens(self) -> int:
        return self._generated_tokens

    def observe(self, event: Event) -> AnomalyObservation:
        self._generated_tokens += 1
        self._symbols.append(semantic_symbol(event))
        if event.type == "shift" and event.value > 0:
            self._tokens_since_shift = 0
        else:
            self._tokens_since_shift += 1

        hard_reason = None
        if (
            self._tokens_since_shift >= self.config.hard_stall_tokens
            and not self._hard_triggered
        ):
            hard_reason = "musical_time_stall"
        if hard_reason is not None:
            self._hard_triggered = True
            metrics = self._compute_metrics()
            self._last_metrics = metrics
            self._poll_metrics.append(metrics)
            if (
                metrics.repeated_4gram_rate >= 0.55
                or metrics.repeated_8gram_rate >= 0.35
                or metrics.repeated_16gram_rate >= 0.20
            ):
                hard_reason = "repetition_and_time_stall"
            return self._observation(metrics, hard_reason)

        if not self._should_poll():
            return AnomalyObservation()

        metrics = self._compute_metrics()
        self._polled_here = True
        self._last_metrics = metrics
        self._poll_metrics.append(metrics)
        if self.allow_periodic_recovery:
            if periodic_mode_collapse(metrics, self.config):
                self._periodic_window_streak += 1
                if self._periodic_window_streak >= self.periodic_trigger_windows:
                    return self._observation(metrics, "periodic_mode_collapse")
            else:
                self._periodic_window_streak = 0
        if metrics.score >= self.config.trigger_score:
            self._consecutive_high_scores += 1
        else:
            self._consecutive_high_scores = 0

        reason = hard_reason
        if (
            reason is None
            and self._consecutive_high_scores >= self.config.consecutive_trigger_windows
        ):
            reason = "anomaly_score"

        return self._observation(metrics, reason)

    def _observation(
        self,
        metrics: AnomalyMetrics,
        reason: str | None,
    ) -> AnomalyObservation:
        if reason is not None and self._trigger_reason is None:
            self._trigger_reason = reason
        return AnomalyObservation(metrics, reason)

    def finalize(self, emitted_eos: bool) -> MonitorSummary:
        periodic_window_streak = (
            self._periodic_window_streak if self._polled_here else 0
        )
        return MonitorSummary(
            generated_tokens=self._generated_tokens,
            emitted_eos=emitted_eos,
            trigger_reason=self._trigger_reason,
            last_metrics=self._last_metrics,
            poll_metrics=tuple(self._poll_metrics),
            periodic_window_streak=periodic_window_streak,
        )

    def _should_poll(self) -> bool:
        if self._generated_tokens < self.config.warmup_tokens:
            return False
        return (
            self._generated_tokens - self.config.warmup_tokens
        ) % self.config.poll_tokens == 0

    def _compute_metrics(self) -> AnomalyMetrics:
        symbols = list(self._symbols)
        repeated = {size: repeated_ngram_rate(symbols, size) for size in (4, 8, 16)}
        unique_rate = len(set(symbols)) / len(symbols) if symbols else 1.0
        gzip_ratio = semantic_gzip_ratio(symbols)
        repetition_severity = max(
            severity(repeated[4], 0.35, 0.55, higher_is_worse=True),
            severity(repeated[8], 0.20, 0.35, higher_is_worse=True),
            severity(repeated[16], 0.10, 0.20, higher_is_worse=True),
        )
        stall_severity = severity(
            float(self._tokens_since_shift),
            max(1, round(self.config.hard_stall_tokens * 0.6)),
            self.config.hard_stall_tokens,
            higher_is_worse=True,
        )
        unique_severity = severity(unique_rate, 0.45, 0.30, higher_is_worse=False)
        score = 2.0 * repetition_severity + 1.5 * stall_severity + 1.0 * unique_severity
        return AnomalyMetrics(
            generated_tokens=self._generated_tokens,
            window_tokens=len(symbols),
            repeated_4gram_rate=repeated[4],
            repeated_8gram_rate=repeated[8],
            repeated_16gram_rate=repeated[16],
            unique_token_rate=unique_rate,
            gzip_ratio=gzip_ratio,
            tokens_since_shift=self._tokens_since_shift,
            score=score,
        )


def semantic_symbol(event: Event) -> str:
    """Stable semantic representation shared by rolling and candidate GZIP."""
    if event.type == "shift":
        return "s"
    prefixes = {"velocity": "v", "program": "r", "pitch": "p", "drum": "d"}
    prefix = prefixes.get(event.type, event.type)
    return f"{prefix}{event.value}"


def semantic_gzip_ratio(symbols: list[str]) -> float:
    raw = "|".join(symbols).encode()
    return len(raw) / len(gzip.compress(raw)) if raw else 0.0


def repeated_ngram_rate(symbols: list[str], size: int) -> float:
    if size <= 0:
        raise ValueError("ngram size must be positive")
    total = len(symbols) - size + 1
    if total <= 0:
        return 0.0
    counts = Counter(tuple(symbols[index : index + size]) for index in range(total))
    repeated_occurrences = sum(count for count in counts.values() if count > 1)
    return repeated_occurrences / total


def periodic_mode_collapse(
    metrics: AnomalyMetrics,
    config: AnomalyMonitorConfig = DEFAULT_ANOMALY_CONFIG,
) -> bool:
    """Detect an exact long token cycle while musical time still advances."""
    return (
        metrics.window_tokens >= config.window_tokens
        and metrics.repeated_8gram_rate >= config.periodic_8gram_rate
        and metrics.repeated_16gram_rate >= config.periodic_16gram_rate
        and metrics.unique_token_rate <= config.periodic_unique_token_rate
    )


def severity(
    value: float,
    suspicious: float,
    severe: float,
    *,
    higher_is_worse: bool,
) -> float:
    if higher_is_worse:
        if value >= severe:
            return 1.0
        if value >= suspicious:
            return 0.5
    else:
        if value <= severe:
            return 1.0
        if value <= suspicious:
            return 0.5
    return 0.0
