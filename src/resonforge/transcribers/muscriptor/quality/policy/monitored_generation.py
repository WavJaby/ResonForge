"""Runtime adapter between token generation and anomaly monitoring."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch
from muscriptor.tokenizer.notes import Event

from resonforge.transcribers.muscriptor.quality.contract.guard_protocol import (
    MonitorSummary,
)
from resonforge.transcribers.muscriptor.quality.policy.generation_anomaly import (
    DEFAULT_ANOMALY_CONFIG,
    AnomalyMonitorConfig,
)
from resonforge.transcribers.muscriptor.quality.policy.generation_guard import (
    GenerationGuard,
    GenerationGuardConfig,
    GuardDecision,
    TokenBatchObserved,
)


@dataclass(frozen=True)
class MonitoredToken:
    token: int
    event: Event
    decision: GuardDecision


class MonitoredGeneration(Iterator[MonitoredToken]):
    """Consume model steps while excluding forced prompt tokens from metrics."""

    def __init__(
        self,
        steps: Iterable[torch.Tensor],
        *,
        eos_id: int,
        vocab: Sequence[Event],
        anomaly_detection: bool,
        config: AnomalyMonitorConfig = DEFAULT_ANOMALY_CONFIG,
        prompt_tokens: int = 0,
        allow_periodic_recovery: bool = True,
        summary: MonitorSummary | None = None,
    ) -> None:
        self._step_sources = deque((iter(steps),))
        self._eos_id = eos_id
        self._vocab = vocab
        self._prompt_tokens = prompt_tokens
        self._seen_tokens = 0
        self._guard = GenerationGuard(
            GenerationGuardConfig(
                mode="diagnostics" if anomaly_detection else "off",
                role="primary",
                anomaly=config,
                allow_periodic_recovery=allow_periodic_recovery,
            )
        )
        self._summary = summary
        self.ended = False

    def __iter__(self) -> MonitoredGeneration:
        return self

    def __next__(self) -> MonitoredToken:
        while True:
            while self._step_sources:
                try:
                    step = next(self._step_sources[0])
                    break
                except StopIteration:
                    self._step_sources.popleft()
            else:
                raise StopIteration
            token = int(step[0])
            if token == self._eos_id:
                self.ended = True
                raise StopIteration
            self._seen_tokens += 1
            event = self._vocab[token]
            decision = (
                self._guard.dispatch(TokenBatchObserved((event,)))
                if self._seen_tokens > self._prompt_tokens
                and self._summary is None
                else GuardDecision("continue")
            )
            return MonitoredToken(token, event, decision)

    def extend(self, steps: Iterable[torch.Tensor]) -> None:
        """Append continuation steps while preserving anomaly-monitor state."""
        if self.ended:
            raise ValueError("cannot extend generation after EOS")
        self._step_sources.append(iter(steps))

    def finalize(self) -> MonitorSummary:
        summary = self._summary or self._guard.summary(self.ended)
        assert summary is not None
        return summary

    def set_summary(self, summary: MonitorSummary | None) -> None:
        if summary is not None:
            self._summary = summary

    def close(self) -> None:
        while self._step_sources:
            close = getattr(self._step_sources.popleft(), "close", None)
            if close is not None:
                close()
