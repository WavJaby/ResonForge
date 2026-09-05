"""Running one `RecoveryCandidateSpec`, from the executor's side.

A spec arrives fully built -- that is its point. What still needs the model
that will actually run it lives here: the tokenizer compatibility gate, the
margin collector the decode loop feeds, and the adapter that turns a
`GenerationResult` back into a `RecoveryCandidateResult`.

Contract rather than policy because nothing here decides anything. The decode
loop drives `ShiftMarginCollector.observe` itself (`continuous_generation`),
and `transcription` calls prepare/complete to build scheduler work items --
neither may reach into the recovery policy, and neither needs to.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from muscriptor.tokenizer.notes import Event
from muscriptor.utils.sampling import chosen_token_margins

from resonforge.transcribers.muscriptor.quality.contract.generation_batch import (
    GenerationRequest,
    GenerationResult,
    RecoveryCandidateResult,
    RecoveryCandidateSpec,
)
from resonforge.transcribers.muscriptor.quality.contract.model_protocol import (
    ModelProtocol,
)


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
    margin_collector: ShiftMarginCollector | None


class ShiftMarginCollector:
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
            or tuple(tokenizer._vocab) != spec.expected_vocab
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
            ShiftMarginCollector(
                spec.expected_vocab,
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