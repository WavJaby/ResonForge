"""Token-level overlap prompt, evidence, and ownership helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from muscriptor.tokenizer.notes import DRUM_PROGRAM, NoteEvent

from resonforge.transcribers.muscriptor.quality.model_protocol import TokenizerProtocol
from resonforge.transcribers.muscriptor.quality.overlap import (
    OverlapMatch,
    OverlapWindow,
    match_note_events,
)

_PROMPT_TOKEN_BUCKETS = (16, 32, 64, 96)


@dataclass(frozen=True)
class PromptPlan:
    """A semantic prompt fitted to a stable generation tensor shape."""

    prompt_ids: tuple[int, ...]
    padded_tokens: int
    retained_event_count: int
    dropped_event_count: int
    open_note_count: int


def plan_prompt(
    tokenizer: TokenizerProtocol,
    open_keys: list[tuple[int, int]],
    replay: list[NoteEvent],
    seek: float,
    max_gen_len: int,
) -> PromptPlan:
    """Keep complete replay events within the model's hard token budget."""
    generation_limit = max(0, max_gen_len - 1)
    tie_only = tokenizer.tie_section_token_ids(open_keys)
    if len(tie_only) > generation_limit:
        return PromptPlan((), 0, 0, len(replay), len(open_keys))

    budget = generation_limit
    retained = list(replay)
    prompt_ids = tokenizer.overlap_prompt_token_ids(open_keys, retained, seek)
    while retained and len(prompt_ids) > budget:
        retained.pop(0)
        prompt_ids = tokenizer.overlap_prompt_token_ids(open_keys, retained, seek)
    if len(prompt_ids) > generation_limit:
        prompt_ids = tie_only
        retained = []

    actual_tokens = len(prompt_ids)
    padded_tokens = next(
        (
            bucket
            for bucket in _PROMPT_TOKEN_BUCKETS
            if actual_tokens <= bucket <= generation_limit
        ),
        actual_tokens,
    )
    return PromptPlan(
        prompt_ids=tuple(prompt_ids),
        padded_tokens=padded_tokens,
        retained_event_count=len(retained),
        dropped_event_count=len(replay) - len(retained),
        open_note_count=len(open_keys),
    )


def forcing_prompt_plan(
    tokenizer: TokenizerProtocol,
    open_keys: list[tuple[int, int]],
    previous_tokens: list[int],
    previous_seek: float,
    seek: float,
    overlap_window: OverlapWindow | None,
    max_gen_len: int,
    chunk_index: int,
    stderr_logger: logging.Logger | None = None,
) -> PromptPlan:
    if overlap_window is None and not open_keys:
        return PromptPlan((), 0, 0, 0, 0)
    replay = (
        []
        if overlap_window is None
        else note_events_in_range(
            tokenizer,
            previous_tokens,
            previous_seek,
            seek,
            seek + overlap_window.replay_seconds,
        )
    )
    plan = plan_prompt(tokenizer, open_keys, replay, seek, max_gen_len)
    logger = stderr_logger or logging.getLogger("muscriptor.stderr")
    if plan.dropped_event_count:
        logger.warning(
            "chunk %s (seek=%.1fs): dropped %s oldest replay events to fit "
            "the %s-token generation budget",
            chunk_index,
            seek,
            plan.dropped_event_count,
            plan.padded_tokens,
        )
        if plan.retained_event_count == 0:
            logger.warning(
                "chunk %s (seek=%.1fs): using tie-only forcing",
                chunk_index,
                seek,
            )
    if not plan.prompt_ids and open_keys:
        logger.warning(
            "chunk %s (seek=%.1fs): tie prompt exceeds the %s-token "
            "generation budget; generating without forcing",
            chunk_index,
            seek,
            max_gen_len,
        )
    return plan


def forcing_prompt_ids(
    tokenizer: TokenizerProtocol,
    open_keys: list[tuple[int, int]],
    previous_tokens: list[int],
    previous_seek: float,
    seek: float,
    overlap_window: OverlapWindow | None,
    max_gen_len: int,
    chunk_index: int,
    stderr_logger: logging.Logger | None = None,
) -> list[int]:
    return list(
        forcing_prompt_plan(
            tokenizer,
            open_keys,
            previous_tokens,
            previous_seek,
            seek,
            overlap_window,
            max_gen_len,
            chunk_index,
            stderr_logger,
        ).prompt_ids
    )


def has_continuity_evidence(
    tokenizer: TokenizerProtocol,
    reference_tokens: list[int],
    reference_seek: float,
    seek: float,
    overlap_window: OverlapWindow,
) -> bool:
    """Return whether a previous chunk can inform this overlap boundary."""
    end = seek + overlap_window.overlap_seconds
    return bool(
        note_events_in_range(
            tokenizer,
            reference_tokens,
            reference_seek,
            seek,
            end,
        )
        or open_keys_at(tokenizer, reference_tokens, reference_seek, seek)
        or open_keys_at(tokenizer, reference_tokens, reference_seek, end)
    )


def note_events_in_range(
    tokenizer: TokenizerProtocol,
    tokens: list[int],
    chunk_seek: float,
    start: float,
    end: float,
) -> list[NoteEvent]:
    """Decode raw semantic events in the half-open absolute-time range."""
    frame_rate = tokenizer.frame_rate
    start_tick = round(chunk_seek * frame_rate)
    tick = start_tick
    program: int | None = None
    velocity: int | None = None
    in_prologue = True
    events: list[NoteEvent] = []
    for token in tokens:
        event = tokenizer._vocab[token]
        if in_prologue:
            if event.type == "tie":
                in_prologue = False
            elif event.type == "program":
                program = event.value
            elif event.type == "shift":
                return []
            continue
        if event.type == "shift" and event.value > 0:
            tick = start_tick + event.value
        elif event.type == "program":
            program = event.value
        elif event.type == "velocity":
            velocity = event.value
        elif event.type == "drum":
            time = tick / frame_rate
            if start <= time < end:
                events.append(NoteEvent(True, DRUM_PROGRAM, time, 1, event.value))
        elif event.type == "pitch" and program is not None and velocity is not None:
            time = tick / frame_rate
            if start <= time < end:
                events.append(NoteEvent(False, program, time, velocity, event.value))
    return events


def open_keys_at(
    tokenizer: TokenizerProtocol,
    tokens: list[int],
    chunk_seek: float,
    target_time: float,
) -> list[tuple[int, int]]:
    """Return notes open immediately before an absolute target time."""
    frame_rate = tokenizer.frame_rate
    tick = round(chunk_seek * frame_rate)
    target_tick = round(target_time * frame_rate)
    program: int | None = None
    velocity: int | None = None
    in_prologue = True
    open_keys: set[tuple[int, int]] = set()
    for token in tokens:
        event = tokenizer._vocab[token]
        if in_prologue:
            if event.type == "tie":
                in_prologue = False
            elif event.type == "program":
                program = event.value
            elif event.type == "pitch" and program is not None:
                open_keys.add((program, event.value))
            continue
        if event.type == "shift" and event.value > 0:
            tick = round(chunk_seek * frame_rate) + event.value
            if tick >= target_tick:
                break
        elif event.type == "program":
            program = event.value
        elif event.type == "velocity":
            velocity = event.value
        elif event.type == "pitch" and program is not None and velocity is not None:
            key = (program, event.value)
            if velocity > 0:
                open_keys.add(key)
            else:
                open_keys.discard(key)
    return sorted(open_keys)


def match_verification(
    tokenizer: TokenizerProtocol,
    reference_tokens: list[int],
    reference_seek: float,
    candidate_tokens: list[int],
    candidate_seek: float,
    start: float,
    end: float,
) -> OverlapMatch:
    """Compare decoded events and sustained-note state in one interval."""
    reference = note_events_in_range(
        tokenizer,
        reference_tokens,
        reference_seek,
        start,
        end,
    )
    candidate = note_events_in_range(
        tokenizer,
        candidate_tokens,
        candidate_seek,
        start,
        end,
    )
    return match_note_events(
        reference,
        candidate,
        reference_open_start=set(
            open_keys_at(
                tokenizer,
                reference_tokens,
                reference_seek,
                start,
            )
        ),
        candidate_open_start=set(
            open_keys_at(
                tokenizer,
                candidate_tokens,
                candidate_seek,
                start,
            )
        ),
        reference_open_end=set(
            open_keys_at(
                tokenizer,
                reference_tokens,
                reference_seek,
                end,
            )
        ),
        candidate_open_end=set(
            open_keys_at(
                tokenizer,
                candidate_tokens,
                candidate_seek,
                end,
            )
        ),
    )


def resolve_overlap_window(
    tokenizer: TokenizerProtocol,
    reference_tokens: list[int],
    reference_seek: float,
    seek: float,
    base: OverlapWindow,
    *,
    target_evidence: int = 4,
    maximum_verification_seconds: float = 0.8,
    step_seconds: float = 0.1,
) -> OverlapWindow:
    """Extend verification backward without reducing replay history."""
    verification = base.verification_seconds
    while True:
        extension = verification - base.verification_seconds
        verification_start = seek + base.replay_seconds - extension
        verification_end = seek + base.overlap_seconds
        events = note_events_in_range(
            tokenizer,
            reference_tokens,
            reference_seek,
            verification_start,
            verification_end,
        )
        open_start = open_keys_at(
            tokenizer,
            reference_tokens,
            reference_seek,
            verification_start,
        )
        open_end = open_keys_at(
            tokenizer,
            reference_tokens,
            reference_seek,
            verification_end,
        )
        evidence = len(events) + len(open_start) + len(open_end)
        if evidence >= target_evidence or verification >= maximum_verification_seconds:
            return OverlapWindow(
                replay_seconds=base.replay_seconds,
                verification_seconds=verification,
            )
        verification = min(maximum_verification_seconds, verification + step_seconds)


def canonicalize_tokens(
    tokenizer: TokenizerProtocol,
    tokens: list[int],
    source_seek: float,
    output_seek: float,
    output_end: float,
) -> list[int]:
    """Rebase a shifted recovery result onto canonical chunk ownership."""
    open_keys = open_keys_at(
        tokenizer,
        tokens,
        source_seek,
        output_seek,
    )
    events = note_events_in_range(
        tokenizer,
        tokens,
        source_seek,
        output_seek,
        output_end,
    )
    return tokenizer.overlap_prompt_token_ids(
        open_keys,
        events,
        output_seek,
    )
