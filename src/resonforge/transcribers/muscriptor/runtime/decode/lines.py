"""The two ways a decode session advances its rows, as one replaceable thing.

Before this, "which line is running" was a boolean (`_graph_mode`), a guard
that raised when the boolean and the code path disagreed, and an `if` inside
the step loop that tried the graph and fell through to the eager block. Three
expressions of one fact, which is three places to keep in agreement.

Here it is one object chosen when the session opens. A graph session holds a
`GraphLine`, which advances a whole quantum or raises; an eager session holds
an `EagerLine`, which advances exactly one token and cannot fail for want of a
capture. **The guard disappears with the flag**: "graph decode has no
per-token fallback" stops being a check inside shared code and becomes the
simple fact that `GraphLine` has no per-token branch to reach.

## Why both take the session

Not an oversight. The coupling was measured before this split (2026-08-26):
`self.slots` is touched by 27 of the session's 61 methods, and of the twelve
methods touching the per-row tensors -- `sampling_mask`, `sampling_seeds`,
`sampling_positions`, `forbidden`, `temporal_floors` -- **none touches only
those**. A line that took the tensors instead of the session would take twelve
arguments and still reach back for `slots`, which moves coupling from fields
into signatures without removing any of it.

So the interface is honest about what a decode step needs: the session. What it
buys is that the *choice* is now one value with two implementations, and adding
a third line touches neither the loop nor the other two.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from muscriptor.generation_batch import GenerationRequest, GenerationResult
from muscriptor.modules.streaming import increment_state_rows, select_state_rows

#: Quanta a captured decode graph is built for, mirrored from the session so a
#: line can answer "is this step mine" without importing it back.
GRAPH_QUANTUM_FAMILY = frozenset({4, 8})


class StepOutcome:
    """What one advance did, in the terms the quantum loop accumulates.

    `physical_steps` is what the loop adds; `stop` says the line consumed the
    rest of the quantum itself, which a graph replay does and a single eager
    token never does.
    """

    __slots__ = ("physical_steps", "wasted_token_rows", "widths", "stop")

    def __init__(
        self,
        *,
        physical_steps: int,
        wasted_token_rows: int = 0,
        widths: dict[int, int] | None = None,
        stop: bool = False,
    ) -> None:
        self.physical_steps = physical_steps
        self.wasted_token_rows = wasted_token_rows
        self.widths = widths or {}
        self.stop = stop


class DecodeLine(Protocol):
    """How a session advances the rows that have work."""

    name: str

    def advance(
        self,
        session,
        active: tuple[int, ...],
        *,
        remaining: int,
        completed: Callable[[GenerationRequest, GenerationResult], None],
        released: Callable[[int, int], None] | None,
        checkpointed: Callable[[GenerationRequest, GenerationResult], None] | None,
    ) -> StepOutcome: ...


class GraphLine:
    """Replay a captured quantum. No per-token path, by construction.

    A session that cannot replay -- a quantum outside the captured family, or a
    row carrying a trace collector a replay cannot feed -- has no smaller step
    to fall back to, because the capture is what the whole session was opened
    for. Raising here is the same refusal the old flag-and-guard pair made, in
    the one place that can actually know.
    """

    name = "graph"

    def advance(
        self,
        session,
        active: tuple[int, ...],
        *,
        remaining: int,
        completed,
        released,
        checkpointed,
    ) -> StepOutcome:
        candidates = active if session._rows_allow_graph(active) else ()
        tokens = (
            session._try_cuda_graph_quantum(candidates, quantum=remaining)
            if candidates and remaining in GRAPH_QUANTUM_FAMILY
            else None
        )
        if tokens is None:
            raise RuntimeError(
                "graph decode reached the per-token path: "
                f"active={len(active)} remaining={remaining} "
                f"width={session.width}. The mode is chosen when the session "
                "opens and has no fallback; a quantum outside the captured "
                "family, or a row carrying a collector a replay cannot feed, "
                "has to select eager decode for the whole process instead."
            )
        steps, wasted, widths = session._consume_graph_quantum(
            candidates, tokens, completed, released, checkpointed
        )
        # A replay advances the whole quantum, so there is nothing left of it
        # for the loop to iterate.
        return StepOutcome(
            physical_steps=steps,
            wasted_token_rows=wasted,
            widths=widths,
            stop=True,
        )


class EagerLine:
    """Sample one token for every active row. Cannot fail for want of a capture."""

    name = "eager"

    def advance(
        self,
        session,
        active: tuple[int, ...],
        *,
        remaining: int,
        completed,
        released,
        checkpointed,
    ) -> StepOutcome:
        del remaining
        state_rows, state_rows_host = session._active_state_rows(active)
        select_state_rows(
            session.model_state, state_rows, host_rows=state_rows_host
        )
        active_rows = torch.tensor(active, device=session.device)
        with session.lm.autocast:
            next_tokens = session.lm._sample_next_token(
                session.sequence.index_select(0, active_rows),
                session.conditions,
                session.model_state,
                first_step=False,
                use_sampling=True,
                temp=session.temperature,
                top_k=0,
                top_p=0.0,
                cfg_coef=session.cfg_coef,
                forbidden_tokens=session.forbidden.index_select(0, active_rows),
                sample_mask=session.sampling_mask.index_select(0, active_rows),
                generator=None,
                sampling_seeds=session.sampling_seeds.index_select(0, active_rows),
                sampling_positions=session.sampling_positions.index_select(
                    0, active_rows
                ),
                trace_collectors=tuple(
                    session.slots[index].row.request.trace_collector
                    for index in active
                ),
                trace_contexts=tuple(
                    session.slots[index].row.request.trace_context
                    for index in active
                ),
                temporal_shift_values=session.temporal_shift_values,
                temporal_floors=session.temporal_floors.index_select(0, active_rows),
            )
        sampled_shifts = session.temporal_shift_values.index_select(0, next_tokens)
        session.temporal_floors[active_rows] = torch.maximum(
            session.temporal_floors.index_select(0, active_rows), sampled_shifts
        )
        session.sequence[active_rows, 0] = next_tokens
        session.sampling_positions[active_rows] += session.sampling_mask.index_select(
            0, active_rows
        ).long()
        increment_state_rows(
            session.lm.transformer,
            session.model_state,
            state_rows,
            host_rows=state_rows_host,
        )
        next_tokens_host = next_tokens.tolist()
        for compact_index, index in enumerate(active):
            row = session.slots[index].row
            assert row is not None
            token = next_tokens_host[compact_index]
            if not row.observe_temporal_token(token):
                continue
            row.last_token = token
            row.steps += 1
            row.emitted_eos = token == session.eos_id
            if not row.emitted_eos:
                row.tokens.append(token)
                session._observe_guard_token(session.slots[index], token)
            row.finished = row.finished or (
                row.emitted_eos or row.steps >= row.request.max_gen_len
            )
        if checkpointed is not None:
            session.collect_checkpoints(checkpointed)
        session.collect_finished(completed, released)
        return StepOutcome(physical_steps=1, widths={len(active): 1})
