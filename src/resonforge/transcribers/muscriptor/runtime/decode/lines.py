"""The two ways a decode session launches a quantum, as one replaceable thing.

One protocol since 2026-08-30: every session decodes via `launch_quantum` /
`consume_quantum` (S5). A line owns only the *launch* half -- produce this
quantum's tokens as a `[quantum, rows]` device tensor with no host read -- and
both lines share one consume (`_consume_graph_quantum`), so EOS, temporal
floor, guard, checkpoint retention and waste accounting cannot drift between
them. Before this, eager decode was a second full driving protocol
(`run_quantum` stepping one synced token at a time); the scheduler carried an
if/else per dispatch, and the eager consume loop was a hand-kept copy of the
graph one.

The eager line samples on device and feeds `sequence` back tensor-to-tensor,
so it never needed the per-token host read -- the sync was only for its inline
consume. Deferring that consume gives eager the graph line's overshoot
semantics: a row past its stop inside a quantum costs discarded tokens
(`wasted_token_rows`), never different output -- the consume replays the same
per-token break points.

## Why both take the session

Not an oversight. The coupling was measured before this split (2026-08-26):
`self.slots` is touched by 27 of the session's 61 methods, and of the twelve
methods touching the per-row tensors -- `sampling_mask`, `sampling_seeds`,
`sampling_positions`, `forbidden`, `temporal_floors` -- **none touches only
those**. A line that took the tensors instead of the session would take twelve
arguments and still reach back for `slots`, which moves coupling from fields
into signatures without removing any of it.
"""

from __future__ import annotations

from typing import Protocol

import torch

#: Quanta a captured decode graph is built for, mirrored from the session so a
#: line can answer "is this step mine" without importing it back.
GRAPH_QUANTUM_FAMILY = frozenset({4, 8})


class DecodeLine(Protocol):
    """How a session launches one quantum of decode work."""

    name: str

    def launch(
        self,
        session,
        active: tuple[int, ...],
        *,
        quantum: int,
    ) -> torch.Tensor:
        """Produce `[quantum, rows]` tokens for the active rows, device-side."""
        ...


class GraphLine:
    """Replay a captured quantum. No per-token path, by construction.

    A session that cannot replay -- a quantum outside the captured family, or a
    row carrying a trace collector a replay cannot feed -- has no smaller step
    to fall back to, because the capture is what the whole session was opened
    for. Raising here is the same refusal the old flag-and-guard pair made, in
    the one place that can actually know.
    """

    name = "graph"

    def launch(
        self,
        session,
        active: tuple[int, ...],
        *,
        quantum: int,
    ) -> torch.Tensor:
        if quantum not in GRAPH_QUANTUM_FAMILY:
            raise ValueError(
                f"graph decode runs quanta of {sorted(GRAPH_QUANTUM_FAMILY)}, "
                f"not {quantum}; a session that needs another size has to be "
                "opened with graph decode off"
            )
        tokens = (
            session._try_cuda_graph_quantum(active, quantum=quantum)
            if session._rows_allow_graph(active)
            else None
        )
        if tokens is None:
            raise RuntimeError(
                "graph decode reached the per-token path: "
                f"active={len(active)} quantum={quantum} "
                f"width={session.width}. The mode is chosen when the session "
                "opens and has no fallback; a quantum outside the captured "
                "family, or a row carrying a collector a replay cannot feed, "
                "has to select eager decode for the whole process instead."
            )
        return tokens


class EagerLine:
    """Sample `quantum` tokens per row on device. Cannot fail for want of a capture.

    Each step feeds `sequence` back tensor-to-tensor, so the loop never reads a
    token on the host; trace collectors still observe inside
    `_sample_next_token`, live, which is why collector rows select this line.
    """

    name = "eager"

    def launch(
        self,
        session,
        active: tuple[int, ...],
        *,
        quantum: int,
    ) -> torch.Tensor:
        from muscriptor.modules.streaming import (
            increment_state_rows,
            select_state_rows,
        )

        state_rows, state_rows_host = session._active_state_rows(active)
        active_rows = session._active_rows_tensor(active)
        forbidden = session.forbidden.index_select(0, active_rows)
        sample_mask = session.sampling_mask.index_select(0, active_rows)
        seeds = session.sampling_seeds.index_select(0, active_rows)
        collectors = [
            session.slots[index].row.request.trace_collector for index in active
        ]
        contexts = [
            session.slots[index].row.request.trace_context for index in active
        ]
        # Observation happens inside `_sample_next_token`, at sample time --
        # but the consume side skips a row's overshoot (steps past its EOS or
        # length cap), and graph-line collectors observe from consume, so a
        # blind eager quantum would record one extra row per finish. Only
        # when something observes (debug capture): read each step's tokens
        # and null the finished rows' observers for the remaining steps. The
        # per-step host read exists solely for this parity; production eager
        # carries no collectors and stays host-free. Residual: a temporal
        # floor break still observes up to quantum-1 extra steps -- that
        # check lives on host row state the launch must not mutate.
        observed = any(c is not None for c in collectors) or any(
            c is not None for c in contexts
        )
        remaining_by_row = [
            session.slots[index].row.request.max_gen_len
            - session.slots[index].row.steps
            for index in active
        ]
        steps: list[torch.Tensor] = []
        for _ in range(quantum):
            select_state_rows(
                session.model_state, state_rows, host_rows=state_rows_host
            )
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
                    forbidden_tokens=forbidden,
                    sample_mask=sample_mask,
                    generator=None,
                    sampling_seeds=seeds,
                    sampling_positions=session.sampling_positions.index_select(
                        0, active_rows
                    ),
                    trace_collectors=tuple(collectors),
                    trace_contexts=tuple(contexts),
                    temporal_shift_values=session.temporal_shift_values,
                    temporal_floors=session.temporal_floors.index_select(
                        0, active_rows
                    ),
                )
            sampled_shifts = session.temporal_shift_values.index_select(
                0, next_tokens
            )
            session.temporal_floors[active_rows] = torch.maximum(
                session.temporal_floors.index_select(0, active_rows),
                sampled_shifts,
            )
            session.sequence[active_rows, 0] = next_tokens
            session.sampling_positions[active_rows] += sample_mask.long()
            increment_state_rows(
                session.lm.transformer,
                session.model_state,
                state_rows,
                host_rows=state_rows_host,
            )
            steps.append(next_tokens)
            if observed:
                for i, token in enumerate(next_tokens.tolist()):
                    remaining_by_row[i] -= 1
                    if token == session.eos_id or remaining_by_row[i] <= 0:
                        collectors[i] = None
                        contexts[i] = None
        return torch.stack(steps)
