"""Bounded CUDA Graph dispatch for fixed-shape decode runtimes."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Protocol

import torch


class DecodeGraph(Protocol):
    """Captured decode step with caller-owned static input and state buffers."""

    def replay(self) -> torch.Tensor: ...

    def estimated_static_bytes(self) -> int: ...


def private_pool_bytes(device: torch.device) -> int:
    """Bytes the allocator holds in CUDA Graph private pools on ``device``.

    Capture routes a graph's intermediates into a private pool that outlives
    the capture, is counted by ``memory_reserved``, and is never returned by
    ``empty_cache`` while the graph is alive. No session owns those bytes, so
    without this they are indistinguishable from allocator fragmentation.
    Reserved-total deltas cannot substitute: entering capture empties the
    default pool first, which nets the two movements against each other.
    """
    if device.type != "cuda":
        return 0
    index = device.index if device.index is not None else torch.cuda.current_device()
    return sum(
        int(segment.get("total_size", 0))
        for segment in torch.cuda.memory_snapshot()
        if int(segment.get("device", -1)) == index
        and tuple(segment.get("segment_pool_id", (0, 0))) != (0, 0)
    )


@dataclass
class CapturedDecodeGraph:
    """One CUDA Graph and its graph-owned static output tensor."""

    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    pool_bytes: int = 0

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        return self.output


@dataclass
class CapturedDecodeQuantum:
    """One graph replay containing several dependent decode steps."""

    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    static_input: torch.Tensor
    static_forbidden: torch.Tensor
    static_sampling_mask: torch.Tensor
    static_temporal_floors: torch.Tensor
    static_sampling_seeds: torch.Tensor | None = None
    static_sampling_positions: torch.Tensor | None = None
    # Written inside the capture, read after every replay. Not an input, so it
    # is never copied in -- the graph overwrites it each time.
    margins: torch.Tensor | None = None
    #: Activation-export buffers, written inside the capture when the export is
    #: on. `None` otherwise, and nothing is allocated for it.
    traces: tuple[torch.Tensor, ...] | None = None
    pool_bytes: int = 0

    def graph_pool_bytes(self) -> int:
        """Private-pool bytes this capture added, measured at capture time."""
        return self.pool_bytes

    def estimated_static_bytes(self) -> int:
        """Lower bound for tensors retained by this graph cache entry."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.output,
                self.static_input,
                self.static_forbidden,
                self.static_sampling_mask,
                self.static_temporal_floors,
                self.static_sampling_seeds,
                self.static_sampling_positions,
                self.margins,
            )
            if tensor is not None
        )

    def replay(
        self,
        sequence: torch.Tensor,
        forbidden_tokens: torch.Tensor,
        sampling_mask: torch.Tensor,
        temporal_floors: torch.Tensor,
        sampling_seeds: torch.Tensor | None = None,
        sampling_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.static_input.copy_(sequence)
        self.static_forbidden.copy_(forbidden_tokens)
        self.static_sampling_mask.copy_(sampling_mask)
        self.static_temporal_floors.copy_(temporal_floors)
        if self.static_sampling_seeds is not None:
            if sampling_seeds is None or sampling_positions is None:
                raise ValueError("Graph sampling requires seeds and positions")
            self.static_sampling_seeds.copy_(sampling_seeds)
            self.static_sampling_positions.copy_(sampling_positions)
        self.graph.replay()
        return self.output


def capture_decode_graph(call, device: torch.device) -> CapturedDecodeGraph:
    """Capture a warmed, fixed-address decode call exactly once."""
    if device.type != "cuda":
        raise ValueError("CUDA Graph capture requires a CUDA device")
    graph = torch.cuda.CUDAGraph()
    pool_before = private_pool_bytes(device)
    with torch.cuda.graph(graph):
        output = call()
    if not isinstance(output, torch.Tensor):
        raise TypeError("captured decode call must return one tensor")
    pool_after = private_pool_bytes(device)
    return CapturedDecodeGraph(graph, output, max(0, pool_after - pool_before))


@dataclass
class DecodeGraphVariantTelemetry:
    hits: int = 0
    misses: int = 0
    captures: int = 0
    replays: int = 0
    evictions: int = 0
    evicted_static_bytes: int = 0
    pool_bytes: int = 0
    evicted_pool_bytes: int = 0


@dataclass
class DecodeGraphTelemetry:
    attempts: int = 0
    captures: int = 0
    replays: int = 0
    graph_refusals: int = 0
    evictions: int = 0
    rejections: Counter[str] = field(default_factory=Counter)
    variants: dict[str, DecodeGraphVariantTelemetry] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class DecodeGraphKey:
    """All shape and sampling fields that determine graph reusability."""

    width: int
    state_width: int
    bucket_end: int
    quantum: int
    cfg_coef: float
    temperature: float

    @property
    def label(self) -> str:
        return (
            f"w{self.width}:sw{self.state_width}:b{self.bucket_end}:"
            f"q{self.quantum}:cfg{self.cfg_coef:g}:"
            f"mixed-sampling:temp{self.temperature:g}"
        )


@dataclass(frozen=True)
class DecodeStateEligibility:
    eligible: bool
    reason: str


def inspect_decode_state(model_state: dict[str, dict[str, object]]) -> DecodeStateEligibility:
    """Reject state that cannot retain fixed addresses and shapes during replay."""
    for module_name, state in model_state.items():
        if isinstance(state.get("offset"), int):
            return DecodeStateEligibility(
                False,
                f"{module_name}:python-kv-offset",
            )
        for name, value in state.items():
            if value is None and name in {"row_offsets", "valid_starts"}:
                return DecodeStateEligibility(
                    False,
                    f"{module_name}:{name}-not-materialized",
                )
    return DecodeStateEligibility(True, "fixed-tensor-state")


class DecodeGraphDispatcher:
    """Cache fixed-shape graph runners without unbounded CUDA graph pools."""

    def __init__(self, max_entries: int = 16, *, evict_on_full: bool = False):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self.evict_on_full = evict_on_full
        self.telemetry = DecodeGraphTelemetry()
        self._entries: OrderedDict[Hashable, DecodeGraph] = OrderedDict()

    @staticmethod
    def _label(key: Hashable) -> str:
        return key.label if isinstance(key, DecodeGraphKey) else repr(key)

    def _variant(self, key: Hashable) -> DecodeGraphVariantTelemetry:
        label = self._label(key)
        return self.telemetry.variants.setdefault(label, DecodeGraphVariantTelemetry())

    def lookup(self, key: Hashable) -> DecodeGraph | None:
        self.telemetry.attempts += 1
        entry = self._entries.get(key)
        if entry is None:
            self._variant(key).misses += 1
            return None
        self._variant(key).hits += 1
        self._entries.move_to_end(key)
        return entry

    def install(self, key: Hashable, graph: DecodeGraph) -> bool:
        if key in self._entries:
            self._entries[key] = graph
            self._entries.move_to_end(key)
            return True
        if len(self._entries) >= self.max_entries:
            if self.evict_on_full:
                evicted_key, evicted_graph = self._entries.popitem(last=False)
                self.telemetry.evictions += 1
                variant = self._variant(evicted_key)
                variant.evictions += 1
                variant.evicted_static_bytes += self._estimated_static_bytes(
                    evicted_graph
                )
                variant.evicted_pool_bytes += self._graph_pool_bytes(evicted_graph)
            else:
                self.reject("graph-cache-full")
                return False
        self._entries[key] = graph
        self.telemetry.captures += 1
        installed = self._variant(key)
        installed.captures += 1
        installed.pool_bytes += self._graph_pool_bytes(graph)
        return True

    def replay(self, key: Hashable) -> torch.Tensor | None:
        graph = self.lookup(key)
        if graph is None:
            return None
        self.telemetry.replays += 1
        self._variant(key).replays += 1
        return graph.replay()

    @staticmethod
    def _estimated_static_bytes(graph: DecodeGraph) -> int:
        estimate = getattr(graph, "estimated_static_bytes", None)
        return int(estimate()) if estimate is not None else 0

    @staticmethod
    def _graph_pool_bytes(graph: DecodeGraph) -> int:
        measured = getattr(graph, "graph_pool_bytes", None)
        return int(measured()) if measured is not None else 0

    def variant_telemetry(self) -> dict[str, dict[str, int]]:
        return {
            key: {
                "hits": value.hits,
                "misses": value.misses,
                "captures": value.captures,
                "replays": value.replays,
                "evictions": value.evictions,
                "evicted_static_bytes": value.evicted_static_bytes,
                "pool_bytes": value.pool_bytes,
                "evicted_pool_bytes": value.evicted_pool_bytes,
            }
            for key, value in self.telemetry.variants.items()
        }

    def cache_telemetry(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "capacity": self.max_entries,
            "static_bytes": sum(
                self._estimated_static_bytes(graph)
                for graph in self._entries.values()
            ),
            # Cached graphs only. Eviction drops the entry but the pool lives
            # until the graph object itself dies, so this understates the
            # device total whenever the cache has evicted anything.
            "pool_bytes": sum(
                self._graph_pool_bytes(graph) for graph in self._entries.values()
            ),
        }

    def reject(self, reason: str) -> None:
        """Count a refusal. The caller then **raises** -- see `_reject`.

        Named for what it is. This was `eager_fallbacks`, which reads as
        "silently degraded to the per-token path" and is the opposite of what
        happens: `replay` never changes to eager once entered, it reports the
        broken invariant and the decode fails. Every reason it can carry is
        unreachable by construction, so a non-zero value is a defect, not a
        slow path. Since G1-3 there is no per-token decode in graph mode at
        all: the session picks a mode when it opens and reaching the other one
        raises.
        """
        self.telemetry.graph_refusals += 1
        self.telemetry.rejections[reason] += 1

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


class ContinuousDecodeGraphRuntime:
    """Bounded multi-token CUDA Graph cache for a fixed continuous session.

    Graph entries retain the session's state addresses and a stable row-index
    tensor per active width. The caller updates static inputs and row-index
    contents between replays; model state advances inside the captured graph.
    """

    def __init__(
        self,
        model,
        model_state: dict[str, dict[str, object]],
        *,
        bucket_size: int,
        capture_bucket_size: int | None = None,
        max_batch_size: int = 8,
        max_graphs: int = 16,
    ) -> None:
        if bucket_size < 1:
            raise ValueError("bucket_size must be positive")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.model = model
        self.model_state = model_state
        self.bucket_size = bucket_size
        self.capture_bucket_size = capture_bucket_size or bucket_size
        if self.capture_bucket_size < self.bucket_size:
            raise ValueError("capture bucket size cannot be smaller than KV bucket")
        self.max_batch_size = max_batch_size
        self.dispatcher = DecodeGraphDispatcher(
            max_entries=max_graphs,
            evict_on_full=True,
        )
        self._capture_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def drain_capture_gpu_ms(self) -> float:
        """Return finished capture time without synchronizing the stream."""
        completed = 0.0
        pending = []
        for start, end in self._capture_events:
            if end.query():
                completed += start.elapsed_time(end)
            else:
                pending.append((start, end))
        self._capture_events = pending
        return completed

    def _host_offset_state(self) -> dict[str, object] | None:
        return next(
            (
                state
                for state in self.model_state.values()
                if state.get("row_offsets_host") is not None
            ),
            None,
        )

    def _restore_host_offsets(
        self,
        snapshots: tuple[tuple[list[int], tuple[int, ...]], ...],
    ) -> None:
        for offsets, values in snapshots:
            offsets[:] = values

    def _snapshot_host_offsets(
        self,
    ) -> tuple[tuple[list[int], tuple[int, ...]], ...]:
        snapshots = []
        seen: set[int] = set()
        for state in self.model_state.values():
            offsets = state.get("row_offsets_host")
            if offsets is None or id(offsets) in seen:
                continue
            seen.add(id(offsets))
            snapshots.append((offsets, tuple(offsets)))
        return tuple(snapshots)

    def advance_host_offsets(
        self,
        rows: tuple[int, ...],
        increment: int,
    ) -> None:
        """Mirror graph-side tensor offset increments for future captures."""
        seen: set[int] = set()
        for state in self.model_state.values():
            offsets = state.get("row_offsets_host")
            if (
                offsets is None
                or not state.get("shared_metadata_owner", True)
                or id(offsets) in seen
            ):
                continue
            seen.add(id(offsets))
            for row in rows:
                offsets[row] += increment

    def _bucket_end(
        self,
        rows: tuple[int, ...],
        quantum: int,
        state: dict[str, object] | None = None,
    ) -> int | None:
        """The captured-graph bucket these rows fit in, or `None`.

        `state` is the caller's already-resolved offset state. `replay` holds
        one and passes it: resolving it is a linear scan over every layer, and
        it used to run twice per replay -- once inside here, once again right
        after -- for the same answer.
        """
        if state is None:
            state = self._host_offset_state()
        if state is None:
            return None
        offsets = state["row_offsets_host"]
        maximum_offset = max(offsets[row] for row in rows)
        maximum_end = maximum_offset + quantum
        bucket_end = (
            (maximum_end + self.capture_bucket_size - 1) // self.capture_bucket_size
        ) * self.capture_bucket_size
        # `padded_capacity`, not `capacity`, and deliberately: this bounds a
        # graph replay against the KV's *physical* extent. Reaching into the
        # block padding cannot overrun the transformer's position table --
        # `position_capacity_for` keeps at least 128 positions of headroom
        # against a padding of at most `BLOCK_TOKENS - 1` = 15 -- and the
        # decode loop is bounded by the slot, not by this. Narrowing it to
        # `capacity` would only refuse graphs in the last 15 positions.
        capacity = state["kv"].padded_capacity
        if maximum_end > bucket_end or maximum_end > capacity:
            return None
        return min(bucket_end, capacity)

    def can_replay(self, rows: tuple[int, ...], quantum: int) -> bool:
        """Whether the current KV tails can hold one complete graph quantum."""
        return self._bucket_end(rows, quantum) is not None

    #: Chosen-token margins from the most recent replay, `[quantum, rows]`.
    #: Read immediately after `replay` returns; the next replay overwrites it.
    last_margins: torch.Tensor | None = None
    #: Activation-export buffers from the most recent replay, or None.
    last_traces: tuple[torch.Tensor, ...] | None = None

    def _allocate_trace_buffers(self, quantum: int, sequence: torch.Tensor):
        """Static outputs for the activation export, or None when it is off.

        Allocated per captured graph, so the cost is per cache entry rather
        than per replay: at production geometry -- quantum 4, 20 rows, vocab
        1395, dim 768 -- 354 KB each and 5.7 MB across the 16-entry cache. The
        export's real cost is the disk it fills, which is its purpose.

        Shapes come from the model rather than from a probe call: `card` is the
        vocabulary and the attention projection's input width is the model
        dimension, both fixed for a loaded model.
        """
        if not getattr(self.model, "_activation_export", None):
            return None
        rows = sequence.shape[0]
        device = sequence.device
        dimension = int(
            self.model.transformer.layers[0].self_attn.in_proj_weight.shape[1]
        )
        vocabulary = int(self.model.card)
        top = min(16, vocabulary)
        return (
            torch.empty((quantum, rows, vocabulary), dtype=torch.float16, device=device),
            torch.empty((quantum, rows, top), dtype=torch.long, device=device),
            torch.empty((quantum, rows, top), dtype=torch.float32, device=device),
            torch.empty((quantum, rows), dtype=torch.float32, device=device),
            torch.empty((quantum, rows, dimension), dtype=torch.float16, device=device),
        )

    def _reject(self, reason: str) -> None:
        self.dispatcher.reject(reason)
        raise RuntimeError(f"CUDA Graph quantum is unavailable: {reason}")

    def replay(
        self,
        sequence: torch.Tensor,
        forbidden_tokens: torch.Tensor,
        *,
        active_rows: torch.Tensor,
        state_rows: torch.Tensor,
        state_rows_host: tuple[int, ...],
        quantum: int,
        cfg_coef: float,
        temperature: float = 1.0,
        sampling_mask: torch.Tensor,
        sampling_seeds: torch.Tensor,
        sampling_positions: torch.Tensor,
        temporal_shift_values: torch.Tensor,
        temporal_floors: torch.Tensor,
    ) -> torch.Tensor:
        """Replay one strict 4/8-token graph quantum.

        Graph-enabled continuous decode never silently changes to eager after
        entering this operation. A missing graph is captured, and an invalid
        shape is reported to the scheduler so the caller can use the explicit
        graph-off runtime instead.
        """
        if quantum not in {4, 8}:
            self._reject("unsupported-quantum")
        if sequence.shape[0] > self.max_batch_size:
            self._reject("batch-too-wide")
        if sequence.device.type != "cuda":
            self._reject("not-cuda")
        if sampling_mask.shape != sampling_seeds.shape or (
            sampling_mask.shape != sampling_positions.shape
        ):
            self._reject("sampling-metadata-shape")
        if temporal_shift_values.shape != (self.model.card,) or (
            temporal_floors.shape != sampling_mask.shape
        ):
            self._reject("temporal-metadata-shape")
        state = self._host_offset_state()
        bucket_end = self._bucket_end(state_rows_host, quantum, state)
        if bucket_end is None:
            self._reject("kv-capacity")
        assert state is not None
        key = DecodeGraphKey(
            width=sequence.shape[0],
            state_width=state_rows.shape[0],
            bucket_end=bucket_end,
            quantum=quantum,
            cfg_coef=float(cfg_coef),
            temperature=float(temperature),
        )
        graph = self.dispatcher.lookup(key)
        if graph is None:
            static_input = torch.empty_like(sequence)
            static_forbidden = torch.empty_like(forbidden_tokens)
            static_sampling_mask = torch.empty_like(sampling_mask)
            static_temporal_floors = torch.empty_like(temporal_floors)
            static_sampling_seeds = torch.empty_like(sampling_seeds)
            static_sampling_positions = torch.empty_like(sampling_positions)
            output = torch.empty(
                (quantum, sequence.shape[0]),
                dtype=torch.long,
                device=sequence.device,
            )
            # Chosen-token margins, one per step per row. 320 bytes at
            # production width, and the reason a recovery row can decode inside
            # a graph at all: the margin is computed on device here instead of
            # by a per-step Python callback the replay cannot run.
            margins = torch.empty(
                (quantum, sequence.shape[0]),
                dtype=torch.float32,
                device=sequence.device,
            )
            traces = self._allocate_trace_buffers(quantum, sequence)
            snapshots = self._snapshot_host_offsets()

            def decode_quantum() -> torch.Tensor:
                next_input = static_input
                for step in range(quantum):
                    next_tokens = self.model._sample_next_token(
                        next_input,
                        {},
                        self.model_state,
                        first_step=False,
                        use_sampling=True,
                        temp=temperature,
                        cfg_coef=cfg_coef,
                        forbidden_tokens=static_forbidden,
                        sample_mask=static_sampling_mask,
                        sampling_seeds=static_sampling_seeds,
                        sampling_positions=static_sampling_positions + step,
                        temporal_shift_values=temporal_shift_values,
                        temporal_floors=static_temporal_floors,
                        margin_out=margins[step],
                        trace_out=(
                            None if traces is None
                            else [None if t is None else t[step] for t in traces]
                        ),
                    )
                    output[step].copy_(next_tokens)
                    sampled_shifts = temporal_shift_values.index_select(
                        0, next_tokens
                    )
                    static_temporal_floors.copy_(
                        torch.maximum(
                            static_temporal_floors,
                            sampled_shifts,
                        )
                    )
                    next_input = next_tokens.view(-1, 1)
                    from muscriptor.modules.streaming import increment_state_rows

                    increment_state_rows(
                        self.model.transformer,
                        self.model_state,
                        state_rows,
                        increment=1,
                        host_rows=state_rows_host,
                    )
                return output

            capture_start = torch.cuda.Event(enable_timing=True)
            capture_end = torch.cuda.Event(enable_timing=True)
            capture_start.record()
            try:
                captured = capture_decode_graph(decode_quantum, sequence.device)
            finally:
                capture_end.record()
                self._restore_host_offsets(snapshots)
            self._capture_events.append((capture_start, capture_end))
            graph = CapturedDecodeQuantum(
                captured.graph,
                captured.output,
                static_input,
                static_forbidden,
                static_sampling_mask,
                static_temporal_floors,
                static_sampling_seeds,
                static_sampling_positions,
                margins,
                traces,
                captured.pool_bytes,
            )
            if not self.dispatcher.install(key, graph):
                raise RuntimeError("CUDA Graph cache rejected a captured graph")
        self.dispatcher.telemetry.replays += 1
        output = graph.replay(
            sequence,
            forbidden_tokens,
            sampling_mask,
            temporal_floors,
            sampling_seeds,
            sampling_positions,
        )
        self.advance_host_offsets(state_rows_host, quantum)
        self.last_margins = graph.margins
        self.last_traces = graph.traces
        return output
