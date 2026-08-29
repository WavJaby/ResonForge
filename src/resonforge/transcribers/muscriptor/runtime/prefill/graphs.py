"""Captured prefill forwards, and the reusable states they are captured against.

Decode has had CUDA Graphs since G1, prefill hasn't, and the reason was never the forward -- it was addressing.
A prefill allocates a fresh `ModelState` per admission and a graph bakes device pointers, so a capture of one admission addressed the PREVIOUS admission's memory.
Keying a cache on those addresses doesn't work: identical lease addresses almost never recur, so it pays far more captures than it serves replays.

Three changes made this possible, each measured rather than argued:
  * a pool-borrowed prefill writes through its BLOCK TABLE, so pages are the table's contents, not a baked pointer (`complete_kv`);
  * `KVArena.relet` points an existing table tensor at a new lease IN PLACE, so the arena and any graph built against it outlive the pages.
    A graph captured on one lease and replayed after a relet onto a disjoint lease wrote every page of the new lease and left all 462 pages of the old one untouched;
  * `last_index` names the true final position, so a prompt can be padded up to a bucket without sampling from a pad.

! What it buys depends entirely on the host and the dev box is the wrong one. A capture reclaims launch gap, so it pays where the forward is launch-bound (BAIR, fp16)
  and is worth almost nothing where the forward already saturates the card (this box, fp32, compute-6.1). NEVER price this here.

! Bucketing is a budget, not a detail. A capture is expensive and a song's prefills concentrate in a couple of buckets, so capturing every bucket costs more than it saves and capturing the head doesn't.
  Both sets of figures: docs/HANDOFF.md.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch
from muscriptor.modules.paged_kv import KVPagesExhausted, record_shortfall
from muscriptor.modules.streaming import ModelState

from resonforge.transcribers.muscriptor.runtime.prefill import (
    memory as prefill_memory,
)

#: Positions rounded up to a multiple of this before a capture. Coarse on purpose, and measured rather than chosen:
#: scanning `--prompt-tokens` 9 -> 512 at width 1 moved the forward's wall by 0.9% (8.20 -> 8.13 ms) while device occupancy rose, because the time is set by launch count and launches follow the layer count.
#: So padding to a bucket is close to free and the bucket COUNT is what costs -- see the module header.
BUCKET_POSITIONS = 32


class _AddressesMoved(RuntimeError):
    """A captured entry's baked pointers no longer point where it left them."""


def bucket_for(positions: int) -> int:
    """The capture length a prompt of this many positions is padded up to."""
    if positions < 1:
        raise ValueError("a prefill covers at least one position")
    return -(-int(positions) // BUCKET_POSITIONS) * BUCKET_POSITIONS


@dataclass(frozen=True)
class PrefillGraphKey:
    """Everything a captured prefill forward fixes.

    Shapes and sampling mode both, for the same reason the decode key carries
    them: a graph is a fixed sequence of launches over fixed buffers, so any
    field that changes which kernels run is part of its identity, not an input.
    """

    rows: int
    positions: int
    prepend_length: int
    cfg_coef: float
    temperature: float
    sampling: bool

    @property
    def label(self) -> str:
        return (
            f"r{self.rows}:p{self.positions}:c{self.prepend_length}:"
            f"cfg{self.cfg_coef:g}:t{self.temperature:g}:"
            f"{'sampled' if self.sampling else 'greedy'}"
        )


@dataclass
class PrefillGraphTelemetry:
    captures: int = 0
    replays: int = 0
    hits: int = 0
    misses: int = 0
    entries: int = 0
    refusals: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        return {
            "captures": self.captures,
            "replays": self.replays,
            "hits": self.hits,
            "misses": self.misses,
            "entries": self.entries,
            "refusals": dict(self.refusals),
        }


@dataclass
class _Buffers:
    """Every tensor a replay writes before it runs."""

    sequence: torch.Tensor
    forbidden: torch.Tensor
    temporal_shift_values: torch.Tensor
    temporal_floors: torch.Tensor
    sample_mask: torch.Tensor
    last_index: torch.Tensor
    conditions: dict[str, tuple[torch.Tensor, torch.Tensor]]
    sampling_seeds: torch.Tensor | None
    sampling_positions: torch.Tensor | None


@dataclass
class _Entry:
    """One captured forward, the state it addresses, and its input buffers.

    The state is the entry's, not the caller's. That inversion is the whole
    module: a prefill state used to live for one admission and be thrown away,
    and now it outlives every admission that replays this graph while its pages
    turn over underneath it.
    """

    key: PrefillGraphKey
    state: ModelState
    buffers: _Buffers
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    busy: bool = False
    #: Every device pointer the capture baked that this module does not own,
    #: recorded so a replay can refuse rather than read somebody else's memory.
    #: A graph is only valid while these hold, and "these hold" is exactly the
    #: claim the whole module rests on -- so it is checked, not asserted in a
    #: comment. Cheap: one `data_ptr()` per layer per replay, no device work.
    addresses: tuple[int, ...] = ()

    def current_addresses(self) -> tuple[int, ...]:
        pointers: list[int] = []
        for layer in self.state.values():
            arena = layer.get("kv")
            if arena is not None:
                pointers.append(arena.storage.data_ptr())
                pointers.append(arena.table.data_ptr())
            offsets = layer.get("offsets")
            if offsets is not None:
                pointers.append(offsets.data_ptr())
        return tuple(pointers)


class PrefillGraphCache:
    """Captured prefill forwards for one loaded model.

    Bounded by `max_entries` across all keys. Nothing is evicted: an entry owns a state whose arenas may be relet at any time by a row that hasn't installed yet, so dropping one isn't a local decision.
    A full cache refuses instead and the refusal is counted -- non-zero `graph-cache-full` says the bucket budget is wrong, not that the cache is working.

    `max_entries` trades a fixed capture cost against a per-replay saving, and which way it pays depends on how much work the model sees.
    Cost is paid once per entry PER LOADED MODEL, and a model outlives every song in a run, so the saving accumulates across songs while the bill doesn't -- break-even ~1.3 songs a model load.
    ! a one-song manifest is therefore the worst case it can be measured at, and measuring it there is how a cache that pays looks like one that doesn't.

    16 rather than 6: a smaller cache was tried on the theory it would earn more per entry, and the hit rate collapsed far faster than the capture bill fell. Arms: docs/HANDOFF.md.
    """

    def __init__(self, *, max_entries: int = 16) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self.telemetry = PrefillGraphTelemetry()
        self._entries: dict[PrefillGraphKey, list[_Entry]] = {}

    # -- lookup -----------------------------------------------------------
    def free_entry(self, key: PrefillGraphKey) -> _Entry | None:
        """A captured entry for this key that no row is still reading.

        A prefill state outlives its forward: a batched prefill is installed one
        row at a time, and a row that is refused is released later still. So
        "captured for this key" and "usable right now" are two questions.
        """
        for entry in self._entries.get(key, ()):
            if not entry.busy:
                self.telemetry.hits += 1
                entry.busy = True
                return entry
        self.telemetry.misses += 1
        return None

    def refuse(self, reason: str) -> None:
        """Count a prefill this cache did not serve. Never raises.

        Unlike the decode dispatcher's refusals, these are **ordinary**: a
        prompt longer than any bucket, a traced request, a cold key. The eager
        path is correct and always available, so a refusal is a cost rather
        than a defect -- but an unread counter is how a cache that never hits
        goes unnoticed for a month.
        """
        self.telemetry.refusals[reason] += 1

    @property
    def entry_count(self) -> int:
        return sum(len(entries) for entries in self._entries.values())

    def has_room(self) -> bool:
        return self.entry_count < self.max_entries

    # -- capture ----------------------------------------------------------
    def capture(
        self,
        key: PrefillGraphKey,
        state: ModelState,
        buffers: _Buffers,
        forward,
    ) -> _Entry | None:
        """Capture one forward against `state`, or refuse and leave it eager.

        The caller has already run `forward` eagerly at least once against this
        exact state, which is the warm-up a capture needs and also the reason
        capture happens here rather than at cache construction: the first
        admission of a bucket pays for the graph and gets its own answer out of
        the same work.
        """
        if not self.has_room():
            self.refuse("graph-cache-full")
            return None
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                output = forward()
        except RuntimeError as error:
            # A capture cannot contain a synchronisation, and the message names
            # the wrong thing when it does -- `operation not permitted when
            # stream is capturing`, from whichever kernel followed the sync.
            # Refusing keeps the run correct; the counter is what says to go
            # looking.
            self.refuse(f"capture-failed:{type(error).__name__}")
            return None
        entry = _Entry(key, state, buffers, graph, output, busy=True)
        entry.addresses = entry.current_addresses()
        # A verification replay used to run here, deleted. It separated "this capture is invalid" from "this capture stops being valid once its pages turn over"
        # -- two defects, one symptom: an illegal access whose traceback names whatever ran next.
        # It answered that: the capture was fine, and what broke replays was autocast freeing the weight copies the graph had baked (`TorchAutocast.without_weight_cache`).
        # Question settled, `verify-failed` zero on every run, and it cost a forward per capture (~20 ms here) when captures are already this module's largest fixed cost.
        self._entries.setdefault(key, []).append(entry)
        self.telemetry.captures += 1
        self.telemetry.entries = self.entry_count
        return entry

    def replay(self, entry: _Entry) -> torch.Tensor:
        """Replay, or refuse because an address the capture baked has moved.

        The check is the module's central claim made falsifiable: pages may change freely because they are table CONTENTS; everything else (the pool's storage, each table tensor, the offsets tensor) may not, because the graph holds pointers to them.
        Reading a moved pointer is an illegal access with an asynchronous traceback naming whatever ran next -- the least diagnosable failure this path has.
        """
        if entry.current_addresses() != entry.addresses:
            self.refuse("addresses-moved")
            raise _AddressesMoved(entry.key.label)
        # The invariant `physical_positions`' device-side verdict used to enforce, asked on the host where the answer already is.
        # An `UNMAPPED` entry is -1, which the captured `index_copy_` turns into a negative flat index -- not an error, an illegal access whose traceback names whatever ran next.
        unmapped = [
            layer["kv"]
            for layer in entry.state.values()
            if layer.get("kv") is not None
            and not layer["kv"].rows_are_mapped_whole()
        ]
        if unmapped:
            self.refuse("entry-not-mapped-whole")
            raise _AddressesMoved(
                f"{entry.key.label}: {len(unmapped)} layers hold unmapped rows"
            )
        entry.graph.replay()
        self.telemetry.replays += 1
        return entry.output

    # -- lifetime ---------------------------------------------------------
    def release(self, entry: _Entry) -> None:
        """Give an entry's pages back and mark it usable again.

        NOT `prefill_memory.release_state`, which empties the dict and retires every arena -- those arenas are the addresses a captured graph baked, and retiring them leaves the graph writing into a pool the arena no longer belongs to.
        `hand_back` returns the pages and keeps the arena, exactly the half that has to survive.
        """
        for layer in entry.state.values():
            arena = layer.get("kv")
            if arena is not None:
                arena.hand_back()
        entry.busy = False

    def relet(self, entry: _Entry) -> None:
        """Give the entry's arenas a fresh lease, in place.

        Straight from the pool rather than through `KVArena.take_pages`: that
        method adds what it draws to the arena's held set, and `relet` refuses
        an arena holding anything -- deliberately, because a page a table stops
        naming can never be returned. Taking and mapping are two steps here and
        the arena is empty between them.

        Raises `KVPagesExhausted` exactly as a fresh allocation would -- same supply, same pages asked of it.
        ! still uncaught between `prepare_generation_rows` and the scheduler. Unchanged by this module, still owed.
        """
        for layer in entry.state.values():
            arena = layer.get("kv")
            if arena is None:
                continue
            need = arena.rows * arena.blocks_per_row
            pool = arena.pool
            drawn = None if pool is None else pool.acquire(need)
            if drawn is None:
                free = 0 if pool is None else pool.free_blocks
                record_shortfall(pool.key, need, free)
                raise KVPagesExhausted(
                    f"{need} pages wanted to relet a captured prefill, {free} free"
                )
            arena.relet(drawn)

    def close(self) -> None:
        """Retire every entry, for a model being unloaded."""
        for entries in self._entries.values():
            for entry in entries:
                prefill_memory.release_state(entry.state)
        self._entries.clear()
        self.telemetry.entries = 0

    def entries_for(self, key: PrefillGraphKey) -> Sequence[_Entry]:
        return tuple(self._entries.get(key, ()))

    def __iter__(self) -> Iterator[_Entry]:
        for entries in self._entries.values():
            yield from entries
