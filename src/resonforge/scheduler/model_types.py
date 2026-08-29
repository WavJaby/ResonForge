"""Shared model-worker identity, work, and telemetry contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from resonforge.scheduler.device.block_pool import ArenaCost

ModelScope = Literal["shared", "stem"]
GenerationJobType = Literal["primary", "bootstrap", "recovery", "control"]


@runtime_checkable
class ExecutionProfile(Protocol):
    """The execution identity a backend declares to the scheduler.

    Deliberately the *minimum* the scheduler acts on, not the backend's whole
    profile: the two names that distinguish one execution path from another,
    and the one flag a liveness check reads. The scheduler never names a
    concrete implementation, which is what keeps `scheduler` from importing
    `transcribers`.
    """

    runtime: str
    resolved_backend: str
    cuda_graphs: bool


@dataclass(frozen=True)
class ModelIdentity:
    """Immutable weight ownership within one execution lane."""

    scope: ModelScope
    device: str
    model: str
    dtype: str | None
    stem: str | None = None


@dataclass(frozen=True)
class ModelKey:
    """One model identity paired with one resolved execution profile."""

    identity: ModelIdentity
    execution_profile: ExecutionProfile

    def __post_init__(self) -> None:
        if self.identity.scope == "shared" and self.identity.stem is not None:
            raise ValueError("shared model keys cannot name a stem")
        if self.identity.scope == "stem" and not self.identity.stem:
            raise ValueError("stem model keys require a stem name")
        if not self.identity.device or not self.identity.model:
            raise ValueError("model device and name must be non-empty")

    @property
    def scope(self) -> ModelScope:
        return self.identity.scope

    @property
    def device(self) -> str:
        return self.identity.device

    @property
    def model(self) -> str:
        return self.identity.model

    @property
    def dtype(self) -> str | None:
        return self.identity.dtype

    @property
    def stem(self) -> str | None:
        return self.identity.stem


@dataclass(frozen=True)
class PreparedWorkItem:
    """Session-ready item plus its role-specific result adapter."""

    session_item: object
    complete: Callable[[object], object] | None = None


# What a backend answers when the pool asks a loaded model what one of its
# logical rows costs at a given KV length. Measured by the backend, never
# stored: the pool asks again on a new device rather than reading a profile.
# `(model, kv_capacity, lane)`. The lane name is here because pricing a row
# and declaring that lane's KV floor are the same per-lane moment, and the
# floor has to be on record *before* any lane sizes itself against the
# device -- see `paged_kv.declared_lane_count` and `scheduler.device.ledger`.
ArenaCostProbe = Callable[[object, int, str], ArenaCost]

# Declare this lane's KV page supply, given the arena it is about to open:
# `(model, width, kv_capacity) -> bytes declared`. The same seam as
# `ArenaCostProbe` and for the same reason -- the scheduler is generic over the
# model it drives, and a page's geometry (heads, head dim, dtype, layer count)
# is the model's business. The scheduler contributes the one thing the model
# cannot know: the width it just resolved against this device.
#
# Width, not a byte budget, because the rule is "no greedier than this arena
# would have been" and only the model can turn a width into pages exactly the
# way `init_state` will. Returning bytes keeps the answer comparable across
# geometries, where a page count is not a quantity at all.
#: `(model, width, capacity, lane, role) -> declared bytes`. The role is
#: here because the device is divided by role, not by model name -- see
#: `scheduler.device.ledger`. Without it the supply is sized by whichever
#: lane declared first, which on BAIR gave the idle recovery lane 81 rows
#: and the primary lane 28.
#: `(model, width, kv_capacity, lane, role, device) -> declared bytes`.
#: **`device` is the scheduler's own key, not one re-derived from a tensor.**
#: `str(param.device)` yields `cuda:0` where the scheduler says `cuda`, and the
#: block-pool registry is keyed by that string -- so deriving it a second time
#: opened a *second pool for the same card*, priced against no measurement at
#: all. One producer of device identity, passed down.
KVPoolDeclaration = Callable[[object, int, int, str, str, str], int]


@dataclass(frozen=True)
class DependencyClaimMember:
    """One downstream row a parent handle cannot release without.

    Carries identity only. What the member *costs* is the pool's declared lane
    cost for its model, measured once when the device's lanes were declared —
    so a claim cannot state a second, disagreeing price for the same lane.
    """

    key: ModelKey


@dataclass(frozen=True)
class ArenaLaneDeclaration:
    """One model a pipeline session will hold resident on one device."""

    key: ModelKey
    # The KV length this lane's row cost is priced at. A lane that later opens
    # a longer session re-prices itself; this is what its *floor* costs.
    kv_capacity: int = 2048
    # Rows this lane needs resident at once before it can do its job. A
    # recovery lane needs one, and that one is what the pool refuses to let
    # another lane consume -- the reservation that used to live in
    # `spill_claims_per_row`.
    minimum_width: int = 1
    # Rows this lane can ever put to work. Backend policy: a bootstrap lane
    # that runs one recovery group deep cannot use a wide arena however much
    # memory is free, and letting it take one starves the lane that can.
    maximum_width: int | None = None
    arena_cost: ArenaCostProbe | None = None

    def __post_init__(self) -> None:
        if self.kv_capacity < 1:
            raise ValueError("arena lane kv capacity must be positive")
        if self.minimum_width < 1:
            raise ValueError("arena lane minimum width must be positive")
        if self.maximum_width is not None and self.maximum_width < self.minimum_width:
            raise ValueError("arena lane maximum width cannot be below its minimum")


@dataclass(frozen=True)
class ArenaDeclaration:
    """Every lane one pipeline session will hold resident on one device.

    This is all that survives `InitialAllocationDeclaration`. It declares
    residency; it does not solve a width, rank a throughput curve, or consult a
    stored profile. Its only job is to let the pool know, before the first
    arena opens, which lanes still need a floor.
    """

    device: str
    lanes: tuple[ArenaLaneDeclaration, ...]

    def __post_init__(self) -> None:
        if not self.device or not self.lanes:
            raise ValueError("arena declaration cannot be empty")
        if any(lane.key.device != self.device for lane in self.lanes):
            raise ValueError("arena declaration crossed devices")
        identities = [lane.key.identity for lane in self.lanes]
        if len(identities) != len(set(identities)):
            raise ValueError("arena declaration lanes must be unique")


@dataclass(frozen=True)
class TransitiveDependencyClaim:
    """Maximum scheduler work required before one parent handle can release."""

    members: tuple[DependencyClaimMember, ...]

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise ValueError("transitive dependency claim requires multiple members")
        devices = {member.key.device for member in self.members}
        if len(devices) != 1:
            raise ValueError("transitive dependency claim crossed devices")


@dataclass(frozen=True)
class DependencyClaimToken:
    """Opaque ownership identity for one admitted downstream claim."""

    value: int


@dataclass(frozen=True)
class ClaimedBatchResult:
    """Generation result paired with its still-live downstream claim."""

    value: object
    claim_token: DependencyClaimToken


@dataclass(frozen=True)
class BatchSubmission:
    """One typed persistent row submission, including its resource contract."""

    key: ModelKey
    compatibility_key: object
    item: object
    max_batch_size: int
    condition_compatibility_key: object | None = None
    # How this lane prices one logical row, asked of the loaded model. `None`
    # means the lane is unpriced and the pool never bounds it -- the CPU and
    # test paths, where there is no device to run out of.
    arena_cost: ArenaCostProbe | None = None
    # Declares the KV supply this lane's arenas draw from. `None` leaves every
    # arena private, which is exactly today's behaviour.
    kv_pool: KVPoolDeclaration | None = None
    minimum_width: int = 1
    maximum_width: int | None = None
    priority: int = 10
    job_type: GenerationJobType = "primary"
    batch_dimension: int | None = None
    continuous_factory: Callable[..., object] | None = None
    prepare: Callable[[object, tuple[object, ...]], tuple[PreparedWorkItem, ...]] | None = None
    continuous_replacement: bool = True
    ready_order: str | None = None
    dependency_claim: TransitiveDependencyClaim | None = None


@dataclass(frozen=True)
class ModelTaskTiming:
    """Completed task timing captured without changing Future results."""

    sequence: int
    key: ModelKey
    priority: int
    job_type: GenerationJobType
    queue_wait_seconds: float
    run_seconds: float
    succeeded: bool
    model_loaded: bool
    pipeline_session_id: str | None = None
    batch_size: int = 1
    generation_steps: int = 0
    wasted_token_rows: int = 0
    hot_replacements: int = 0
    resident_checkpoints: int = 0
    resident_resumes: int = 0
    resident_discards: int = 0
    discarded_token_budget: int = 0
    bucket_borrows: int = 0
    ready_order: str = ""
    estimated_prefill_tokens: int = 0
    estimated_decode_tokens: int = 0
    replacement_attempts: int = 0
    replacement_misses: int = 0
    replacement_miss_reasons: dict[str, int] = field(default_factory=dict)
    active_steps_by_width: dict[int, int] = field(default_factory=dict)
    prefill_batches_by_width: dict[int, int] = field(default_factory=dict)
    condition_batches_by_width: dict[int, int] = field(default_factory=dict)
    first_token_batches_by_width: dict[int, int] = field(default_factory=dict)
    packed_prefill_batches_by_width: dict[int, int] = field(default_factory=dict)
    quantum_steps_by_size: dict[int, int] = field(default_factory=dict)
    scheduler_decision_cpu_us: float = 0.0
    scheduler_decision_phase_us: dict[str, float] = field(default_factory=dict)
    scheduler_action_wall_us: dict[str, float] = field(default_factory=dict)
    scheduler_boundary_gap_us: float = 0.0
    scheduler_boundary_gap_count: int = 0
    scheduler_watchdog_timeout_seconds: float = 0.0
    producer_decision_waits_outstanding: int = 0
    producer_decision_waits_unbound: int = 0
    phase_gpu_ms: dict[str, float] = field(default_factory=dict)
    cuda_graph_captures: int = 0
    cuda_graph_replays: int = 0
    cuda_graph_refusals: int = 0
    cuda_graph_evictions: int = 0
    cuda_graph_variants: dict[str, dict[str, int]] = field(default_factory=dict)
    cuda_graph_cache_entries_peak: int = 0
    cuda_graph_cache_static_bytes_peak: int = 0
    cuda_graph_cache_pool_bytes_peak: int = 0
    #: Captured-prefill counters. **Gauges, not counters**: the cache is per
    #: loaded model and outlives every session, so these are the model's
    #: running totals rather than a per-job delta -- summing them across jobs
    #: would report one cache many times.
    prefill_graph: dict[str, int] = field(default_factory=dict)
    prefill_graph_refusals: dict[str, int] = field(default_factory=dict)
    scheduler_observations: dict[str, int] = field(default_factory=dict)
    # Instantaneous device-wide readings. Separate from observations because
    # those are counters and are summed across jobs; summing a gauge reports a
    # device several times over.
    scheduler_gauges: dict[str, int] = field(default_factory=dict)
    scheduler_state_histograms: dict[str, dict[int, int]] = field(
        default_factory=dict
    )
    candidate_name: str = ""
    trace_context: tuple[object, ...] = ()
    prompt_token_count: int = 0
    max_generation_tokens: int = 0
    result_token_count: int = 0
    generated_token_count: int = 0
    termination_reason: str = ""
    verify_boundary_reached: bool = False
    verify_prefix_token_count: int = 0
    guard_action: str = ""
    guard_warning_findings: int = 0
    guard_critical_findings: int = 0
    guard_reasons: tuple[str, ...] = ()
