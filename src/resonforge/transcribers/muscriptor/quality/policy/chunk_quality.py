"""Generic completed-chunk metrics and accepted-history references."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from statistics import median
from threading import Lock

from muscriptor.tokenizer.notes import DRUM_PROGRAM, NoteEvent

from resonforge.transcribers.muscriptor.quality.contract.generation_position import (
    GenerationPosition,
)
from resonforge.transcribers.muscriptor.quality.contract.model_protocol import (
    TokenizerProtocol,
)


@dataclass(frozen=True)
class AdaptiveChunkQualityConfig:
    """Scale-free robust outlier policy for completed chunks."""

    history_size: int = 64
    min_history: int = 6
    robust_z_threshold: float = 4.0
    min_outlier_features: int = 2
    minimum_scale_fraction: float = 0.25
    min_event_count_ratio: float = 0.5
    anchor_features: tuple[str, ...] = (
        "tokens_per_event",
        "retrigger_rate",
        "pitch_span",
        "pitch_median",
    )

    def __post_init__(self) -> None:
        if self.history_size < 1:
            raise ValueError("history_size must be positive")
        if not 1 <= self.min_history <= self.history_size:
            raise ValueError("min_history must be within history_size")
        if self.robust_z_threshold <= 0:
            raise ValueError("robust_z_threshold must be positive")
        if self.min_outlier_features < 1:
            raise ValueError("min_outlier_features must be positive")
        if self.minimum_scale_fraction <= 0:
            raise ValueError("minimum_scale_fraction must be positive")
        if not 0 < self.min_event_count_ratio <= 1:
            raise ValueError("min_event_count_ratio must be within (0, 1]")
        if not self.anchor_features:
            raise ValueError("anchor_features must not be empty")


DEFAULT_ADAPTIVE_CHUNK_QUALITY_CONFIG = AdaptiveChunkQualityConfig()


@dataclass(frozen=True)
class ChunkQualityMetrics:
    duration_seconds: float
    generated_tokens: int
    canonical_tokens: int
    token_inflation_ratio: float
    generated_token_rate: float
    tokens_per_event: float
    note_on_count: int
    note_off_count: int
    drum_hit_count: int
    event_rate: float
    peak_polyphony: int
    pitch_min: int | None
    pitch_median: float | None
    pitch_max: int | None
    open_notes_start: int
    open_notes_end: int
    same_pitch_retriggers: int
    retrigger_rate: float
    max_sounded_duration_seconds: float
    adjacent_same_pitch_restarts: int = 0
    max_adjacent_same_pitch_restart_chain: int = 0
    max_synchronized_restarts: int = 0

    @property
    def event_count(self) -> int:
        return self.note_on_count + self.drum_hit_count

    @property
    def pitch_span(self) -> float | None:
        if self.pitch_min is None or self.pitch_max is None:
            return None
        return float(self.pitch_max - self.pitch_min)


@dataclass(frozen=True)
class RestartChainCollapseEvidence:
    """One coherent synchronized restart episode and its safe boundary."""

    observed_at: GenerationPosition
    suspected_start: GenerationPosition
    last_safe_frontier: GenerationPosition | None
    adjacent_restarts: int
    maximum_chain: int
    synchronized_restarts: int


@dataclass(frozen=True)
class _PositionedNoteEvent:
    event: NoteEvent
    position: GenerationPosition


def locate_restart_chain_collapse(
    tokenizer: TokenizerProtocol,
    tokens: list[int],
    source_seek: float,
    ownership_start: float,
    ownership_end: float,
    *,
    minimum_adjacent_restarts: int = 8,
    minimum_chain: int = 6,
    minimum_synchronized_restarts: int = 2,
    prompt_tokens: int = 0,
) -> RestartChainCollapseEvidence | None:
    """Locate one coherent restart episode without combining unrelated maxima."""
    if min(
        minimum_adjacent_restarts,
        minimum_chain,
        minimum_synchronized_restarts,
    ) <= 0:
        raise ValueError("restart collapse thresholds must be positive")
    if not 0 <= prompt_tokens <= len(tokens):
        raise ValueError("prompt_tokens must be within the token sequence")
    positioned, frontiers, active_at_start = _positioned_note_events(
        tokenizer,
        tokens,
        source_seek,
        ownership_start,
        ownership_end,
        prompt_tokens,
    )
    grouped: dict[float, list[_PositionedNoteEvent]] = defaultdict(list)
    for item in positioned:
        grouped[item.event.time].append(item)

    active = set(active_at_start)
    chain_lengths: dict[tuple[int, int], int] = {
        key: 1 for key in active_at_start
    }
    chain_starts: dict[tuple[int, int], GenerationPosition] = {
        key: GenerationPosition(0, None) for key in active_at_start
    }
    for event_time in sorted(grouped):
        items = grouped[event_time]
        ended_keys = {
            (item.event.program, item.event.pitch)
            for item in items
            if not item.event.is_drum
            and item.event.velocity <= 0
            and (item.event.program, item.event.pitch) in active
        }
        active.difference_update(ended_keys)
        restarted: list[tuple[tuple[int, int], _PositionedNoteEvent]] = []
        for item in items:
            event = item.event
            if event.is_drum or event.velocity <= 0:
                continue
            key = (event.program, event.pitch)
            if key in ended_keys:
                chain_lengths[key] = chain_lengths.get(key, 1) + 1
                restarted.append((key, item))
            else:
                chain_lengths[key] = 1
                chain_starts[key] = item.position
            active.add(key)
        synchronized_restarts = len(restarted)
        adjacent_restarts = sum(
            chain_lengths[key] - 1 for key, _ in restarted
        )
        maximum_chain = max(
            (chain_lengths[key] for key, _ in restarted),
            default=0,
        )
        if (
            adjacent_restarts < minimum_adjacent_restarts
            or maximum_chain < minimum_chain
            or synchronized_restarts < minimum_synchronized_restarts
        ):
            continue
        suspected_start = min(
            (chain_starts[key] for key, _ in restarted),
            key=lambda position: position.token_index,
        )
        observed_at = max(
            (item.position for _, item in restarted),
            key=lambda position: position.token_index,
        )
        last_safe_frontier = next(
            (
                frontier
                for frontier in reversed(frontiers)
                if frontier.token_index < suspected_start.token_index
            ),
            None,
        )
        return RestartChainCollapseEvidence(
            observed_at,
            suspected_start,
            last_safe_frontier,
            adjacent_restarts,
            maximum_chain,
            synchronized_restarts,
        )
    return None


def _positioned_note_events(
    tokenizer: TokenizerProtocol,
    tokens: list[int],
    source_seek: float,
    ownership_start: float,
    ownership_end: float,
    prompt_tokens: int,
) -> tuple[
    list[_PositionedNoteEvent],
    list[GenerationPosition],
    set[tuple[int, int]],
]:
    from resonforge.transcribers.muscriptor.quality.policy import overlap_runtime

    frame_rate = tokenizer.frame_rate
    start_tick = round(source_seek * frame_rate)
    tick = start_tick
    shift_value: int | None = None
    program: int | None = None
    velocity: int | None = None
    in_prologue = True
    active_at_start = {
        key
        for key in overlap_runtime.open_keys_at(
            tokenizer,
            tokens,
            source_seek,
            ownership_start,
        )
        if key[0] != DRUM_PROGRAM
    }
    positioned: list[_PositionedNoteEvent] = []
    frontiers: list[GenerationPosition] = []
    for token_index, token in enumerate(tokens):
        event = tokenizer._vocab[token]
        if in_prologue:
            if event.type == "tie":
                in_prologue = False
            elif event.type == "program":
                program = event.value
            elif event.type == "shift":
                return [], [], set()
            continue
        if event.type == "shift" and event.value > 0:
            tick = start_tick + event.value
            shift_value = event.value
            if token_index >= prompt_tokens:
                frontiers.append(
                    GenerationPosition(token_index - prompt_tokens, shift_value)
                )
        elif event.type == "program":
            program = event.value
        elif event.type == "velocity":
            velocity = event.value
        elif event.type == "drum":
            time = tick / frame_rate
            if ownership_start <= time < ownership_end:
                positioned.append(
                    _PositionedNoteEvent(
                        NoteEvent(True, DRUM_PROGRAM, time, 1, event.value),
                        GenerationPosition(
                            max(0, token_index - prompt_tokens),
                            shift_value,
                        ),
                    )
                )
        elif event.type == "pitch" and program is not None and velocity is not None:
            time = tick / frame_rate
            if ownership_start <= time < ownership_end:
                positioned.append(
                    _PositionedNoteEvent(
                        NoteEvent(False, program, time, velocity, event.value),
                        GenerationPosition(
                            max(0, token_index - prompt_tokens),
                            shift_value,
                        ),
                    )
                )
    return positioned, frontiers, active_at_start


@dataclass(frozen=True)
class ChunkQualityReference:
    """Immutable accepted-chunk history supplied to a pure detector."""

    samples: tuple[ChunkQualityMetrics, ...] = ()


@dataclass(frozen=True)
class AdaptiveFeatureScore:
    value: float
    history_median: float
    history_scale: float
    robust_z: float


@dataclass(frozen=True)
class AdaptiveChunkQualityAssessment:
    status: str
    reference_samples: int
    active_samples: int
    reliable_samples: int
    minimum_event_count: float
    scores: dict[str, AdaptiveFeatureScore]
    outliers: tuple[str, ...]


class ChunkQualityHistory:
    """Own accepted metrics; candidates can inspect but never mutate it."""

    def __init__(self, history_size: int) -> None:
        if history_size < 1:
            raise ValueError("history_size must be positive")
        self._samples: deque[ChunkQualityMetrics] = deque(maxlen=history_size)
        self._lock = Lock()

    def reference(self) -> ChunkQualityReference:
        with self._lock:
            return ChunkQualityReference(tuple(self._samples))

    def accept(self, metrics: ChunkQualityMetrics) -> None:
        with self._lock:
            self._samples.append(metrics)


def measure_chunk_quality(
    tokenizer: TokenizerProtocol,
    tokens: list[int],
    source_seek: float,
    ownership_start: float,
    ownership_end: float,
    *,
    generated_tokens: int,
) -> ChunkQualityMetrics:
    """Measure scale-normalized output owned by one completed chunk."""
    from resonforge.transcribers.muscriptor.quality.policy import overlap_runtime

    duration = max(0.01, ownership_end - ownership_start)
    events = overlap_runtime.note_events_in_range(
        tokenizer,
        tokens,
        source_seek,
        ownership_start,
        ownership_end,
    )
    open_start = {
        key
        for key in overlap_runtime.open_keys_at(
            tokenizer,
            tokens,
            source_seek,
            ownership_start,
        )
        if key[0] != 128
    }
    active = {key: ownership_start for key in open_start}
    grouped = defaultdict(list)
    note_on_count = 0
    note_off_count = 0
    drum_hit_count = 0
    retriggers = 0
    adjacent_restarts = 0
    restart_chain_lengths: dict[tuple[int, int], int] = {}
    maximum_restart_chain = 0
    maximum_synchronized_restarts = 0
    maximum_duration = 0.0
    pitches: list[int] = []

    for event in events:
        grouped[event.time].append(event)
        if event.is_drum:
            drum_hit_count += 1
        elif event.velocity > 0:
            note_on_count += 1
            pitches.append(event.pitch)
        else:
            note_off_count += 1

    peak_polyphony = len(active)
    for event_time in sorted(grouped):
        timestamp_events = grouped[event_time]
        ended_keys: set[tuple[int, int]] = set()
        for event in timestamp_events:
            if event.is_drum or event.velocity > 0:
                continue
            key = (event.program, event.pitch)
            started_at = active.pop(key, None)
            if started_at is not None:
                ended_keys.add(key)
                maximum_duration = max(maximum_duration, event_time - started_at)
        synchronized_restarts = 0
        for event in timestamp_events:
            if event.is_drum or event.velocity <= 0:
                continue
            key = (event.program, event.pitch)
            started_at = active.get(key)
            if started_at is not None:
                retriggers += 1
                maximum_duration = max(maximum_duration, event_time - started_at)
            if key in ended_keys:
                adjacent_restarts += 1
                synchronized_restarts += 1
                restart_chain_lengths[key] = restart_chain_lengths.get(key, 1) + 1
            else:
                restart_chain_lengths[key] = 1
            maximum_restart_chain = max(
                maximum_restart_chain,
                restart_chain_lengths[key],
            )
            active[key] = event_time
        maximum_synchronized_restarts = max(
            maximum_synchronized_restarts,
            synchronized_restarts,
        )
        peak_polyphony = max(peak_polyphony, len(active))

    for started_at in active.values():
        maximum_duration = max(maximum_duration, ownership_end - started_at)
    event_count = note_on_count + drum_hit_count
    canonical_tokens = len(
        overlap_runtime.canonicalize_tokens(
            tokenizer,
            tokens,
            source_seek,
            ownership_start,
            ownership_end,
        )
    )
    return ChunkQualityMetrics(
        duration_seconds=duration,
        generated_tokens=generated_tokens,
        canonical_tokens=canonical_tokens,
        token_inflation_ratio=(
            generated_tokens / canonical_tokens
            if canonical_tokens
            else float(generated_tokens)
        ),
        generated_token_rate=generated_tokens / duration,
        tokens_per_event=(generated_tokens / event_count if event_count else 0.0),
        note_on_count=note_on_count,
        note_off_count=note_off_count,
        drum_hit_count=drum_hit_count,
        event_rate=event_count / duration,
        peak_polyphony=peak_polyphony,
        pitch_min=min(pitches) if pitches else None,
        pitch_median=float(median(pitches)) if pitches else None,
        pitch_max=max(pitches) if pitches else None,
        open_notes_start=len(open_start),
        open_notes_end=len(active),
        same_pitch_retriggers=retriggers,
        retrigger_rate=retriggers / duration,
        max_sounded_duration_seconds=maximum_duration,
        adjacent_same_pitch_restarts=adjacent_restarts,
        max_adjacent_same_pitch_restart_chain=maximum_restart_chain,
        max_synchronized_restarts=maximum_synchronized_restarts,
    )


def adaptive_feature_outliers(
    metrics: ChunkQualityMetrics,
    reference: ChunkQualityReference,
    config: AdaptiveChunkQualityConfig = DEFAULT_ADAPTIVE_CHUNK_QUALITY_CONFIG,
) -> dict[str, tuple[float, float, float, float]]:
    """Return high-side robust outliers as value/median/scale/z-score."""
    assessment = assess_adaptive_chunk_quality(metrics, reference, config)
    return {
        name: (
            assessment.scores[name].value,
            assessment.scores[name].history_median,
            assessment.scores[name].history_scale,
            assessment.scores[name].robust_z,
        )
        for name in assessment.outliers
    }


def assess_adaptive_chunk_quality(
    metrics: ChunkQualityMetrics,
    reference: ChunkQualityReference,
    config: AdaptiveChunkQualityConfig = DEFAULT_ADAPTIVE_CHUNK_QUALITY_CONFIG,
) -> AdaptiveChunkQualityAssessment:
    """Explain baseline eligibility and every comparable feature score."""
    active_history = tuple(
        sample for sample in reference.samples if sample.event_count > 0
    )
    if metrics.event_count == 0 or len(active_history) < config.min_history:
        return AdaptiveChunkQualityAssessment(
            "insufficient_active_history",
            len(reference.samples),
            len(active_history),
            0,
            0.0,
            {},
            (),
        )
    median_event_count = float(
        median(sample.event_count for sample in active_history)
    )
    minimum_event_count = median_event_count * config.min_event_count_ratio
    reliable_history = tuple(
        sample
        for sample in active_history
        if sample.event_count >= minimum_event_count
    )
    if (
        metrics.event_count < minimum_event_count
        or len(reliable_history) < config.min_history
    ):
        status = (
            "insufficient_event_support"
            if metrics.event_count < minimum_event_count
            else "insufficient_reliable_history"
        )
        return AdaptiveChunkQualityAssessment(
            status,
            len(reference.samples),
            len(active_history),
            len(reliable_history),
            minimum_event_count,
            {},
            (),
        )
    features = {
        "generated_token_rate": metrics.generated_token_rate,
        "tokens_per_event": metrics.tokens_per_event,
        "event_rate": metrics.event_rate,
        "peak_polyphony": float(metrics.peak_polyphony),
        "retrigger_rate": metrics.retrigger_rate,
    }
    scores: dict[str, AdaptiveFeatureScore] = {}
    outlier_names: list[str] = []
    for name, value in features.items():
        history_values = [
            float(getattr(sample, name)) for sample in reliable_history
        ]
        center = float(median(history_values))
        mad = float(median(abs(item - center) for item in history_values))
        if center == 0.0 and mad == 0.0:
            continue
        scale = max(
            1.4826 * mad,
            abs(center) * config.minimum_scale_fraction,
        )
        robust_z = (value - center) / scale
        scores[name] = AdaptiveFeatureScore(value, center, scale, robust_z)
        if robust_z >= config.robust_z_threshold:
            outlier_names.append(name)
    pitch_features = {
        "pitch_span": metrics.pitch_span,
        "pitch_median": metrics.pitch_median,
    }
    for name, value in pitch_features.items():
        if value is None:
            continue
        history_values = [
            float(sample_value)
            for sample in reliable_history
            if (sample_value := getattr(sample, name)) is not None
        ]
        if len(history_values) < config.min_history:
            continue
        center = float(median(history_values))
        mad = float(median(abs(item - center) for item in history_values))
        if name == "pitch_median":
            history_spans = [
                float(span)
                for sample in reliable_history
                if (span := sample.pitch_span) is not None
            ]
            scale_floor = float(median(history_spans)) * config.minimum_scale_fraction
            delta = abs(float(value) - center)
        else:
            scale_floor = abs(center) * config.minimum_scale_fraction
            delta = float(value) - center
        scale = max(1.4826 * mad, scale_floor, 1.0)
        robust_z = delta / scale
        scores[name] = AdaptiveFeatureScore(
            float(value), center, scale, robust_z
        )
        if robust_z >= config.robust_z_threshold:
            outlier_names.append(name)
    if len(outlier_names) < config.min_outlier_features or not any(
        name in outlier_names for name in config.anchor_features
    ):
        outlier_names.clear()
    return AdaptiveChunkQualityAssessment(
        "evaluated",
        len(reference.samples),
        len(active_history),
        len(reliable_history),
        minimum_event_count,
        scores,
        tuple(outlier_names),
    )
