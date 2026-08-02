"""General-purpose adaptive RMS downward expander for separated stems."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

__all__ = [
    "GATE_PRESETS",
    "GatePreset",
    "GateResult",
    "adaptive_volume_gate",
    "hard_zero_audio_blocks",
    "get_gate_preset",
]

SILENCE_EPSILON = 1e-7


@dataclass(frozen=True)
class GatePreset:
    """Reusable gate shape; a null offset selects automatic thresholding."""

    name: str
    threshold_offset_db: float | None
    block_ms: float
    attack_ms: float
    release_ms: float
    lookahead_ms: float
    exponent: float
    floor: float
    auto_active_headroom_db: float


GATE_PRESETS: dict[str, GatePreset] = {
    "gentle": GatePreset(
        "gentle", None, 20.0, 8.0, 400.0, 20.0, 2.0, 0.03, 14.0
    ),
    "balanced": GatePreset(
        "balanced", None, 20.0, 8.0, 300.0, 20.0, 3.0, 0.01, 10.0
    ),
    "strong": GatePreset(
        "strong", None, 20.0, 5.0, 180.0, 20.0, 5.0, 0.001, 8.0
    ),
    # Threshold: reference + 6 dB. Look-ahead: preserve transient attacks.
    "extreme": GatePreset(
        "extreme", -6.0, 20.0, 8.0, 120.0, 20.0, 24.0, 0.0, 6.0
    ),
}


@dataclass(frozen=True)
class GateResult:
    input_path: Path
    output_path: Path
    preset: str
    threshold_mode: str
    sample_rate: int
    duration_seconds: float
    reference_dbfs: float | None
    noise_floor_dbfs: float | None
    threshold_dbfs: float
    input_rms_dbfs: float | None
    output_rms_dbfs: float | None
    blocks_reduced_over_6db_percent: float
    blocks_below_threshold_percent: float
    blocks_at_unity_percent: float
    zero_below_dbfs: float | None
    blocks_zeroed_percent: float

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["input_path"] = str(self.input_path)
        result["output_path"] = str(self.output_path)
        return result


def get_gate_preset(preset: str | GatePreset) -> GatePreset:
    if isinstance(preset, GatePreset):
        return preset
    try:
        return GATE_PRESETS[preset]
    except KeyError as error:
        raise ValueError(
            f"unknown gate preset {preset!r}; choose from "
            + ",".join(GATE_PRESETS)
        ) from error


def db(value: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(value, 1e-12))


def _optional_db(value: float) -> float | None:
    return None if value <= 1e-12 else float(db(value))


def _validate_parameters(
    *,
    block_ms: float,
    attack_ms: float,
    release_ms: float,
    lookahead_ms: float,
    exponent: float,
    floor: float,
    auto_active_headroom_db: float,
    threshold_offset_db: float | None,
    threshold_dbfs: float | None,
    zero_below_dbfs: float | None,
) -> None:
    if block_ms <= 0:
        raise ValueError("block_ms must be greater than zero")
    if attack_ms < 0 or release_ms < 0 or lookahead_ms < 0:
        raise ValueError(
            "attack_ms, release_ms, and lookahead_ms cannot be negative"
        )
    if exponent <= 0:
        raise ValueError("exponent must be greater than zero")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be between 0 and 1")
    if auto_active_headroom_db <= 0:
        raise ValueError("auto_active_headroom_db must be greater than zero")
    if threshold_offset_db is not None and threshold_dbfs is not None:
        raise ValueError(
            "threshold_offset_db and threshold_dbfs are mutually exclusive"
        )
    if zero_below_dbfs is not None and zero_below_dbfs > 0:
        raise ValueError("zero_below_dbfs must be 0 dBFS or lower")


def _block_rms(audio: np.ndarray, block_size: int) -> np.ndarray:
    padding = (-len(audio)) % block_size
    padded = np.pad(audio, ((0, padding), (0, 0)))
    blocks = padded.reshape(-1, block_size, audio.shape[1])
    return np.sqrt(
        np.mean(np.square(blocks, dtype=np.float64), axis=(1, 2))
    )


def hard_zero_audio_blocks(
    input_path: Path | str,
    output_path: Path | str,
    *,
    threshold_dbfs: float = -80.0,
    block_ms: float = 20.0,
) -> dict[str, object]:
    """Write exact zero for complete blocks below an absolute RMS threshold."""
    if threshold_dbfs > 0:
        raise ValueError("threshold_dbfs must be 0 dBFS or lower")
    if block_ms <= 0:
        raise ValueError("block_ms must be greater than zero")
    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input audio not found: {source}")
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    if not len(audio):
        raise ValueError(f"input audio is empty: {source}")
    block_size = max(1, round(sample_rate * block_ms / 1000.0))
    rms = _block_rms(audio, block_size)
    zero_blocks = rms < 10.0 ** (threshold_dbfs / 20.0)
    zero_samples = np.repeat(zero_blocks, block_size)[: len(audio)]
    processed = audio.copy()
    processed[zero_samples] = 0.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, processed, sample_rate, subtype="FLOAT")
    return {
        "input_path": source,
        "output_path": destination,
        "threshold_dbfs": float(threshold_dbfs),
        "block_ms": float(block_ms),
        "blocks_zeroed_percent": float(100.0 * np.mean(zero_blocks)),
    }


def _legacy_active_reference(rms: np.ndarray) -> float | None:
    """Keep the previous p10-trimmed median for explicit offset mode."""
    nonzero = rms[rms > SILENCE_EPSILON]
    if not nonzero.size:
        return None
    noise_floor = float(np.percentile(nonzero, 10))
    active = nonzero[nonzero > noise_floor]
    return float(np.median(active if active.size else nonzero))


def _otsu_threshold(values_db: np.ndarray) -> float:
    """Split low/high block energy by maximum between-class variance."""
    low, high = np.percentile(values_db, [1.0, 99.0])
    if high - low < 1e-6:
        return float(low)
    bin_count = int(np.clip(np.sqrt(values_db.size) * 2, 32, 256))
    histogram, edges = np.histogram(
        np.clip(values_db, low, high),
        bins=bin_count,
        range=(low, high),
    )
    probability = histogram.astype(np.float64)
    probability /= max(1.0, float(np.sum(probability)))
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight = np.cumsum(probability)
    mean_sum = np.cumsum(probability * centers)
    total_mean = mean_sum[-1]
    denominator = weight * (1.0 - weight)
    between = np.zeros_like(denominator)
    valid = denominator > 1e-12
    between[valid] = (
        total_mean * weight[valid] - mean_sum[valid]
    ) ** 2 / denominator[valid]
    return float(centers[int(np.argmax(between))])


def _automatic_threshold(
    rms: np.ndarray,
    *,
    active_headroom_db: float,
) -> tuple[float, float | None, float | None]:
    """Return threshold, active reference, and noise floor in dBFS."""
    nonzero = rms[rms > SILENCE_EPSILON]
    if not nonzero.size:
        return -120.0, None, None

    values_db = np.asarray(db(nonzero), dtype=np.float64)
    noise_floor_db = float(np.percentile(values_db, 10))
    dynamic_range_db = float(
        np.percentile(values_db, 95) - noise_floor_db
    )
    if dynamic_range_db < 12.0:
        reference_db = float(np.percentile(values_db, 75))
        return reference_db - active_headroom_db, reference_db, noise_floor_db

    otsu_db = _otsu_threshold(values_db)
    active = values_db[values_db > otsu_db]
    if active.size < max(2, int(round(values_db.size * 0.02))):
        reference_db = float(np.percentile(values_db, 75))
    else:
        reference_db = float(np.median(active))

    # Keep threshold between noise + 3 dB and active reference - headroom.
    upper = reference_db - active_headroom_db
    lower = noise_floor_db + 3.0
    threshold_db = min(max(otsu_db, lower), upper)
    return threshold_db, reference_db, noise_floor_db


def _apply_lookahead(desired: np.ndarray, lookahead_blocks: int) -> np.ndarray:
    if lookahead_blocks <= 0 or not desired.size:
        return desired
    opened = desired.copy()
    for shift in range(1, lookahead_blocks + 1):
        opened[:-shift] = np.maximum(opened[:-shift], desired[shift:])
    return opened


def _smooth_gain(
    desired: np.ndarray,
    *,
    block_ms: float,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    if not desired.size:
        return desired
    attack_blocks = max(1.0, attack_ms / block_ms)
    release_blocks = max(1.0, release_ms / block_ms)
    attack_alpha = np.exp(-1.0 / attack_blocks)
    release_alpha = np.exp(-1.0 / release_blocks)

    envelope = np.empty_like(desired)
    envelope[0] = desired[0]
    for index in range(1, len(desired)):
        opening = desired[index] > envelope[index - 1]
        alpha = attack_alpha if opening else release_alpha
        envelope[index] = (
            alpha * envelope[index - 1]
            + (1.0 - alpha) * desired[index]
        )
    return envelope


def adaptive_volume_gate(
    input_path: Path | str,
    output_path: Path | str,
    *,
    preset: str | GatePreset = "balanced",
    threshold_offset_db: float | None = None,
    threshold_dbfs: float | None = None,
    block_ms: float | None = None,
    attack_ms: float | None = None,
    release_ms: float | None = None,
    lookahead_ms: float | None = None,
    exponent: float | None = None,
    floor: float | None = None,
    auto_active_headroom_db: float | None = None,
    zero_below_dbfs: float | None = -80.0,
) -> GateResult:
    """Apply a linked-stereo downward expander.

    Threshold: absolute -> explicit offset -> preset offset -> RMS clustering.
    Explicit envelope values override the preset.
    """
    selected = get_gate_preset(preset)
    resolved_block_ms = selected.block_ms if block_ms is None else block_ms
    resolved_attack_ms = selected.attack_ms if attack_ms is None else attack_ms
    resolved_release_ms = (
        selected.release_ms if release_ms is None else release_ms
    )
    resolved_lookahead_ms = (
        selected.lookahead_ms if lookahead_ms is None else lookahead_ms
    )
    resolved_exponent = selected.exponent if exponent is None else exponent
    resolved_floor = selected.floor if floor is None else floor
    resolved_headroom = (
        selected.auto_active_headroom_db
        if auto_active_headroom_db is None
        else auto_active_headroom_db
    )
    selected_offset = (
        threshold_offset_db
        if threshold_offset_db is not None
        else selected.threshold_offset_db
    )
    if threshold_dbfs is not None:
        selected_offset = None

    _validate_parameters(
        block_ms=resolved_block_ms,
        attack_ms=resolved_attack_ms,
        release_ms=resolved_release_ms,
        lookahead_ms=resolved_lookahead_ms,
        exponent=resolved_exponent,
        floor=resolved_floor,
        auto_active_headroom_db=resolved_headroom,
        threshold_offset_db=selected_offset,
        threshold_dbfs=threshold_dbfs,
        zero_below_dbfs=zero_below_dbfs,
    )
    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input audio not found: {source}")

    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    if not len(audio):
        raise ValueError(f"input audio is empty: {source}")
    block_size = max(
        1, round(sample_rate * resolved_block_ms / 1000.0)
    )
    rms = _block_rms(audio, block_size)

    if threshold_dbfs is not None:
        threshold_mode = "absolute"
        resolved_threshold_dbfs = float(threshold_dbfs)
        reference = _legacy_active_reference(rms)
        reference_dbfs = _optional_db(reference or 0.0)
        nonzero = rms[rms > SILENCE_EPSILON]
        noise_floor_dbfs = (
            float(np.percentile(db(nonzero), 10))
            if nonzero.size
            else None
        )
    elif selected_offset is not None:
        threshold_mode = "offset"
        reference = _legacy_active_reference(rms)
        reference_dbfs = _optional_db(reference or 0.0)
        if reference is None:
            resolved_threshold_dbfs = -120.0
            noise_floor_dbfs = None
        else:
            resolved_threshold_dbfs = float(
                db(reference) - selected_offset
            )
            nonzero = rms[rms > SILENCE_EPSILON]
            noise_floor_dbfs = float(np.percentile(db(nonzero), 10))
    else:
        threshold_mode = "auto"
        (
            resolved_threshold_dbfs,
            reference_dbfs,
            noise_floor_dbfs,
        ) = _automatic_threshold(
            rms,
            active_headroom_db=resolved_headroom,
        )

    threshold = 10.0 ** (resolved_threshold_dbfs / 20.0)
    desired = np.ones_like(rms)
    below = rms < threshold
    desired[below] = np.maximum(
        (rms[below] / threshold) ** resolved_exponent,
        resolved_floor,
    )
    desired = _apply_lookahead(
        desired,
        int(np.ceil(resolved_lookahead_ms / resolved_block_ms)),
    )
    envelope = _smooth_gain(
        desired,
        block_ms=resolved_block_ms,
        attack_ms=resolved_attack_ms,
        release_ms=resolved_release_ms,
    )

    sample_positions = np.arange(len(audio), dtype=np.float64) / block_size
    gain = np.interp(
        sample_positions,
        np.arange(len(envelope)),
        envelope,
    )
    processed = audio * gain[:, None]
    if zero_below_dbfs is not None:
        processed_rms = _block_rms(processed, block_size)
        zero_blocks = processed_rms < 10.0 ** (zero_below_dbfs / 20.0)
        zero_samples = np.repeat(zero_blocks, block_size)[: len(processed)]
        processed[zero_samples] = 0.0
    else:
        zero_blocks = np.zeros_like(rms, dtype=bool)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, processed, sample_rate, subtype="FLOAT")

    input_rms = float(
        np.sqrt(np.mean(np.square(audio, dtype=np.float64)))
    )
    output_rms = float(
        np.sqrt(np.mean(np.square(processed, dtype=np.float64)))
    )
    return GateResult(
        input_path=source,
        output_path=destination,
        preset=selected.name,
        threshold_mode=threshold_mode,
        sample_rate=sample_rate,
        duration_seconds=len(audio) / sample_rate,
        reference_dbfs=reference_dbfs,
        noise_floor_dbfs=noise_floor_dbfs,
        threshold_dbfs=resolved_threshold_dbfs,
        input_rms_dbfs=_optional_db(input_rms),
        output_rms_dbfs=_optional_db(output_rms),
        blocks_reduced_over_6db_percent=float(
            100.0 * np.mean(envelope < 0.5)
        ),
        blocks_below_threshold_percent=float(100.0 * np.mean(below)),
        blocks_at_unity_percent=float(
            100.0 * np.mean(envelope >= 0.999)
        ),
        zero_below_dbfs=zero_below_dbfs,
        blocks_zeroed_percent=float(100.0 * np.mean(zero_blocks)),
    )
