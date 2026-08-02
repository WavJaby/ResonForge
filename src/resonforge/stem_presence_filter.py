"""Conservative detection of globally absent separated stems."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class StemPresenceMetrics:
    path: Path
    duration_seconds: float
    global_rms_dbfs: float
    block_p99_dbfs: float
    coverage_above_minus_40_percent: float
    loudest_1s_dbfs: float
    loudest_200ms_dbfs: float

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["path"] = str(self.path)
        return result


@dataclass(frozen=True)
class StemPresenceDecision:
    present: bool
    relative_deficit_db: float
    metrics: StemPresenceMetrics
    failed_presence_checks: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["metrics"]["path"] = str(self.metrics.path)
        return result


def _dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-12)))


def analyze_stem_presence(
    audio_path: str | Path,
    *,
    block_ms: float = 100.0,
) -> StemPresenceMetrics:
    path = Path(audio_path).resolve()
    with sf.SoundFile(path) as audio_file:
        sample_rate = audio_file.samplerate
        frame_count = len(audio_file)
        block_frames = max(1, round(sample_rate * block_ms / 1000.0))
        sum_squares = 0.0
        sample_count = 0
        block_powers: list[float] = []
        for block in audio_file.blocks(
            blocksize=block_frames,
            dtype="float32",
            always_2d=True,
        ):
            power = float(np.mean(np.square(block, dtype=np.float64)))
            block_powers.append(power)
            sum_squares += float(
                np.sum(np.square(block, dtype=np.float64))
            )
            sample_count += block.size
    if not sample_count:
        raise ValueError(f"empty audio: {path}")

    powers = np.asarray(block_powers, dtype=np.float64)
    block_dbfs = 10.0 * np.log10(np.maximum(powers, 1e-24))

    def loudest_window(window_ms: float) -> float:
        count = max(1, round(window_ms / block_ms))
        if powers.size <= count:
            return _dbfs(float(np.sqrt(np.mean(powers))))
        cumulative = np.concatenate(([0.0], np.cumsum(powers)))
        means = (cumulative[count:] - cumulative[:-count]) / count
        return _dbfs(float(np.sqrt(np.max(means))))

    return StemPresenceMetrics(
        path=path,
        duration_seconds=frame_count / sample_rate,
        global_rms_dbfs=_dbfs(
            float(np.sqrt(sum_squares / sample_count))
        ),
        block_p99_dbfs=float(np.percentile(block_dbfs, 99)),
        coverage_above_minus_40_percent=float(
            100.0 * np.mean(block_dbfs > -40.0)
        ),
        loudest_1s_dbfs=loudest_window(1000.0),
        loudest_200ms_dbfs=loudest_window(200.0),
    )


def assess_stem_presence(
    candidate: StemPresenceMetrics,
    reference_metrics: list[StemPresenceMetrics],
) -> StemPresenceDecision:
    """Return absent only when every deliberately narrow condition is met."""
    reference_levels = [
        metrics.global_rms_dbfs
        for metrics in reference_metrics
        if metrics.path != candidate.path
        and metrics.global_rms_dbfs > -50.0
    ]
    reference_level = (
        float(np.median(reference_levels))
        if reference_levels
        else candidate.global_rms_dbfs
    )
    relative_deficit = reference_level - candidate.global_rms_dbfs
    checks = {
        "global_rms": candidate.global_rms_dbfs < -50.0,
        "p99_block": candidate.block_p99_dbfs < -38.0,
        "active_coverage": (
            candidate.coverage_above_minus_40_percent < 1.0
        ),
        "loudest_1s": candidate.loudest_1s_dbfs < -32.0,
        # Rescue sub-second but clearly audible notes/fills.
        "loudest_200ms": candidate.loudest_200ms_dbfs < -28.0,
        "relative_deficit": relative_deficit > 25.0,
    }
    absent = all(checks.values())
    return StemPresenceDecision(
        present=not absent,
        relative_deficit_db=round(relative_deficit, 3),
        metrics=candidate,
        failed_presence_checks=tuple(
            name for name, matched_absence in checks.items()
            if not matched_absence
        ),
    )
