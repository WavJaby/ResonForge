"""Frame-exact active-audio regions for silence-aware transcription."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

from .buffer import AudioBuffer

ActivityMode = Literal["rms", "exact-zero"]


@dataclass(frozen=True)
class AudioRegion:
    """Half-open source-frame interval used for extraction and timestamp rebasing."""

    start_frame: int
    end_frame: int
    sample_rate: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if self.end_frame <= self.start_frame:
            raise ValueError("end_frame must be greater than start_frame")

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def start_seconds(self) -> float:
        return self.start_frame / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_frame / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate

    @property
    def source_offset_frames(self) -> int:
        """Source frame corresponding to local frame zero of an extracted region."""
        return self.start_frame

    @property
    def source_offset_seconds(self) -> float:
        return self.start_seconds

    def to_source_frame(self, local_frame: int) -> int:
        """Map a local frame boundary onto the original source timeline."""
        if not 0 <= local_frame <= self.frame_count:
            raise ValueError("local_frame must be within the region")
        return self.start_frame + local_frame

    def rebase_seconds(self, local_seconds: float) -> float:
        """Map a local event timestamp onto the original source timeline."""
        if not 0.0 <= local_seconds <= self.duration_seconds:
            raise ValueError("local_seconds must be within the region")
        return self.source_offset_seconds + local_seconds

    def to_dict(self) -> dict[str, int | float]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_count": self.frame_count,
            "sample_rate": self.sample_rate,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds,
            "source_offset_frames": self.source_offset_frames,
            "source_offset_seconds": self.source_offset_seconds,
        }


@dataclass(frozen=True)
class ActivityRegionAnalysis:
    """Immutable regions plus JSON-safe diagnostics for one source file."""

    source: Path
    enabled: bool
    mode: ActivityMode
    threshold_dbfs: float
    sample_rate: int
    total_frames: int
    block_frames: int
    minimum_silence_frames: int
    pre_roll_frames: int
    post_roll_frames: int
    blocks_analyzed: int
    regions: tuple[AudioRegion, ...]
    skipped_ranges: tuple[AudioRegion, ...]

    @property
    def included_frames(self) -> int:
        return sum(region.frame_count for region in self.regions)

    @property
    def skipped_frames(self) -> int:
        return sum(region.frame_count for region in self.skipped_ranges)

    def to_dict(self) -> dict[str, object]:
        """Return metadata suitable for run diagnostics and cache inspection."""
        settings = {
            "block_frames": self.block_frames,
            "block_seconds": self.block_frames / self.sample_rate,
            "minimum_silence_frames": self.minimum_silence_frames,
            "minimum_silence_seconds": (self.minimum_silence_frames / self.sample_rate),
            "pre_roll_frames": self.pre_roll_frames,
            "pre_roll_seconds": self.pre_roll_frames / self.sample_rate,
            "post_roll_frames": self.post_roll_frames,
            "post_roll_seconds": self.post_roll_frames / self.sample_rate,
        }
        return {
            "source": str(self.source),
            "enabled": self.enabled,
            "mode": self.mode,
            "threshold_dbfs": self.threshold_dbfs,
            "sample_rate": self.sample_rate,
            "total_frames": self.total_frames,
            "duration_seconds": self.total_frames / self.sample_rate,
            "blocks_analyzed": self.blocks_analyzed,
            "settings": settings,
            "summary": {
                "region_count": len(self.regions),
                "included_frames": self.included_frames,
                "included_seconds": self.included_frames / self.sample_rate,
                "skipped_range_count": len(self.skipped_ranges),
                "skipped_frames": self.skipped_frames,
                "skipped_seconds": self.skipped_frames / self.sample_rate,
            },
            "regions": [region.to_dict() for region in self.regions],
            "skipped_ranges": [region.to_dict() for region in self.skipped_ranges],
        }


def analyze_activity_regions(
    audio_path: str | Path,
    *,
    enabled: bool = True,
    mode: ActivityMode = "rms",
    threshold_dbfs: float = -50.0,
    block_ms: float = 20.0,
    minimum_silence_seconds: float = 1.5,
    pre_roll_seconds: float = 0.75,
    post_roll_seconds: float = 0.25,
) -> ActivityRegionAnalysis:
    """Find padded active regions while preserving exact source-frame offsets.

    Blocks are active when their RMS reaches ``threshold_dbfs`` or, in
    ``exact-zero`` mode, when any sample is not exactly zero. Interior inactive
    runs shorter than ``minimum_silence_seconds`` remain within one region.
    Only confirmed silent interiors are skipped: leading and trailing source
    audio stays in the first and last regions. An entirely silent source is
    returned as one unsplit region. Padded regions that touch or overlap merge.
    """
    if mode not in ("rms", "exact-zero"):
        raise ValueError("mode must be 'rms' or 'exact-zero'")
    if not np.isfinite(threshold_dbfs):
        raise ValueError("threshold_dbfs must be finite")
    if block_ms <= 0:
        raise ValueError("block_ms must be positive")
    if minimum_silence_seconds <= 0:
        raise ValueError("minimum_silence_seconds must be positive")
    if pre_roll_seconds < 0 or post_roll_seconds < 0:
        raise ValueError("roll durations must be non-negative")

    source = Path(audio_path).expanduser().resolve()
    with sf.SoundFile(source) as audio_file:
        sample_rate = audio_file.samplerate
        total_frames = len(audio_file)
        block_frames = max(1, round(sample_rate * block_ms / 1000.0))
        minimum_silence_frames = round(minimum_silence_seconds * sample_rate)
        pre_roll_frames = round(pre_roll_seconds * sample_rate)
        post_roll_frames = round(post_roll_seconds * sample_rate)

        if not enabled:
            regions = (
                (AudioRegion(0, total_frames, sample_rate),) if total_frames else ()
            )
            return ActivityRegionAnalysis(
                source=source,
                enabled=False,
                mode=mode,
                threshold_dbfs=threshold_dbfs,
                sample_rate=sample_rate,
                total_frames=total_frames,
                block_frames=block_frames,
                minimum_silence_frames=minimum_silence_frames,
                pre_roll_frames=pre_roll_frames,
                post_roll_frames=post_roll_frames,
                blocks_analyzed=0,
                regions=regions,
                skipped_ranges=(),
            )

        threshold_power = 10.0 ** (threshold_dbfs / 10.0)
        active_blocks: list[tuple[int, int]] = []
        frame = 0
        blocks_analyzed = 0
        for block in audio_file.blocks(
            blocksize=block_frames,
            dtype="float32",
            always_2d=True,
        ):
            end = frame + len(block)
            blocks_analyzed += 1
            if _block_is_active(block, mode=mode, threshold_power=threshold_power):
                active_blocks.append((frame, end))
            frame = end

    raw_regions = _group_active_blocks(
        active_blocks,
        minimum_silence_frames=minimum_silence_frames,
    )
    padded = [
        (
            max(0, start - pre_roll_frames),
            min(total_frames, end + post_roll_frames),
        )
        for start, end in raw_regions
    ]
    if padded:
        padded[0] = (0, padded[0][1])
        padded[-1] = (padded[-1][0], total_frames)
    elif total_frames:
        padded = [(0, total_frames)]
    merged = _merge_ranges(padded)
    regions = tuple(
        AudioRegion(start, end, sample_rate) for start, end in merged if start < end
    )
    skipped_ranges = tuple(
        AudioRegion(start, end, sample_rate)
        for start, end in _complement_ranges(merged, total_frames)
        if start < end
    )
    return ActivityRegionAnalysis(
        source=source,
        enabled=True,
        mode=mode,
        threshold_dbfs=threshold_dbfs,
        sample_rate=sample_rate,
        total_frames=total_frames,
        block_frames=block_frames,
        minimum_silence_frames=minimum_silence_frames,
        pre_roll_frames=pre_roll_frames,
        post_roll_frames=post_roll_frames,
        blocks_analyzed=blocks_analyzed,
        regions=regions,
        skipped_ranges=skipped_ranges,
    )


def analyze_activity_buffer(
    audio: AudioBuffer,
    *,
    source: Path,
    enabled: bool = True,
    mode: ActivityMode = "rms",
    threshold_dbfs: float = -50.0,
    block_ms: float = 20.0,
    minimum_silence_seconds: float = 1.5,
    pre_roll_seconds: float = 0.75,
    post_roll_seconds: float = 0.25,
) -> ActivityRegionAnalysis:
    """Find activity regions from the immutable pipeline audio buffer."""
    if mode not in ("rms", "exact-zero"):
        raise ValueError("mode must be 'rms' or 'exact-zero'")
    sample_rate = audio.sample_rate
    total_frames = len(audio.samples)
    block_frames = max(1, round(sample_rate * block_ms / 1000.0))
    minimum_silence_frames = round(minimum_silence_seconds * sample_rate)
    pre_roll_frames = round(pre_roll_seconds * sample_rate)
    post_roll_frames = round(post_roll_seconds * sample_rate)
    if not enabled:
        regions = (AudioRegion(0, total_frames, sample_rate),) if total_frames else ()
        return ActivityRegionAnalysis(
            source=source,
            enabled=False,
            mode=mode,
            threshold_dbfs=threshold_dbfs,
            sample_rate=sample_rate,
            total_frames=total_frames,
            block_frames=block_frames,
            minimum_silence_frames=minimum_silence_frames,
            pre_roll_frames=pre_roll_frames,
            post_roll_frames=post_roll_frames,
            blocks_analyzed=0,
            regions=regions,
            skipped_ranges=(),
        )
    threshold_power = 10.0 ** (threshold_dbfs / 10.0)
    active_blocks = _active_blocks_from_buffer(
        audio.samples,
        block_frames=block_frames,
        mode=mode,
        threshold_power=threshold_power,
    )
    raw_regions = _group_active_blocks(
        active_blocks, minimum_silence_frames=minimum_silence_frames
    )
    padded = [
        (max(0, start - pre_roll_frames), min(total_frames, end + post_roll_frames))
        for start, end in raw_regions
    ]
    if padded:
        padded[0] = (0, padded[0][1])
        padded[-1] = (padded[-1][0], total_frames)
    elif total_frames:
        padded = [(0, total_frames)]
    merged = _merge_ranges(padded)
    regions = tuple(AudioRegion(start, end, sample_rate) for start, end in merged)
    skipped_ranges = tuple(
        AudioRegion(start, end, sample_rate)
        for start, end in _complement_ranges(merged, total_frames)
    )
    return ActivityRegionAnalysis(
        source=source,
        enabled=enabled,
        mode=mode,
        threshold_dbfs=threshold_dbfs,
        sample_rate=sample_rate,
        total_frames=total_frames,
        block_frames=block_frames,
        minimum_silence_frames=minimum_silence_frames,
        pre_roll_frames=pre_roll_frames,
        post_roll_frames=post_roll_frames,
        blocks_analyzed=(total_frames + block_frames - 1) // block_frames,
        regions=regions,
        skipped_ranges=skipped_ranges,
    )


def _block_is_active(
    block: np.ndarray,
    *,
    mode: ActivityMode,
    threshold_power: float,
) -> bool:
    if block.size == 0:
        return False
    if not np.all(np.isfinite(block)):
        return True
    if mode == "exact-zero":
        return bool(np.any(block != 0.0))
    power = float(np.mean(np.square(block, dtype=np.float64)))
    return power >= threshold_power


def _active_blocks_from_buffer(
    samples: np.ndarray,
    *,
    block_frames: int,
    mode: ActivityMode,
    threshold_power: float,
    batch_blocks: int = 512,
) -> list[tuple[int, int]]:
    """Classify bounded batches while preserving the scalar block contract."""
    total_frames = len(samples)
    active: list[tuple[int, int]] = []
    total_blocks = (total_frames + block_frames - 1) // block_frames
    for first_block in range(0, total_blocks, batch_blocks):
        block_count = min(batch_blocks, total_blocks - first_block)
        full_count = min(
            block_count,
            max(0, (total_frames - first_block * block_frames) // block_frames),
        )
        if full_count:
            first_frame = first_block * block_frames
            last_frame = first_frame + full_count * block_frames
            blocks = samples[first_frame:last_frame].reshape(
                full_count, block_frames, samples.shape[1]
            )
            finite = np.all(np.isfinite(blocks), axis=(1, 2))
            if mode == "exact-zero":
                mask = ~finite | np.any(blocks != 0.0, axis=(1, 2))
            elif np.all(finite):
                powers = np.mean(
                    np.square(blocks, dtype=np.float64),
                    axis=(1, 2),
                )
                mask = powers >= threshold_power
            else:
                mask = np.fromiter(
                    (
                        _block_is_active(
                            block,
                            mode=mode,
                            threshold_power=threshold_power,
                        )
                        for block in blocks
                    ),
                    dtype=bool,
                    count=full_count,
                )
            for offset in np.flatnonzero(mask):
                start = (first_block + int(offset)) * block_frames
                active.append((start, start + block_frames))
        tail_block = first_block + full_count
        if full_count < block_count:
            start = tail_block * block_frames
            end = min(total_frames, start + block_frames)
            if _block_is_active(
                samples[start:end], mode=mode, threshold_power=threshold_power
            ):
                active.append((start, end))
    return active


def _group_active_blocks(
    active_blocks: list[tuple[int, int]],
    *,
    minimum_silence_frames: int,
) -> list[tuple[int, int]]:
    if not active_blocks:
        return []
    groups: list[tuple[int, int]] = []
    start, end = active_blocks[0]
    for block_start, block_end in active_blocks[1:]:
        if block_start - end >= minimum_silence_frames:
            groups.append((start, end))
            start = block_start
        end = block_end
    groups.append((start, end))
    return groups


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _complement_ranges(
    ranges: list[tuple[int, int]],
    total_frames: int,
) -> list[tuple[int, int]]:
    skipped: list[tuple[int, int]] = []
    cursor = 0
    for start, end in ranges:
        if cursor < start:
            skipped.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total_frames:
        skipped.append((cursor, total_frames))
    return skipped
