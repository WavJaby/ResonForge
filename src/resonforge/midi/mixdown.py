"""Render MIDI tracks, mix audio, normalize, and write WAV or MP3 output."""

from __future__ import annotations

import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .merge import merge_midis
from .rendering import find_soundfont, render_midi

LOGGER = logging.getLogger(__name__)

MP3_BITRATES = (
    32,
    40,
    48,
    56,
    64,
    80,
    96,
    112,
    128,
    160,
    192,
    224,
    256,
    320,
)


@dataclass(frozen=True)
class MixdownConfig:
    midi: tuple[Path, ...]
    out: Path
    soundfont: Path | None = None
    sample_rate: int = 44_100
    gain_db: float = 0.0
    track_gains_db: str | None = None
    normalize_dbfs: float | None = None
    midi_normalize_velocity: int | None = None
    mp3_quality: float = 0.8
    mp3_bitrate: int | None = None
    jobs: int = 1
    render_timeout: float = 120.0
    polyphony: int = 256
    mix_device: str = "auto"
    midi_out: Path | None = None


def validate_mixdown_config(config: MixdownConfig) -> None:
    """Validate mixdown options and required local inputs."""
    if config.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if config.out.suffix.lower() not in {".wav", ".mp3"}:
        raise ValueError("--out must end in .wav or .mp3")
    if not 0.0 <= config.mp3_quality <= 1.0:
        raise ValueError("--mp3-quality must be between 0.0 and 1.0")
    if config.mp3_bitrate is not None and config.mp3_bitrate not in MP3_BITRATES:
        raise ValueError(f"unsupported MP3 bitrate: {config.mp3_bitrate}")
    if config.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if config.render_timeout <= 0:
        raise ValueError("--render-timeout must be positive")
    if config.polyphony <= 0:
        raise ValueError("--polyphony must be positive")
    if config.mix_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("--mix-device must be auto, cpu, or cuda")
    if config.mix_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--mix-device cuda requested but CUDA is unavailable")
    if config.normalize_dbfs is not None and config.normalize_dbfs > 0:
        raise ValueError("--normalize-dbfs must be 0 or lower")
    if (
        config.midi_normalize_velocity is not None
        and not 1 <= config.midi_normalize_velocity <= 127
    ):
        raise ValueError("--midi-normalize-velocity must be between 1 and 127")
    if not config.midi:
        raise ValueError("at least one MIDI input is required")
    if shutil.which("fluidsynth") is None:
        raise FileNotFoundError("fluidsynth not found on PATH")
    missing = [str(path) for path in config.midi if not path.is_file()]
    if missing:
        raise FileNotFoundError("MIDI file not found:\n" + "\n".join(missing))


def resolve_source_gains(config: MixdownConfig) -> list[float]:
    """Resolve global and per-track dB gains into one value per MIDI input."""
    if config.track_gains_db:
        try:
            track_gains = [
                float(value.strip()) for value in config.track_gains_db.split(",")
            ]
        except ValueError as error:
            raise ValueError(
                "--track-gains-db must be comma-separated numbers"
            ) from error
        if len(track_gains) != len(config.midi):
            raise ValueError(
                "--track-gains-db must contain exactly one value per MIDI input"
            )
    else:
        track_gains = [0.0] * len(config.midi)
    return [value + config.gain_db for value in track_gains]


def render_midi_tracks(
    config: MixdownConfig,
    soundfont: Path,
    output_dir: Path,
) -> list[Path]:
    """Render every MIDI source, optionally with parallel FluidSynth jobs."""
    rendered_files = [
        output_dir / f"{index}.wav" for index in range(1, len(config.midi) + 1)
    ]
    workers = min(config.jobs, len(config.midi))
    if workers == 1:
        for index, (midi, rendered) in enumerate(
            zip(config.midi, rendered_files, strict=True),
            start=1,
        ):
            LOGGER.info(f"[{index}/{len(config.midi)}] rendering {midi}")
            render_midi(
                midi,
                rendered,
                soundfont,
                config.sample_rate,
                config.render_timeout,
                config.polyphony,
            )
        return rendered_files

    LOGGER.info(
        f"rendering {len(config.midi)} MIDI files with {workers} parallel jobs"
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                render_midi,
                midi,
                rendered,
                soundfont,
                config.sample_rate,
                config.render_timeout,
                config.polyphony,
            ): midi
            for midi, rendered in zip(config.midi, rendered_files, strict=True)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            LOGGER.info(
                f"[{completed}/{len(config.midi)}] rendered {futures[future]}"
            )
    return rendered_files


def load_rendered_tracks(
    rendered_files: list[Path],
    source_gains_db: list[float],
    *,
    sample_rate: int,
    midi_files: tuple[Path, ...] | None = None,
    tail_seconds: float = 2.0,
) -> list[np.ndarray]:
    """Load stereo FluidSynth WAVs and apply per-source gains."""
    tracks: list[np.ndarray] = []
    for source_index, rendered in enumerate(rendered_files):
        frame_limit = -1
        if midi_files is not None:
            import mido

            expected_seconds = (
                mido.MidiFile(midi_files[source_index]).length + tail_seconds
            )
            frame_limit = round(expected_seconds * sample_rate)
        audio, actual_rate = sf.read(
            rendered,
            dtype="float32",
            always_2d=True,
            frames=frame_limit,
        )
        if actual_rate != sample_rate:
            raise RuntimeError(f"unexpected sample rate {actual_rate} from FluidSynth")
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]
        gain = 10.0 ** (source_gains_db[source_index] / 20.0)
        tracks.append(audio * gain)
    return tracks


def mix_audio_tracks(
    tracks: list[np.ndarray],
    *,
    device_name: str,
) -> torch.Tensor:
    """Sum variable-length stereo tracks on the requested torch device."""
    device = torch.device(device_name)
    length = max(len(track) for track in tracks)
    mixed = torch.zeros((length, 2), dtype=torch.float32, device=device)
    for track in tracks:
        mixed[: len(track)] += torch.from_numpy(track).to(device)
    return mixed


def normalize_audio_mix(
    mixed: torch.Tensor,
    *,
    target_dbfs: float | None,
) -> torch.Tensor:
    """Apply requested peak normalization or automatic clip protection."""
    peak = float(torch.max(torch.abs(mixed)).item())
    if target_dbfs is not None and peak > 0:
        target_peak = 10.0 ** (target_dbfs / 20.0)
        adjustment = target_peak / peak
        mixed *= adjustment
        LOGGER.info(
            f"peak normalize: {20 * np.log10(adjustment):+.2f} dB "
            f"(target {target_dbfs:g} dBFS)"
        )
    elif peak > 0.999:
        attenuation = 0.999 / peak
        mixed *= attenuation
        LOGGER.info(
            f"peak protection: {20 * np.log10(attenuation):.2f} dB "
            f"(unprotected peak {peak:.3f})"
        )
    return mixed


def write_audio_mix(
    config: MixdownConfig,
    mix: np.ndarray,
) -> str:
    """Write the completed mix and return a concise encoding description."""
    config.out.parent.mkdir(parents=True, exist_ok=True)
    if config.out.suffix.lower() == ".mp3":
        if config.mp3_bitrate is not None:
            compression = (320 - config.mp3_bitrate) / (320 - 32)
            sf.write(
                config.out,
                mix,
                config.sample_rate,
                format="MP3",
                subtype="MPEG_LAYER_III",
                bitrate_mode="CONSTANT",
                compression_level=compression,
            )
            return f"MP3 {config.mp3_bitrate} kbps CBR"
        sf.write(
            config.out,
            mix,
            config.sample_rate,
            format="MP3",
            subtype="MPEG_LAYER_III",
            bitrate_mode="VARIABLE",
            compression_level=config.mp3_quality,
        )
        return f"MP3 VBR quality {config.mp3_quality:g}"
    sf.write(config.out, mix, config.sample_rate, subtype="PCM_24")
    return "24-bit WAV"


def write_optional_merged_midi(
    config: MixdownConfig,
    source_gains_db: list[float],
) -> None:
    """Write the optional MIDI overlay, falling back to multiple ports."""
    if config.midi_out is None:
        return
    midi_output = (
        config.out.with_suffix(".mid")
        if config.midi_out == Path("__AUTO__")
        else config.midi_out
    )
    try:
        midi_mode = merge_midis(
            list(config.midi),
            midi_output,
            source_gains_db,
            config.midi_normalize_velocity,
        )
    except ValueError:
        LOGGER.info(
            "merged MIDI exceeds 15 melodic channels; using one MIDI port per source"
        )
        midi_mode = merge_midis(
            list(config.midi),
            midi_output,
            source_gains_db,
            config.midi_normalize_velocity,
            use_ports=True,
        )
    LOGGER.info(
        f"done: {midi_output} (merged MIDI, {len(config.midi)} sources, {midi_mode})"
    )


def execute_mixdown(config: MixdownConfig) -> Path:
    """Execute validated render, mix, normalization, and output phases."""
    validate_mixdown_config(config)
    soundfont = config.soundfont or find_soundfont()
    if not soundfont.is_file():
        raise FileNotFoundError(f"soundfont not found: {soundfont}")
    source_gains_db = resolve_source_gains(config)
    with tempfile.TemporaryDirectory(prefix="midi-mix-") as temp_dir:
        rendered = render_midi_tracks(config, soundfont, Path(temp_dir))
        tracks = load_rendered_tracks(
            rendered,
            source_gains_db,
            sample_rate=config.sample_rate,
            midi_files=config.midi,
        )

    device_name = (
        "cuda"
        if config.mix_device == "auto" and torch.cuda.is_available()
        else config.mix_device
    )
    if device_name == "auto":
        device_name = "cpu"
    LOGGER.info(f"mixing and peak normalization on {device_name}")
    mixed = mix_audio_tracks(tracks, device_name=device_name)
    mixed = normalize_audio_mix(
        mixed,
        target_dbfs=config.normalize_dbfs,
    )
    mix = mixed.cpu().numpy()
    encoding = write_audio_mix(config, mix)
    LOGGER.info(
        f"done: {config.out} "
        f"({len(config.midi)} tracks, "
        f"{len(mix) / config.sample_rate:.2f}s, "
        f"{config.sample_rate} Hz, {encoding})"
    )
    write_optional_merged_midi(config, source_gains_db)
    return config.out


def mixdown_midis(
    midi: list[Path],
    *,
    out: Path,
    soundfont: Path | None = None,
    sample_rate: int = 44_100,
    gain_db: float = 0.0,
    track_gains_db: str | None = None,
    normalize_dbfs: float | None = None,
    midi_normalize_velocity: int | None = None,
    mp3_quality: float = 0.8,
    mp3_bitrate: int | None = None,
    jobs: int = 1,
    render_timeout: float = 120.0,
    polyphony: int = 256,
    mix_device: str = "auto",
    midi_out: Path | None = None,
) -> Path:
    """Render and mix MIDI files through the importable typed API."""
    config = MixdownConfig(
        midi=tuple(Path(path) for path in midi),
        out=Path(out),
        soundfont=Path(soundfont) if soundfont is not None else None,
        sample_rate=sample_rate,
        gain_db=gain_db,
        track_gains_db=track_gains_db,
        normalize_dbfs=normalize_dbfs,
        midi_normalize_velocity=midi_normalize_velocity,
        mp3_quality=mp3_quality,
        mp3_bitrate=mp3_bitrate,
        jobs=jobs,
        render_timeout=render_timeout,
        polyphony=polyphony,
        mix_device=mix_device,
        midi_out=Path(midi_out) if midi_out is not None else None,
    )
    return execute_mixdown(config)


