"""Input hashing, format inspection, temporary WAV staging, and cleanup."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

from .paths import (
    BS_CHANNELS,
    BS_INPUT_ROOT,
    BS_OUTPUT_ROOT,
    DEFAULT_BS_SAMPLE_RATE,
    SERVICE_ROOT,
)
from .process_runner import ProcessRunner

LOGGER = logging.getLogger(__name__)

def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_input(input_file: Path) -> tuple[Path, Path, str, str]:
    source = input_file.expanduser().resolve()
    require_file(source, "Input audio file")
    input_hash = sha256_file(source)
    base_name = source.stem
    output_dir = BS_OUTPUT_ROOT / base_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return source, output_dir, base_name, input_hash


def input_matches_staging_format(
    source: Path,
    *,
    sample_rate: int,
    channels: int = BS_CHANNELS,
) -> bool:
    """Return whether an input WAV can be copied without conversion."""
    if source.suffix.lower() != ".wav":
        return False
    try:
        info = sf.info(source)
    except RuntimeError:
        return False
    return info.samplerate == sample_rate and info.channels == channels


def build_ffmpeg_staging_command(
    ffmpeg: str,
    source: Path,
    destination: Path,
    *,
    sample_rate: int,
    channels: int = BS_CHANNELS,
) -> list[str | Path]:
    """Build the deterministic input conversion command."""
    return [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        source,
        "-map",
        "0:a:0",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_s24le",
        destination,
    ]


def stage_input(
    source: Path,
    input_hash: str,
    *,
    sample_rate: int = DEFAULT_BS_SAMPLE_RATE,
    runner: ProcessRunner | None = None,
    log_dir: Path | None = None,
) -> Path:
    temporary_root = BS_INPUT_ROOT / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    input_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{input_hash[:16]}-{os.getpid()}-",
            dir=temporary_root,
        )
    )
    staged = input_dir / f"{source.stem}.wav"
    try:
        if input_matches_staging_format(source, sample_rate=sample_rate):
            shutil.copy2(source, staged)
            LOGGER.info(f"Copied input: {source} -> {staged}")
            return input_dir

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError(
                f"ffmpeg is required to convert {source.suffix or 'audio'} input to WAV"
            )
        command = build_ffmpeg_staging_command(
            ffmpeg,
            source,
            staged,
            sample_rate=sample_rate,
        )
        if runner is not None:
            return_code = runner.run(
                command,
                cwd=SERVICE_ROOT,
                stdout_log=(
                    log_dir / "input-conversion.stdout.log"
                    if log_dir is not None
                    else None
                ),
                stderr_log=(
                    log_dir / "input-conversion.stderr.log"
                    if log_dir is not None
                    else None
                ),
            )
        else:
            completed = subprocess.run(
                [str(item) for item in command],
                cwd=SERVICE_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            return_code = completed.returncode
        if return_code or not staged.is_file():
            detail = (
                f"; log: {log_dir / 'input-conversion.stderr.log'}"
                if log_dir is not None
                else ""
            )
            raise RuntimeError(
                f"input conversion to WAV failed (exit code {return_code}){detail}"
            )
        LOGGER.info(f"Converted input: {source} -> {staged}")
        return input_dir
    except BaseException:
        shutil.rmtree(input_dir, ignore_errors=True)
        raise


def cleanup_staged_input(input_dir: Path) -> None:
    temporary_root = (BS_INPUT_ROOT / "tmp").resolve()
    resolved = input_dir.resolve()
    if resolved.parent != temporary_root:
        raise RuntimeError(f"refusing to remove non-temporary input: {resolved}")
    shutil.rmtree(resolved, ignore_errors=True)
    LOGGER.info(f"Removed temporary input: {resolved}")


