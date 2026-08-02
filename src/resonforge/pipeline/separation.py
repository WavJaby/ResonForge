"""BS-RoFormer stem definitions, cache handling, and execution."""

from __future__ import annotations

import logging
from pathlib import Path

from .paths import (
    BS_CHANNELS,
    BS_CONFIG_PATH,
    BS_MODEL_PATH,
    DEFAULT_BS_SAMPLE_RATE,
    PYTHON,
    SERVICE_ROOT,
)
from .process_runner import ProcessRunner
from .types import StemTask

LOGGER = logging.getLogger(__name__)

SEPARATION_HASH_FILENAME = "separation-input.sha256.txt"
BS_STEMS_DIRNAME = ".bs_roformer"


def make_tasks(
    stem_dir: Path,
    base_name: str,
    other_instruments: str,
) -> list[StemTask]:
    specifications = (
        ("bass", "electric_bass"),
        ("drums", "drums"),
        ("guitar", "clean_electric_guitar"),
        ("piano", "electric_piano"),
        ("other", other_instruments),
        ("vocals", "voice"),
    )
    return [
        StemTask(name, stem_dir / f"{base_name}_{name}.wav", instruments)
        for name, instruments in specifications
    ]


def separation_cache_key(input_hash: str, sample_rate: int) -> str:
    return f"{input_hash}\nbs_sample_rate={sample_rate}\nbs_channels={BS_CHANNELS}"


def separation_cache_is_valid(
    output_dir: Path,
    input_hash: str,
    tasks: list[StemTask],
    sample_rate: int = DEFAULT_BS_SAMPLE_RATE,
) -> bool:
    hash_file = output_dir / SEPARATION_HASH_FILENAME
    try:
        cached_hash = hash_file.read_text(encoding="ascii").strip()
    except OSError:
        return False
    cache_key = separation_cache_key(input_hash, sample_rate)
    return cached_hash == cache_key and all(task.audio.is_file() for task in tasks)


def write_separation_hash(
    output_dir: Path,
    input_hash: str,
    sample_rate: int = DEFAULT_BS_SAMPLE_RATE,
) -> Path:
    hash_file = output_dir / SEPARATION_HASH_FILENAME
    temporary = hash_file.with_suffix(hash_file.suffix + ".tmp")
    temporary.write_text(
        separation_cache_key(input_hash, sample_rate) + "\n",
        encoding="ascii",
    )
    temporary.replace(hash_file)
    return hash_file


def separate_stems(
    input_dir: Path,
    output_dir: Path,
    log_dir: Path,
    runner: ProcessRunner,
) -> None:
    LOGGER.info("\n== Separating stems with BS-RoFormer ==")
    return_code = runner.run(
        [
            PYTHON,
            "-u",
            "-m",
            "bs_roformer.inference",
            "--input_folder",
            input_dir,
            "--store_dir",
            output_dir,
            "--model_path",
            BS_MODEL_PATH,
            "--config_path",
            BS_CONFIG_PATH,
        ],
        cwd=SERVICE_ROOT,
        stdout_log=log_dir / "separation.stdout.log",
        stderr_log=log_dir / "separation.stderr.log",
        show_stdout=True,
        show_stderr=True,
    )
    if return_code:
        raise RuntimeError(
            f"BS-RoFormer separation failed (exit code {return_code}); "
            f"log: {log_dir / 'separation.stderr.log'}"
        )
    LOGGER.info(f"  separated stems -> {output_dir}")


