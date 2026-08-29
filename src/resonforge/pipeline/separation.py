"""BS-RoFormer stem definitions, cache handling, and execution."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from resonforge.transcribers.base import StemTask

from ..runtime.process_runner import Cancelled, ProcessRunner
from .paths import (
    BS_CHANNELS,
    BS_CONFIG_PATH,
    BS_MODEL_PATH,
    DEFAULT_BS_SAMPLE_RATE,
    PYTHON,
    SERVICE_ROOT,
)

LOGGER = logging.getLogger(__name__)

BS_STEMS_DIRNAME = ".bs_roformer"
_SEPARATION_SLOT = threading.Semaphore(1)


@contextmanager
def global_separation_admission(stop_event: threading.Event):
    """Serialize cache-miss BS-RoFormer processes across pipeline sessions.

    And, since A2e, the one moment the KV supply is asked to shrink: a
    separation subprocess is about to want this card, and transcription is
    holding memory it measurably is not using -- 83% of the pool sits above the
    highest lent page. Released here rather than on a timer because this is the
    only place the *reason* exists; a timer would be a second answer to a
    question this semaphore already answers.

    Nothing is refused if the release returns little or nothing. Separation
    already fits today (146 MiB of headroom, measured); what this buys is the
    room to stop capping the pool for the worst moment.
    """
    while not _SEPARATION_SLOT.acquire(timeout=0.1):
        if stop_event.is_set():
            raise Cancelled()
    try:
        if stop_event.is_set():
            raise Cancelled()
        # Imported here: this module must not pull MuScriptor in at load time,
        # and the pipeline must not know what a KV page is.
        from resonforge.transcribers.muscriptor.transcription import (
            release_idle_kv_supply,
        )

        # Logged unconditionally, including the zero case: the first two-song
        # run released nothing and the log could not say whether that was "no
        # pool exists yet" or "the tail is pinned". Those call for opposite
        # fixes, so the line carries all three numbers.
        released, pools, releasable = release_idle_kv_supply()
        LOGGER.info(
            "KV supply before separation: %d pool(s), %.0f MiB idle, "
            "%.0f MiB released",
            pools,
            releasable / 1024**2,
            released / 1024**2,
        )
        yield
    finally:
        _SEPARATION_SLOT.release()


# One separated stem carries a whole perceptual family, not one GM voice: the
# guitar stem holds nylon, clean electric and distorted takes alike. Naming a
# single member forced every note in the stem to that member's program, which
# cost nothing under a merged instrument space but made exact-program scoring
# unwinnable wherever the arrangement used another member of the same family.
#
# General MIDI programs, not backend group names (T9/D9-1). Nothing above the
# transcriber interface names a MuScriptor group any more; the adapter
# translates inward and declares what it cannot reach. Program 128 is the
# channel-10 drum sentinel, which GM has no program for.
# `test_instrument_vocabulary.py` pins that these resolve to exactly the group
# names they replaced.
DRUM_PROGRAM = 128
STEM_INSTRUMENT_PROGRAMS: dict[str, tuple[int, ...]] = {
    "bass": (32, 33),                # acoustic bass, finger electric bass
    "drums": (DRUM_PROGRAM,),
    "guitar": (24, 27, 30),          # nylon, clean electric, overdriven
    "piano": (0, 4),                 # acoustic grand, electric
    "vocals": (52,),                 # choir aahs
}


def program_spec(programs: tuple[int, ...]) -> str:
    """Render a program set the way the transcriber interface carries it."""
    return ",".join(str(program) for program in programs)


def make_tasks(
    stem_dir: Path,
    other_instruments: str,
) -> list[StemTask]:
    specifications = (
        ("bass", program_spec(STEM_INSTRUMENT_PROGRAMS["bass"])),
        ("drums", program_spec(STEM_INSTRUMENT_PROGRAMS["drums"])),
        ("guitar", program_spec(STEM_INSTRUMENT_PROGRAMS["guitar"])),
        ("piano", program_spec(STEM_INSTRUMENT_PROGRAMS["piano"])),
        ("other", other_instruments),
        ("vocals", program_spec(STEM_INSTRUMENT_PROGRAMS["vocals"])),
    )
    return [
        StemTask(name, stem_dir / f"{name}.wav", instruments)
        for name, instruments in specifications
    ]


def separation_cache_key(input_hash: str, sample_rate: int) -> str:
    return f"{input_hash}\nbs_sample_rate={sample_rate}\nbs_channels={BS_CHANNELS}"


def separation_cache_id(input_hash: str, sample_rate: int) -> str:
    return hashlib.sha256(
        separation_cache_key(input_hash, sample_rate).encode("ascii")
    ).hexdigest()[:8]


def separation_cache_is_valid(
    output_dir: Path,
    input_hash: str,
    tasks: list[StemTask],
    sample_rate: int = DEFAULT_BS_SAMPLE_RATE,
) -> bool:
    expected = separation_cache_id(input_hash, sample_rate)
    return output_dir.name == expected and all(task.audio.is_file() for task in tasks)


@contextmanager
def separation_cache_lock(cache_dir: Path):
    """Hold an OS file lock while one process publishes a cache entry."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".lock"
    handle = lock_path.open("a+b")
    if lock_path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError:
                    time.sleep(0.1)
        else:  # pragma: no cover - production host is Windows
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            acquired = True
        yield lock_path
    finally:
        if acquired and os.name == "nt":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        elif acquired:  # pragma: no cover
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


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


