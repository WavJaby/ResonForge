"""FluidSynth discovery and single-MIDI rendering."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def find_soundfont() -> Path:
    """Resolve the soundfont through MuScriptor's cache-aware helper."""
    try:
        from muscriptor.utils.auralization import _resolve_soundfont

        return _resolve_soundfont(None)
    except Exception as error:
        raise RuntimeError(
            "could not resolve MuseScore_General.sf2; pass --soundfont "
            f"path\\to\\soundfont.sf2\n{error}"
        ) from error


def render_midi(
    midi: Path,
    wav: Path,
    soundfont: Path,
    sample_rate: int,
    timeout: float,
    polyphony: int,
) -> None:
    """Render one MIDI file to WAV with a bounded FluidSynth process."""
    process = subprocess.Popen(
        [
            "fluidsynth",
            "-ni",
            "-o",
            f"synth.polyphony={polyphony}",
            "-F",
            str(wav),
            "-r",
            str(sample_rate),
            str(soundfont),
            str(midi),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        ),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        else:
            process.kill()
        process.wait()
        raise RuntimeError(
            f"FluidSynth timed out after {timeout:g}s for {midi}"
        ) from None
    if process.returncode != 0 or not wav.exists():
        detail = (stderr or stdout)[-1000:]
        raise RuntimeError(f"FluidSynth failed for {midi}:\n{detail}")
