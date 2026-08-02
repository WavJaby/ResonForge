"""Filesystem and executable locations shared by pipeline modules."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[3]


def venv_executable(venv: Path, command: str) -> Path:
    """Return an executable path inside a virtual environment on this OS."""
    if os.name == "nt":
        return venv / "Scripts" / f"{command}.exe"
    return venv / "bin" / command


PYTHON = Path(sys.executable)
BASIC_PITCH_PROJECT = SERVICE_ROOT / "environments" / "basic-pitch"
BASIC_PITCH = venv_executable(BASIC_PITCH_PROJECT / ".venv", "basic-pitch")
BS_INPUT_ROOT = SERVICE_ROOT / "input"
BS_OUTPUT_ROOT = SERVICE_ROOT / "output"
MODELS_ROOT = SERVICE_ROOT / "models"
BS_MODEL_DIR = (
    MODELS_ROOT / "bs-roformer" / "roformer-model-bs-roformer-sw-by-jarredou"
)
BS_MODEL_PATH = BS_MODEL_DIR / "BS-Rofo-SW-Fixed.ckpt"
BS_CONFIG_PATH = BS_MODEL_DIR / "BS-Rofo-SW-Fixed.yaml"
DEFAULT_BS_SAMPLE_RATE = 44_100
BS_CHANNELS = 2
