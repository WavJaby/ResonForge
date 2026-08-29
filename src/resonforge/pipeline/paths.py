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
BS_INPUT_ROOT = SERVICE_ROOT / "input"
BS_OUTPUT_ROOT = SERVICE_ROOT / "output"
# Separated stems are content-addressed by (input hash, sample rate), so they
# belong to the input, not to any one run's output tree. Keeping them here
# rather than under the run's output root is what lets `--output-root` give
# each experiment its own directory without re-separating the same audio for
# every arm — separation is the pipeline's most expensive per-song fixed cost.
BS_CACHE_ROOT = SERVICE_ROOT / "output" / ".separation-cache"
MODELS_ROOT = SERVICE_ROOT / "models"
BS_MODEL_DIR = (
    MODELS_ROOT / "bs-roformer" / "roformer-model-bs-roformer-sw-by-jarredou"
)
BS_MODEL_PATH = BS_MODEL_DIR / "BS-Rofo-SW-Fixed.ckpt"
BS_CONFIG_PATH = BS_MODEL_DIR / "BS-Rofo-SW-Fixed.yaml"
DEFAULT_BS_SAMPLE_RATE = 44_100
BS_CHANNELS = 2
