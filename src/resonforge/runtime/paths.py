"""Process-level file locations shared across packages."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Scheduler waterfall: large, cheap to regenerate, one file per checkout.
# Deleting it costs nothing but the history of runs already reported, which is
# why it belongs under the directory the project treats as disposable.
TELEMETRY_DB = (
    Path(__file__).resolve().parents[3] / "output" / "telemetry" / "resonforge.sqlite3"
)


def _durable_data_root() -> Path:
    """A per-user location that outlives `output/` and this checkout.

    Calibration is a property of the *host*, not of a working copy: every row
    is keyed on an environment fingerprint, and two checkouts on one machine
    measure the same device. Storing it per-checkout would make a second clone
    re-probe hardware it already knows, and storing it under `output/` made a
    routine cleanup cost a warm-up (D12-1).

    `device_lease` falls back to a temporary directory when it cannot find a
    user data root; that is right for a lock, which must not outlive the
    process, and wrong here. A calibration store in a temp directory would be
    silently rebuilt on every reboot, so this raises instead of degrading.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    if not base:
        raise RuntimeError(
            "no per-user data directory available; "
            "set LOCALAPPDATA (Windows) or XDG_DATA_HOME (POSIX)"
        )
    return Path(base) / "ResonForge"


def temporary_data_root() -> Path:
    """Fallback root for locks and other state that must not outlive a boot."""
    return Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
