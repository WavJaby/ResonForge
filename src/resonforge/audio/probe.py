"""Read audio file properties without decoding the stream."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def probe_audio(path: Path) -> dict[str, Any]:
    """Read local audio properties and estimate average file bitrate."""

    size = path.stat().st_size
    result: dict[str, Any] = {"file_size_bytes": size}
    try:
        import soundfile as sf

        info = sf.info(str(path))
        duration = float(info.duration)
        result.update(
            {
                "container": info.format,
                "codec": info.subtype,
                "sample_rate_hz": int(info.samplerate),
                "channels": int(info.channels),
                "duration_seconds": round(duration, 3),
            }
        )
        if duration > 0:
            result["average_bitrate_kbps"] = round(
                size * 8 / duration / 1000,
                2,
            )
    except Exception as error:
        result["probe_error"] = f"{type(error).__name__}: {error}"
    return result
