"""Beautified, JSON-safe run metadata for the ResonForge pipeline."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from .metadata_types import AudioFileMetadata, RunMetadata


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    return str(value)


def audio_file_info(path: str | Path) -> AudioFileMetadata:
    source = Path(path).resolve()
    result: AudioFileMetadata = {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "extension": source.suffix.lower(),
    }
    try:
        info = sf.info(source)
    except (RuntimeError, TypeError):
        return result
    result.update(
        {
            "duration_seconds": round(info.duration, 6),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype,
        }
    )
    return result


def package_provenance(distribution_name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False}
    result: dict[str, Any] = {
        "installed": True,
        "version": distribution.version,
    }
    direct_url_entry = next(
        (
            entry
            for entry in (distribution.files or ())
            if entry.name == "direct_url.json"
        ),
        None,
    )
    direct_url = (
        Path(distribution.locate_file(direct_url_entry))
        if direct_url_entry is not None
        else None
    )
    if direct_url is not None and direct_url.is_file():
        with suppress(OSError, json.JSONDecodeError):
            result["direct_url"] = json.loads(
                direct_url.read_text(encoding="utf-8")
            )
    return result


def create_run_metadata(run_id: str, args: Any) -> RunMetadata:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "finished_at": None,
        "elapsed_seconds": None,
        "command": {
            "argv": list(sys.argv),
            "flags": json_safe(vars(args)),
        },
        "input": {},
        "software": {
            "python": sys.version,
            "packages": {
                name: package_provenance(name)
                for name in (
                    "resonforge",
                    "bs-roformer-infer",
                    "muscriptor",
                    "librosa",
                    "mido",
                    "torch",
                )
            },
        },
        "models": {},
        "separation": {},
        "stems": {},
        "tempo": None,
        "outputs": {},
        "error": None,
    }


def write_run_metadata(metadata: RunMetadata, output_path: str | Path) -> Path:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output
