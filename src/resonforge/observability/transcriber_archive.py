"""Per-run transcriber output packaging."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import PurePosixPath
from typing import Any

from resonforge.transcribers.base import ProcessedTranscription

from .artifacts import RunArtifactRegistry

ARCHIVE_NAME = "transcriber.tar.gz"
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


def archive_memory_transcriptions(
    registry: RunArtifactRegistry,
    *,
    results: dict[str, ProcessedTranscription],
    models: dict[str, str],
    provenance: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Publish in-memory backend results without a transcriber workspace."""
    stems: dict[str, dict[str, Any]] = {}
    members: dict[str, bytes] = {}
    for stem, result in results.items():
        members[f"{stem}/raw.midi"] = result.raw_midi.data
        for artifact in result.artifacts:
            members[f"{stem}/{artifact.name}"] = artifact.data
        members[f"{stem}/clean.midi"] = result.clean_midi.data
        members[f"{stem}/selected.midi"] = result.selected_midi.data
        stems[stem] = {
            "backend": "muscriptor",
            "model": models[stem],
            "files": sorted(name for name in members if name.startswith(f"{stem}/")),
            "selected_midi": f"{stem}/selected.midi",
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": registry.run_id,
        "provenance": provenance or {},
        "stems": stems,
    }
    members[MANIFEST_NAME] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    archive = registry.path(ARCHIVE_NAME)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as bundle:
        for name, payload in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    temporary.replace(archive)
    return registry.describe(
        archive,
        kind="transcriber_outputs",
        encoding="tar+gzip",
        schema_version=SCHEMA_VERSION,
    )


def _safe_stem_member(stem: str, value: object) -> str:
    member = PurePosixPath(str(value))
    if member.is_absolute() or ".." in member.parts:
        raise RuntimeError(f"unsafe transcriber archive member: {value}")
    if len(member.parts) < 2 or member.parts[0] != stem:
        raise RuntimeError(f"transcriber member does not belong to {stem}: {value}")
    return member.as_posix()
