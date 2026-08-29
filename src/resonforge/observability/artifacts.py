"""Canonical paths and descriptors for durable run artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactDescriptor:
    """Portable metadata for one file owned by a pipeline run."""

    kind: str
    path: str
    encoding: str
    sha256: str
    size_bytes: int
    schema_version: int | None = None
    scope: str = "run"

    def to_dict(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


class RunArtifactRegistry:
    """Allocate, write, and describe artifacts under one durable run root."""

    def __init__(
        self, output_dir: str | Path, run_id: str, *, session_root: bool = False
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.run_id = run_id
        self.directory = (
            self.output_dir / "artifacts"
            if session_root
            else self.output_dir / "artifacts" / run_id
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self._descriptors: dict[tuple[str, str], dict[str, object]] = {}

    @property
    def descriptors(self) -> list[dict[str, object]]:
        return list(self._descriptors.values())

    def path(self, name: str) -> Path:
        if Path(name).name != name:
            raise ValueError("artifact names must not contain directories")
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / name

    def describe(
        self,
        path: str | Path,
        *,
        kind: str,
        encoding: str,
        schema_version: int | None = None,
        scope: str = "run",
    ) -> dict[str, object]:
        artifact = Path(path).resolve()
        relative = artifact.relative_to(self.output_dir).as_posix()
        descriptor = ArtifactDescriptor(
            kind=kind,
            path=relative,
            encoding=encoding,
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            size_bytes=artifact.stat().st_size,
            schema_version=schema_version,
            scope=scope,
        ).to_dict()
        self._descriptors[(kind, relative)] = descriptor
        return descriptor

    def write_json_gzip(
        self,
        name: str,
        payload: Any,
        *,
        kind: str,
        schema_version: int,
    ) -> dict[str, object]:
        output = self.path(name)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
        temporary.replace(output)
        return self.describe(
            output,
            kind=kind,
            encoding="gzip+json",
            schema_version=schema_version,
        )
