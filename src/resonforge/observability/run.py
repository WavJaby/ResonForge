"""Version-aware access to one ResonForge run and its artifacts."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class RunRecord:
    """Read one metadata file without exposing storage encodings to callers."""

    def __init__(self, metadata_path: str | Path) -> None:
        self.metadata_path = Path(metadata_path).resolve()
        with self.metadata_path.open("r", encoding="utf-8") as stream:
            self.metadata: dict[str, Any] = json.load(stream)

    @classmethod
    def open(cls, metadata_path: str | Path) -> RunRecord:
        return cls(metadata_path)

    def artifact_path(self, descriptor: dict[str, Any]) -> Path:
        path = Path(str(descriptor["path"]))
        return path if path.is_absolute() else self.metadata_path.parent / path

    def scheduler_jobs(self) -> Iterator[dict[str, Any]]:
        if self.metadata.get("schema_version", 1) < 2:
            yield from self.metadata.get("transcription_timing", {}).get(
                "model_worker_jobs", []
            )
            return
        descriptor = self.metadata.get("telemetry")
        if not descriptor:
            return
        with gzip.open(
            self.artifact_path(descriptor), "rt", encoding="utf-8"
        ) as stream:
            telemetry = json.load(stream)
        columns = telemetry["columns"]
        keys = telemetry["model_keys"]
        for values in telemetry["rows"]:
            row = dict(zip(columns, values, strict=True))
            row["key"] = keys[row.pop("key_id")]
            yield row

    def model_trace_summaries(self) -> list[dict[str, Any]]:
        import numpy as np

        summaries: list[dict[str, Any]] = []
        stems = self.metadata.get("transcription_timing", {}).get("stems", {})
        archive_descriptor = self.metadata.get("transcription_timing", {}).get(
            "transcriber_archive"
        )
        for stem, timing in stems.items():
            for artifact in timing.get("model_traces", []):
                if artifact.get("member") and archive_descriptor:
                    with tarfile.open(
                        self.artifact_path(archive_descriptor), "r:gz"
                    ) as bundle:
                        stream = bundle.extractfile(artifact["member"])
                        if stream is None:
                            raise FileNotFoundError(artifact["member"])
                        trace_bytes = io.BytesIO(stream.read())
                    trace_source = trace_bytes
                else:
                    trace_source = self.artifact_path(artifact)
                with np.load(trace_source) as trace:
                    manifest = json.loads(trace["manifest_json"].tobytes())
                summaries.append(
                    {
                        "stem": stem,
                        "path": artifact.get("member", artifact.get("path")),
                        "rows": artifact["rows"],
                        "bytes": artifact.get("bytes", artifact.get("size_bytes")),
                        "sha256": artifact["sha256"],
                        "level": artifact["level"],
                        "groups": manifest,
                    }
                )
        return summaries

    def scheduler_liveness_report(self) -> dict[str, Any] | None:
        """Read the full scheduler-liveness artifact when one is registered."""
        summary = self.metadata.get("scheduler_liveness")
        if not isinstance(summary, dict):
            return None
        descriptor = summary.get("artifact")
        if not isinstance(descriptor, dict):
            return summary
        path = self.artifact_path(descriptor)
        if not path.is_file():
            return summary
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            report = json.load(stream)
        if not isinstance(report, dict):
            raise ValueError("scheduler liveness artifact must contain an object")
        return report

    def summary(self) -> dict[str, Any]:
        timing = self.metadata.get("transcription_timing", {})
        jobs = list(self.scheduler_jobs())

        def model_name(job: dict[str, Any]) -> object:
            key = job.get("key", {})
            return key.get("model", key.get("identity", {}).get("model"))

        return {
            "run_id": self.metadata.get("run_id"),
            "schema_version": self.metadata.get("schema_version", 1),
            "status": self.metadata.get("status"),
            "elapsed_seconds": self.metadata.get("elapsed_seconds"),
            "queue_elapsed_seconds": timing.get("elapsed_seconds"),
            "scheduler_jobs": len(jobs),
            "models": sorted(
                {
                    str(model_name(job))
                    for job in jobs
                    if model_name(job) is not None
                }
            ),
            "batch_widths": sorted(
                {int(job["batch_size"]) for job in jobs if job.get("batch_size")}
            ),
            "hot_replacements": sum(
                int(job.get("hot_replacements", 0)) for job in jobs
            ),
        }

    def validate_artifacts(self) -> list[dict[str, object]]:
        """Return validation rows; missing or corrupt files are data, not errors."""
        results: list[dict[str, object]] = []
        for descriptor in self.metadata.get("artifacts", []):
            path = self.artifact_path(descriptor)
            exists = path.is_file()
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
            expected = descriptor.get("sha256")
            results.append(
                {
                    "kind": descriptor.get("kind"),
                    "path": descriptor.get("path"),
                    "exists": exists,
                    "sha256_matches": exists and actual == expected,
                }
            )
        return results
