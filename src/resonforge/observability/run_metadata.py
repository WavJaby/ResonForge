"""Beautified, JSON-safe run metadata for the ResonForge pipeline."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import sys
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from .artifacts import RunArtifactRegistry
from .metadata_types import AudioFileMetadata, RunMetadata
from .scheduler_telemetry import (
    CORE_TELEMETRY_FIELDS,
    telemetry_capabilities,
)


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
    effective_flags = json_safe(vars(args))
    return {
        "schema_version": 3,
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "finished_at": None,
        "elapsed_seconds": None,
        "performance": {},
        "effective_flags": effective_flags,
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


def _present_telemetry_fields(jobs: list[dict[str, Any]]) -> list[str]:
    """Keep core fields plus diagnostics that carry information in this run."""
    extra = sorted(
        {
            name
            for job in jobs
            for name, value in job.items()
            if name not in {*CORE_TELEMETRY_FIELDS, "key"}
            and value not in (None, False, 0, "", {}, [])
        }
    )
    return [*CORE_TELEMETRY_FIELDS, *extra]


def write_scheduler_telemetry(
    jobs: list[dict[str, Any]],
    output_path: str | Path,
    *,
    stem_artifacts: dict[str, Any] | None = None,
    relative_to: str | Path | None = None,
    registry: RunArtifactRegistry | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write compact, queryable scheduler rows and return descriptor + summary."""
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["key_id", *_present_telemetry_fields(jobs)]
    model_keys: list[dict[str, Any]] = []
    key_ids: dict[str, int] = {}
    rows: list[list[Any]] = []
    for job in jobs:
        key = json_safe(job.get("key", {}))
        canonical = json.dumps(key, sort_keys=True, separators=(",", ":"))
        key_id = key_ids.get(canonical)
        if key_id is None:
            key_id = len(model_keys)
            key_ids[canonical] = key_id
            model_keys.append(key)
        rows.append([key_id, *(json_safe(job.get(name)) for name in columns[1:])])
    payload = {
        "schema_version": 2,
        "kind": "scheduler_telemetry",
        "capabilities": telemetry_capabilities(jobs),
        "columns": columns,
        "model_keys": model_keys,
        "rows": rows,
        "stem_artifacts": json_safe(stem_artifacts or {}),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    temporary.replace(output)
    widths: dict[str, int] = {}
    scheduler_observations: dict[str, int] = {}
    scheduler_gauges: dict[str, int] = {}
    scheduler_state_histograms: dict[str, dict[str, int]] = {}
    scheduler_decision_phase_us: dict[str, float] = {}
    scheduler_action_wall_us: dict[str, float] = {}
    # GPU time by phase, `graph_capture` among them. Every layer below carried
    # this -- the job record, the worker that fills it, the capability list --
    # and only this summary dropped it, so a run recorded capture *counts* and
    # never capture *time*. P4's resize rule prices a width change against
    # exactly that number.
    phase_gpu_ms: dict[str, float] = {}
    scheduler_boundary_gap_us = 0.0
    scheduler_boundary_gap_count = 0
    for job in jobs:
        width = str(job.get("batch_size", 0))
        widths[width] = widths.get(width, 0) + 1
        for name, value in job.get("scheduler_observations", {}).items():
            scheduler_observations[name] = (
                scheduler_observations.get(name, 0) + int(value)
            )
        # Gauges are instantaneous device-wide readings taken by several runs
        # on the same device. Summing them would report one device repeatedly,
        # so the run keeps the highest reading of each.
        for name, value in job.get("scheduler_gauges", {}).items():
            scheduler_gauges[name] = max(
                scheduler_gauges.get(name, 0), int(value)
            )
        for name, histogram in job.get("scheduler_state_histograms", {}).items():
            target = scheduler_state_histograms.setdefault(name, {})
            for value, count in histogram.items():
                key = str(value)
                target[key] = target.get(key, 0) + int(count)
        for name, value in job.get("scheduler_decision_phase_us", {}).items():
            scheduler_decision_phase_us[name] = (
                scheduler_decision_phase_us.get(name, 0.0) + float(value)
            )
        for name, value in job.get("scheduler_action_wall_us", {}).items():
            scheduler_action_wall_us[name] = (
                scheduler_action_wall_us.get(name, 0.0) + float(value)
            )
        for name, value in job.get("phase_gpu_ms", {}).items():
            phase_gpu_ms[name] = phase_gpu_ms.get(name, 0.0) + float(value)
        scheduler_boundary_gap_us += float(
            job.get("scheduler_boundary_gap_us", 0.0)
        )
        scheduler_boundary_gap_count += int(
            job.get("scheduler_boundary_gap_count", 0)
        )
    summary = {
        "job_count": len(jobs),
        "model_key_count": len(model_keys),
        "queue_wait_seconds": round(
            sum(float(job.get("queue_wait_seconds", 0)) for job in jobs), 6
        ),
        "run_seconds": round(
            sum(float(job.get("run_seconds", 0)) for job in jobs), 6
        ),
        "generated_tokens": sum(
            int(job.get("generated_token_count", 0)) for job in jobs
        ),
        "wasted_token_rows": sum(
            int(job.get("wasted_token_rows", 0)) for job in jobs
        ),
        "jobs_by_batch_size": widths,
        "hot_replacements": sum(
            int(job.get("hot_replacements", 0)) for job in jobs
        ),
        "cuda_graph": {
            "captures": sum(int(job.get("cuda_graph_captures", 0)) for job in jobs),
            "replays": sum(int(job.get("cuda_graph_replays", 0)) for job in jobs),
            "evictions": sum(int(job.get("cuda_graph_evictions", 0)) for job in jobs),
            "cache_entries_peak": max(
                (int(job.get("cuda_graph_cache_entries_peak", 0)) for job in jobs),
                default=0,
            ),
            "cache_static_bytes_peak": max(
                (
                    int(job.get("cuda_graph_cache_static_bytes_peak", 0))
                    for job in jobs
                ),
                default=0,
            ),
            "cache_pool_bytes_peak": max(
                (
                    int(job.get("cuda_graph_cache_pool_bytes_peak", 0))
                    for job in jobs
                ),
                default=0,
            ),
        },
        # **Maximum, not sum.** These are a model's running totals, reported by
        # every job that model ran, so summing them multiplies one cache by the
        # number of jobs. Taking the largest reports it once.
        "prefill_graph": {
            name: max(
                (int(job.get("prefill_graph", {}).get(name, 0)) for job in jobs),
                default=0,
            )
            for name in ("captures", "replays", "hits", "misses", "entries", "refusals")
        },
        "prefill_graph_refusals": {
            name: max(
                (
                    int(job.get("prefill_graph_refusals", {}).get(name, 0))
                    for job in jobs
                ),
                default=0,
            )
            for name in sorted(
                {
                    reason
                    for job in jobs
                    for reason in job.get("prefill_graph_refusals", {})
                }
            )
        },
        "scheduler_observations": scheduler_observations,
        "scheduler_gauges": scheduler_gauges,
        "scheduler_state_histograms": scheduler_state_histograms,
        "scheduler_decision_phase_us": {
            name: round(value, 3)
            for name, value in scheduler_decision_phase_us.items()
        },
        "scheduler_action_wall_us": {
            name: round(value, 3) for name, value in scheduler_action_wall_us.items()
        },
        "phase_gpu_ms": {
            name: round(value, 3) for name, value in phase_gpu_ms.items()
        },
        "scheduler_boundary_gap_us": round(scheduler_boundary_gap_us, 3),
        "scheduler_boundary_gap_count": scheduler_boundary_gap_count,
        "scheduler_watchdog_timeout_seconds": max(
            (
                float(job.get("scheduler_watchdog_timeout_seconds", 0.0))
                for job in jobs
            ),
            default=0.0,
        ),
        "producer_decision_waits_peak": max(
            (
                int(job.get("producer_decision_waits_outstanding", 0))
                for job in jobs
            ),
            default=0,
        ),
        "producer_decision_waits_outstanding_final": (
            int(jobs[-1].get("producer_decision_waits_outstanding", 0))
            if jobs
            else 0
        ),
        "producer_decision_waits_unbound_final": (
            int(jobs[-1].get("producer_decision_waits_unbound", 0))
            if jobs
            else 0
        ),
    }
    descriptor = (
        registry.describe(
            output,
            kind="scheduler_telemetry",
            encoding="gzip+json-column-table",
            schema_version=2,
        )
        if registry is not None
        else {
            "path": (
                output.name
                if relative_to is None
                else output.relative_to(Path(relative_to).resolve()).as_posix()
            ),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "encoding": "gzip+json-column-table",
        }
    )
    return descriptor, summary


def write_scheduler_liveness(
    report: dict[str, Any],
    registry: RunArtifactRegistry,
) -> dict[str, Any]:
    """Persist full fail-closed evidence and return compact run metadata."""
    descriptor = registry.write_json_gzip(
        "scheduler-liveness.json.gz",
        report,
        kind="scheduler_liveness",
        schema_version=int(report.get("schema_version", 1)),
    )
    return {
        "classification": report.get("classification"),
        "message": report.get("message"),
        "state_version": report.get("state_version"),
        "progress_epoch": report.get("progress_epoch"),
        "enabled_action_count": len(report.get("enabled_actions", ())),
        "external_waits": report.get("external_waits", []),
        "artifact": descriptor,
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
