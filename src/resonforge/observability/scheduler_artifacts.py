"""Serialize scheduler events and publish analysis views."""

from __future__ import annotations

import csv
import gzip
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from resonforge.observability.artifacts import RunArtifactRegistry
from resonforge.scheduler.events import SchedulerEvent

from .scheduler_waterfall import (
    ACTION_COLORS,
    render_scheduler_waterfall_svg,
)


def write_scheduler_waterfall(
    events: list[SchedulerEvent],
    output_dir: str | Path,
    run_prefix: str,
    *,
    log_dir: str | Path | None = None,
    registry: RunArtifactRegistry | None = None,
    source_database: str | Path | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Render one bounded view; global SQLite remains the durable source."""
    root = Path(output_dir).resolve()
    logs = root if log_dir is None else Path(log_dir).resolve()
    logs.mkdir(parents=True, exist_ok=True)
    svg_path = root / f"{run_prefix}-scheduler-waterfall.svg"
    svg_path.write_text(render_scheduler_waterfall_svg(events), encoding="utf-8")
    return {
        "events": len(events),
        "session_id": session_id,
        "database": str(Path(source_database).resolve()) if source_database else None,
        "svg": (
            registry.describe(
                svg_path,
                kind="scheduler_waterfall_view",
                encoding="svg",
                schema_version=2,
            )
            if registry is not None
            else _descriptor(svg_path, "svg")
        ),
    }


def _write_excel_csv(events: list[SchedulerEvent], path: Path) -> None:
    """Flatten event/slot state and expose ready-made stacked-bar series."""
    actions = tuple(ACTION_COLORS)
    fields = (
        "event",
        "resource",
        "target_device",
        "thread",
        "lane",
        "session_id",
        "session_label",
        "slot",
        "sequence",
        "job_type",
        "paused",
        "action",
        "color",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "active_before",
        "active_after",
        "occupied_before",
        "occupied_after",
        "kv_capacity",
        "resident_bytes",
        "displaced_count",
        "graph_static_bytes",
        "owned_total_bytes",
        "kv_capacity",
        "pending_jobs",
        "prefill_jobs",
        *(f"{action}_seconds" for action in actions),
    )
    with _open_csv_text(path) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for event_index, event in enumerate(events):
            start = float(event["start_seconds"])
            end = float(event["end_seconds"])
            duration = max(0.0, end - start)
            action = str(event["action"])
            slots = event.get("slots_before") or event.get("slots_after") or ({},)
            for slot in slots:
                session_id = slot.get("session_id", event.get("session_id", ""))
                session_label = str(event.get("session_label", ""))
                slot_index = slot.get("slot", "")
                row = {
                    "event": event_index,
                    "resource": event.get("resource", "gpu"),
                    "target_device": event.get("target_device", ""),
                    "thread": event.get("thread", ""),
                    "lane": event.get("lane") or (
                        f"{session_label} slot {slot_index}"
                        if slot_index != ""
                        else session_label
                    ),
                    "session_id": session_id,
                    "session_label": session_label,
                    "slot": slot_index,
                    "sequence": slot.get("sequence", ""),
                    "job_type": slot.get("job_type", ""),
                    "paused": slot.get("paused", ""),
                    "action": action,
                    "color": ACTION_COLORS.get(action, "#64748b"),
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": duration,
                    "active_before": event.get("active_before", 0),
                    "active_after": event.get("active_after", 0),
                    "occupied_before": event.get("occupied_before", 0),
                    "occupied_after": event.get("occupied_after", 0),
                    "capacity": event.get("capacity", 0),
                    "resident_bytes": event.get("resident_bytes", 0),
                    "displaced_count": event.get("displaced_count", 0),
                    "graph_static_bytes": event.get("graph_static_bytes", 0),
                    "owned_total_bytes": event.get("owned_total_bytes", 0),
                    "kv_capacity": event.get("kv_capacity", ""),
                    "pending_jobs": event.get("pending_jobs", 0),
                    "prefill_jobs": event.get("prefill_jobs", 0),
                    **{
                        f"{candidate}_seconds": (
                            duration if action == candidate else ""
                        )
                        for candidate in actions
                    },
                }
                writer.writerow(row)


@contextmanager
def _open_csv_text(path: Path):
    if path.suffix == ".gz":
        with gzip.open(
            path, "wt", encoding="utf-8-sig", newline=""
        ) as stream:
            yield stream
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        yield stream


def _descriptor(
    path: Path,
    encoding: str,
    *,
    relative_to: Path | None = None,
) -> dict[str, str]:
    return {
        "path": (
            path.name
            if relative_to is None
            else path.relative_to(relative_to).as_posix()
        ),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "encoding": encoding,
    }
