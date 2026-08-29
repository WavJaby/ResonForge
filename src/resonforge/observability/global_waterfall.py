"""Process-safe global SQLite event store for scheduler and pipeline spans."""

from __future__ import annotations

import atexit
import json
import os
import queue
import sqlite3
import threading
import time
import uuid
from concurrent.futures import Future
from pathlib import Path

from resonforge.runtime.paths import TELEMETRY_DB as DEFAULT_WATERFALL_DB


class GlobalWaterfallStore:
    """Serialize telemetry writes off scheduler threads and expose time queries."""

    def __init__(self, path: Path = DEFAULT_WATERFALL_DB) -> None:
        self.path = path
        self.process_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._queue: queue.Queue[tuple[str, object, Future[object] | None]] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="waterfall-sqlite-writer",
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    def begin_session(self, session_id: str, *, label: str = "") -> int:
        started = time.time_ns()
        self._submit("begin", (session_id, label, started))
        return started

    def end_session(self, session_id: str, *, status: str) -> int:
        ended = time.time_ns()
        self._submit("end", (session_id, status, ended))
        return ended

    def record(self, event: dict[str, object]) -> None:
        self._submit("event", dict(event))

    def record_span(
        self,
        action: str,
        started_wall_ns: int,
        ended_wall_ns: int,
        *,
        pipeline_session_id: str,
        resource: str,
        lane: str,
        **details: object,
    ) -> None:
        self.record(
            {
                "action": action,
                "resource": resource,
                "lane": lane,
                "pipeline_session_id": pipeline_session_id,
                "start_wall_ns": started_wall_ns,
                "end_wall_ns": ended_wall_ns,
                **details,
            }
        )

    def flush(self) -> None:
        future: Future[object] = Future()
        self._submit("flush", None, future)
        future.result(timeout=10)

    def session_events(
        self,
        session_id: str,
        *,
        include_context: bool = True,
    ) -> list[dict[str, object]]:
        self.flush()
        with sqlite3.connect(self.path) as connection:
            bounds = connection.execute(
                "SELECT started_wall_ns, ended_wall_ns FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if bounds is None:
                return []
            started, ended = int(bounds[0]), int(bounds[1] or time.time_ns())
            rows = connection.execute(
                """
                SELECT payload_json FROM waterfall_events
                WHERE end_wall_ns >= ? AND start_wall_ns <= ?
                  AND (? OR pipeline_session_id = ?)
                ORDER BY start_wall_ns, id
                """,
                (started, ended, int(include_context), session_id),
            )
            events = [json.loads(row[0]) for row in rows]
        duration_ns = max(1, ended - started)
        for event in events:
            event_start = max(started, int(event["start_wall_ns"]))
            event_end = min(ended, int(event["end_wall_ns"]))
            event["start_seconds"] = (event_start - started) / 1_000_000_000
            event["end_seconds"] = (event_end - started) / 1_000_000_000
            event["view_owner_session_id"] = session_id
            event["view_duration_seconds"] = duration_ns / 1_000_000_000
        return events

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        future: Future[object] = Future()
        self._queue.put(("close", None, future))
        future.result(timeout=10)
        self._thread.join(timeout=10)

    def _submit(
        self,
        operation: str,
        payload: object,
        future: Future[object] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("global waterfall store is closed")
        self._queue.put((operation, payload, future))

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            self._create_schema(connection)
            while True:
                operation, payload, future = self._queue.get()
                try:
                    if operation == "close":
                        connection.commit()
                        if future is not None:
                            future.set_result(None)
                        return
                    if operation == "begin":
                        session_id, label, started = payload
                        connection.execute(
                            """
                            INSERT OR REPLACE INTO sessions
                            (session_id, process_id, label, started_wall_ns, status)
                            VALUES (?, ?, ?, ?, 'running')
                            """,
                            (session_id, self.process_id, label, started),
                        )
                    elif operation == "end":
                        session_id, status, ended = payload
                        connection.execute(
                            "UPDATE sessions SET ended_wall_ns = ?, status = ? WHERE session_id = ?",
                            (ended, status, session_id),
                        )
                    elif operation == "event":
                        self._write_event(connection, payload)
                    connection.commit()
                    if future is not None:
                        future.set_result(None)
                except BaseException as error:
                    if future is not None:
                        future.set_exception(error)

    def _write_event(self, connection: sqlite3.Connection, event: object) -> None:
        value = dict(event)
        start = int(value["start_wall_ns"])
        end = int(value["end_wall_ns"])
        owner = value.get("pipeline_session_id")
        cursor = connection.execute(
            """
            INSERT INTO waterfall_events
            (process_id, pipeline_session_id, action, resource, target_device,
             model, job_type, start_wall_ns, end_wall_ns, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.process_id,
                owner,
                value.get("action", ""),
                value.get("resource", ""),
                value.get("target_device", ""),
                value.get("model", ""),
                value.get("job_type", ""),
                start,
                end,
                json.dumps(value, separators=(",", ":"), default=str),
            ),
        )
        event_id = int(cursor.lastrowid)
        for state in ("before", "after"):
            for slot in value.get(f"slots_{state}", ()):
                connection.execute(
                    """
                    INSERT INTO waterfall_slots
                    (event_id, state, pipeline_session_id, internal_session_id,
                     slot, sequence, job_type, paused)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        state,
                        slot.get("pipeline_session_id"),
                        slot.get("session_id"),
                        slot.get("slot"),
                        slot.get("sequence"),
                        slot.get("job_type"),
                        int(bool(slot.get("paused", False))),
                    ),
                )

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                process_id TEXT NOT NULL,
                label TEXT NOT NULL DEFAULT '',
                started_wall_ns INTEGER NOT NULL,
                ended_wall_ns INTEGER,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS waterfall_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_id TEXT NOT NULL,
                pipeline_session_id TEXT,
                action TEXT NOT NULL,
                resource TEXT,
                target_device TEXT,
                model TEXT,
                job_type TEXT,
                start_wall_ns INTEGER NOT NULL,
                end_wall_ns INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_waterfall_time
                ON waterfall_events(start_wall_ns, end_wall_ns);
            CREATE INDEX IF NOT EXISTS idx_waterfall_owner
                ON waterfall_events(pipeline_session_id, start_wall_ns);
            CREATE TABLE IF NOT EXISTS waterfall_slots (
                event_id INTEGER NOT NULL REFERENCES waterfall_events(id) ON DELETE CASCADE,
                state TEXT NOT NULL,
                pipeline_session_id TEXT,
                internal_session_id INTEGER,
                slot INTEGER,
                sequence INTEGER,
                job_type TEXT,
                paused INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_waterfall_slot_owner
                ON waterfall_slots(pipeline_session_id, event_id);
            """
        )


_STORE: GlobalWaterfallStore | None = None
_STORE_LOCK = threading.Lock()


def global_waterfall_store() -> GlobalWaterfallStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = GlobalWaterfallStore()
        return _STORE


def close_global_waterfall_store() -> None:
    global _STORE
    with _STORE_LOCK:
        store = _STORE
        _STORE = None
    if store is not None:
        store.close()


atexit.register(close_global_waterfall_store)
