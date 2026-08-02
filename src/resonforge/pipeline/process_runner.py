"""Cancellable subprocess execution with live-flushed log streams."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO


class Cancelled(RuntimeError):
    """Raised when a pipeline cancellation stops a child process."""


class ProcessRunner:
    """Runs children with live-flushed logs and owns their cancellation."""

    def __init__(
        self,
        stop_event: threading.Event,
        *,
        model_cache: Path | None = None,
    ) -> None:
        self.stop_event = stop_event
        self.model_cache = model_cache
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen[bytes]] = set()

    def request_stop(self) -> None:
        self.stop_event.set()
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            self._interrupt(process)

    @staticmethod
    def _interrupt(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGINT)
        except (OSError, ValueError):
            with suppress(OSError):
                process.terminate()

    @staticmethod
    def _kill_tree(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        else:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)

    @staticmethod
    def _pump(source: BinaryIO, destinations: Sequence[BinaryIO]) -> None:
        try:
            while chunk := source.read(4096):
                for destination in destinations:
                    destination.write(chunk)
                    destination.flush()
        finally:
            source.close()

    def run(
        self,
        command: Sequence[str | Path],
        *,
        cwd: Path,
        environment_overrides: dict[str, str] | None = None,
        stdout_log: Path | None = None,
        stderr_log: Path | None = None,
        show_stdout: bool = False,
        show_stderr: bool = False,
    ) -> int:
        if self.stop_event.is_set():
            raise Cancelled()

        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        if self.model_cache is not None:
            environment["HF_HUB_CACHE"] = str(self.model_cache)
        if environment_overrides:
            environment.update(environment_overrides)
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        )
        popen_kwargs = {"start_new_session": True} if os.name != "nt" else {}
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=creation_flags,
            **popen_kwargs,
        )
        with self._lock:
            self._processes.add(process)

        stdout_file = stdout_log.open("wb", buffering=0) if stdout_log else None
        stderr_file = stderr_log.open("wb", buffering=0) if stderr_log else None
        stdout_destinations = (
            ([stdout_file] if stdout_file is not None else [])
            + ([sys.stdout.buffer] if show_stdout or stdout_file is None else [])
        )
        stderr_destinations = (
            ([stderr_file] if stderr_file is not None else [])
            + ([sys.stderr.buffer] if show_stderr or stderr_file is None else [])
        )
        pumps = [
            threading.Thread(
                target=self._pump,
                args=(process.stdout, stdout_destinations),
                daemon=True,
            ),
            threading.Thread(
                target=self._pump,
                args=(process.stderr, stderr_destinations),
                daemon=True,
            ),
        ]
        for pump in pumps:
            pump.start()

        interrupted_at: float | None = None
        killed = False
        try:
            while process.poll() is None:
                if self.stop_event.wait(0.1):
                    if interrupted_at is None:
                        self._interrupt(process)
                        interrupted_at = time.monotonic()
                    elif not killed and time.monotonic() - interrupted_at >= 2.0:
                        self._kill_tree(process)
                        killed = True
            for pump in pumps:
                pump.join(timeout=1)
            if self.stop_event.is_set():
                raise Cancelled()
            return process.returncode
        finally:
            with self._lock:
                self._processes.discard(process)
            if process.poll() is None:
                process.kill()
                process.wait()
            for stream in (stdout_file, stderr_file):
                if stream is not None:
                    stream.flush()
                    stream.close()
