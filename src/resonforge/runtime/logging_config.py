"""Run-scoped Python logging configuration."""

from __future__ import annotations

import gzip
import hashlib
import logging
import shutil
import sys
import tarfile
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm

_FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MANAGED_HANDLER = "_resonforge_managed"
_CONSOLE_HANDLER = "_resonforge_console"
_RUN_SESSION: ContextVar[str | None] = ContextVar(
    "resonforge_run_session",
    default=None,
)
_CONFIGURATION_LOCK = threading.Lock()


@dataclass(frozen=True)
class RunLogs:
    directory: Path
    pipeline: Path


@dataclass(frozen=True)
class StemLogs:
    stdout_logger: logging.Logger
    stderr_logger: logging.Logger
    stdout_path: Path
    stderr_path: Path

    def close(self) -> None:
        _clear_managed_handlers(self.stdout_logger, remove_all=True)
        _clear_managed_handlers(self.stderr_logger, remove_all=True)


class _MaximumLevelFilter(logging.Filter):
    def __init__(self, maximum: int) -> None:
        super().__init__()
        self.maximum = maximum

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.maximum


class _ConsoleVisibilityFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "console", True)


class _RunSessionFilter(logging.Filter):
    def __init__(self, session_id: str) -> None:
        super().__init__()
        self.session_id = session_id

    def filter(self, record: logging.LogRecord) -> bool:
        return _RUN_SESSION.get() == self.session_id


class _DynamicStreamHandler(logging.StreamHandler):
    """Resolve stdout/stderr at emit time for tests and embedded callers."""

    def __init__(self, stream_name: str) -> None:
        super().__init__()
        self.stream_name = stream_name

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(
                self.format(record),
                file=getattr(sys, self.stream_name),
            )
        except Exception:  # noqa: BLE001 - logging must not break the pipeline
            self.handleError(record)


def _clear_managed_handlers(
    logger: logging.Logger,
    *,
    remove_all: bool = False,
) -> None:
    for handler in list(logger.handlers):
        if remove_all or getattr(handler, _MANAGED_HANDLER, False):
            logger.removeHandler(handler)
            handler.close()


def _file_handler(output: Path, *, raw: bool = False) -> logging.FileHandler:
    handler = logging.FileHandler(output, mode="w", encoding="utf-8")
    formatter = "%(message)s" if raw else _FILE_FORMAT
    handler.setFormatter(logging.Formatter(formatter, datefmt=_DATE_FORMAT))
    setattr(handler, _MANAGED_HANDLER, True)
    return handler


@contextmanager
def bind_run_logging(session_id: str):
    """Route project logs emitted by this execution context to one run."""
    token = _RUN_SESSION.set(session_id)
    try:
        yield
    finally:
        _RUN_SESSION.reset(token)


def configure_run_logging(
    output_dir: str | Path,
    log_dir: str | Path,
    run_prefix: str,
) -> RunLogs:
    """Configure file logging plus INFO/stdout and WARNING/stderr consoles."""
    output_directory = Path(output_dir)
    detail_directory = Path(log_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    detail_directory.mkdir(parents=True, exist_ok=True)
    pipeline_log = detail_directory / "pipeline.log"

    project_logger = logging.getLogger("resonforge")
    with _CONFIGURATION_LOCK:
        session_handler = _file_handler(pipeline_log)
        session_handler.addFilter(_RunSessionFilter(run_prefix))
        project_logger.addHandler(session_handler)

        if not any(
            getattr(handler, _CONSOLE_HANDLER, False)
            for handler in project_logger.handlers
        ):
            stdout_handler = _DynamicStreamHandler("stdout")
            stdout_handler.setLevel(logging.INFO)
            stdout_handler.addFilter(_MaximumLevelFilter(logging.INFO))
            stdout_handler.addFilter(_ConsoleVisibilityFilter())
            stdout_handler.setFormatter(logging.Formatter("%(message)s"))
            setattr(stdout_handler, _MANAGED_HANDLER, True)
            setattr(stdout_handler, _CONSOLE_HANDLER, True)
            project_logger.addHandler(stdout_handler)

            stderr_handler = _DynamicStreamHandler("stderr")
            stderr_handler.setLevel(logging.WARNING)
            stderr_handler.addFilter(_ConsoleVisibilityFilter())
            stderr_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            setattr(stderr_handler, _MANAGED_HANDLER, True)
            setattr(stderr_handler, _CONSOLE_HANDLER, True)
            project_logger.addHandler(stderr_handler)
        project_logger.setLevel(logging.INFO)
        project_logger.propagate = False

    return RunLogs(
        directory=detail_directory,
        pipeline=pipeline_log,
    )


def configure_model_download_logging(models_root: str | Path) -> Path:
    """Route shared Hugging Face download logs outside session directories."""
    log_path = Path(models_root) / "logs" / "huggingface.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("huggingface_hub")
    target = str(log_path.resolve())
    if not any(
        str(getattr(handler, "baseFilename", "")) == target
        for handler in logger.handlers
    ):
        _clear_managed_handlers(logger)
        logger.addHandler(_file_handler(log_path))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return log_path


def configure_muscriptor_loggers(
    log_dir: str | Path,
    stem: str,
) -> StemLogs:
    """Create isolated stdout/stderr loggers for one MuScriptor stem."""
    directory = Path(log_dir) / "muscriptor" / stem
    directory.mkdir(parents=True, exist_ok=True)
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    loggers: dict[str, logging.Logger] = {}
    session_key = hashlib.blake2s(
        str(directory.resolve()).encode("utf-8"), digest_size=4
    ).hexdigest()
    for channel, path in (("stdout", stdout_path), ("stderr", stderr_path)):
        logger = logging.getLogger(
            f"muscriptor.output.{session_key}.{stem}.{channel}"
        )
        _clear_managed_handlers(logger, remove_all=True)
        logger.addHandler(_file_handler(path, raw=True))
        logger.setLevel(logging.INFO)
        logger.propagate = False
        loggers[channel] = logger
    return StemLogs(
        stdout_logger=loggers["stdout"],
        stderr_logger=loggers["stderr"],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def finalize_run_logs(
    log_dir: str | Path,
    *,
    status: str,
    retention: str,
) -> dict[str, object]:
    """Close run handlers, then prune or gzip logs according to policy."""
    directory = Path(log_dir).resolve()
    if retention not in {"errors", "all", "none"}:
        raise ValueError(f"unsupported log retention policy: {retention}")

    for logger_value in logging.Logger.manager.loggerDict.values():
        if not isinstance(logger_value, logging.Logger):
            continue
        for handler in tuple(logger_value.handlers):
            path = getattr(handler, "baseFilename", None)
            if path is None:
                continue
            try:
                Path(path).resolve().relative_to(directory)
            except ValueError:
                continue
            logger_value.removeHandler(handler)
            handler.close()

    keep = retention == "all" or (retention == "errors" and status != "complete")
    if not keep:
        if directory.is_dir():
            shutil.rmtree(directory)
        return {"retained": False, "policy": retention, "files": []}

    retained: list[str] = []
    muscriptor_directory = directory / "muscriptor"
    if muscriptor_directory.is_dir():
        archive = directory / "muscriptor.tar.gz"
        temporary = directory / "muscriptor.tar.gz.tmp"
        with tarfile.open(temporary, "w:gz") as bundle:
            bundle.add(muscriptor_directory, arcname="muscriptor")
        temporary.replace(archive)
        shutil.rmtree(muscriptor_directory)
        retained.append(archive.relative_to(directory).as_posix())
    for source in sorted(path for path in directory.rglob("*") if path.is_file()):
        if source.name == "muscriptor.tar.gz":
            continue
        if source.stat().st_size == 0:
            source.unlink()
            continue
        if source.suffix == ".gz":
            retained.append(source.relative_to(directory).as_posix())
            continue
        target = source.with_suffix(source.suffix + ".gz")
        temporary = target.with_suffix(target.suffix + ".tmp")
        with (
            source.open("rb") as input_stream,
            gzip.open(temporary, "wb", compresslevel=9) as output_stream,
        ):
            shutil.copyfileobj(input_stream, output_stream)
        temporary.replace(target)
        source.unlink()
        retained.append(target.relative_to(directory).as_posix())
    return {"retained": bool(retained), "policy": retention, "files": retained}
