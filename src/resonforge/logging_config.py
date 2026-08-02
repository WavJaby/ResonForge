"""Run-scoped Python logging configuration."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm

_FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MANAGED_HANDLER = "_resonforge_managed"


@dataclass(frozen=True)
class RunLogs:
    pipeline: Path
    huggingface: Path


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
    pipeline_log = output_directory / f"{run_prefix}-pipeline.log"
    huggingface_log = detail_directory / "huggingface.log"

    project_logger = logging.getLogger("resonforge")
    _clear_managed_handlers(project_logger)
    project_logger.addHandler(_file_handler(pipeline_log))

    stdout_handler = _DynamicStreamHandler("stdout")
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(_MaximumLevelFilter(logging.INFO))
    stdout_handler.addFilter(_ConsoleVisibilityFilter())
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(stdout_handler, _MANAGED_HANDLER, True)
    project_logger.addHandler(stdout_handler)

    stderr_handler = _DynamicStreamHandler("stderr")
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.addFilter(_ConsoleVisibilityFilter())
    stderr_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    setattr(stderr_handler, _MANAGED_HANDLER, True)
    project_logger.addHandler(stderr_handler)
    project_logger.setLevel(logging.INFO)
    project_logger.propagate = False

    huggingface_logger = logging.getLogger("huggingface_hub")
    _clear_managed_handlers(huggingface_logger, remove_all=True)
    huggingface_logger.addHandler(_file_handler(huggingface_log))
    huggingface_logger.setLevel(logging.INFO)
    huggingface_logger.propagate = False
    return RunLogs(pipeline=pipeline_log, huggingface=huggingface_log)


def configure_muscriptor_loggers(
    log_dir: str | Path,
    stem: str,
) -> StemLogs:
    """Create isolated stdout/stderr loggers for one MuScriptor stem."""
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stdout_path = directory / f"{stem}.muscriptor.stdout.log"
    stderr_path = directory / f"{stem}.muscriptor.stderr.log"
    loggers: dict[str, logging.Logger] = {}
    for channel, path in (("stdout", stdout_path), ("stderr", stderr_path)):
        logger = logging.getLogger(f"muscriptor.output.{stem}.{channel}")
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

