"""Job-oriented service boundary shared by CLI and future network transports."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from resonforge.application.failures import (
    FailureKind,
    JobFailure,
    classify_failure,
)
from resonforge.observability.performance import PerformanceMetrics
from resonforge.pipeline.orchestrator import make_run_prefix, run_pipeline
from resonforge.pipeline.types import PipelineConfig, PipelineRunOutcome


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class JobRequest:
    config: PipelineConfig


@dataclass(frozen=True)
class JobResult:
    session_id: str
    status: JobStatus
    exit_code: int
    output_dir: Path | None
    metadata_path: Path | None
    performance: PerformanceMetrics | None
    # Typed, with the traceback kept. `str(failure)` is the old one-line form,
    # so a caller that only wants to print it does not have to change (D1-5).
    failure: JobFailure | None = None

    @property
    def error(self) -> str | None:
        return None if self.failure is None else str(self.failure)


@dataclass
class _JobRecord:
    request: JobRequest
    session_id: str
    stop_event: threading.Event
    completion: Future[JobResult]
    status: JobStatus = JobStatus.QUEUED
    worker: Future[None] | None = None


class JobHandle:
    """Stable control surface suitable for CLI, HTTP, or queue adapters."""

    def __init__(self, service: InProcessPipelineService, record: _JobRecord) -> None:
        self._service = service
        self._record = record

    @property
    def session_id(self) -> str:
        return self._record.session_id

    @property
    def status(self) -> JobStatus:
        return self._service._status(self._record)

    def cancel(self) -> bool:
        return self._service._cancel(self._record)

    def result(self, timeout: float | None = None) -> JobResult:
        return self._record.completion.result(timeout=timeout)


class PipelineService(Protocol):
    def submit(self, request: JobRequest) -> JobHandle:
        """Accept one immutable job and return its lifecycle handle."""

    def get(self, session_id: str) -> JobHandle | None:
        """Return one known job without exposing pipeline internals."""


class PipelineRunner(Protocol):
    def __call__(
        self,
        config: PipelineConfig,
        *,
        session_id: str,
        stop_event: threading.Event,
        device_job_slots: int,
    ) -> PipelineRunOutcome: ...


def _outcome_failure(outcome: PipelineRunOutcome | None) -> JobFailure | None:
    """Re-type a failure the pipeline already caught and recorded itself.

    The pipeline persists its own error into `metadata.json` and returns a
    message; the traceback is in the run's log, not here, so this is a
    classification without one rather than a fabricated one.
    """
    if outcome is None or outcome.error is None:
        return None
    return JobFailure(
        kind=FailureKind.CANCELLED
        if outcome.exit_code == 130
        else FailureKind.INTERNAL,
        type_name="",
        message=outcome.error,
        traceback_text="",
    )


class InProcessPipelineService:
    """Concurrent application host used by CLI now and HTTP later."""

    def __init__(
        self,
        *,
        max_concurrent_jobs: int = 1,
        pipeline_runner: PipelineRunner = run_pipeline,
    ) -> None:
        if max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be positive")
        self._pipeline_runner = pipeline_runner
        # The divisor the capacity solver needs is this host's real concurrency
        # bound, which only the service knows (D1-1).
        self._device_job_slots = max_concurrent_jobs
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_jobs,
            thread_name_prefix="pipeline-job",
        )
        self._jobs: dict[str, _JobRecord] = {}
        self._lock = threading.Lock()
        self._closed = False

    def submit(self, request: JobRequest) -> JobHandle:
        session_id = make_run_prefix()
        record = _JobRecord(
            request=request,
            session_id=session_id,
            stop_event=threading.Event(),
            completion=Future(),
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("pipeline service is closed")
            self._jobs[session_id] = record
            record.worker = self._executor.submit(self._run_job, record)
        return JobHandle(self, record)

    def get(self, session_id: str) -> JobHandle | None:
        with self._lock:
            record = self._jobs.get(session_id)
        return None if record is None else JobHandle(self, record)

    def close(self, *, cancel_running: bool = False) -> None:
        with self._lock:
            self._closed = True
            records = tuple(self._jobs.values())
        if cancel_running:
            for record in records:
                self._cancel(record)
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> InProcessPipelineService:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _run_job(self, record: _JobRecord) -> None:
        with self._lock:
            if record.stop_event.is_set():
                record.status = JobStatus.CANCELLED
            else:
                record.status = JobStatus.RUNNING
        if record.status == JobStatus.CANCELLED:
            record.completion.set_result(self._result(record, 130))
            return
        failure: JobFailure | None = None
        outcome: PipelineRunOutcome | None = None
        try:
            outcome = self._pipeline_runner(
                record.request.config,
                session_id=record.session_id,
                stop_event=record.stop_event,
                device_job_slots=self._device_job_slots,
            )
            exit_code = outcome.exit_code
        except Exception as raised:  # service boundary contains job failures
            exit_code = 1
            failure = classify_failure(raised)
        with self._lock:
            record.status = (
                JobStatus.CANCELLED
                if exit_code == 130
                else JobStatus.COMPLETE
                if exit_code == 0
                else JobStatus.FAILED
            )
        record.completion.set_result(
            self._result(record, exit_code, outcome=outcome, failure=failure)
        )

    def _result(
        self,
        record: _JobRecord,
        exit_code: int,
        *,
        outcome: PipelineRunOutcome | None = None,
        failure: JobFailure | None = None,
    ) -> JobResult:
        return JobResult(
            session_id=record.session_id,
            status=record.status,
            exit_code=exit_code,
            output_dir=None if outcome is None else outcome.output_dir,
            metadata_path=None if outcome is None else outcome.metadata_path,
            performance=None if outcome is None else outcome.performance,
            failure=(
                failure
                if failure is not None
                else _outcome_failure(outcome)
            ),
        )

    def _status(self, record: _JobRecord) -> JobStatus:
        with self._lock:
            return record.status

    def _cancel(self, record: _JobRecord) -> bool:
        with self._lock:
            if record.status in {
                JobStatus.COMPLETE,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                return False
            record.status = JobStatus.CANCELLING
            record.stop_event.set()
            return True
