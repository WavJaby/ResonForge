"""Command-line interface for the ResonForge pipeline."""

import argparse
import faulthandler
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from resonforge.application import InProcessPipelineService, JobRequest
from resonforge.application.request import (
    ConfigurationError,
    build_pipeline_configs,
    unavailable_gpu_error,
)
from resonforge.observability.performance import (
    PerformanceMetrics,
    aggregate_performance,
)
from resonforge.pipeline.transcription import OTHER_INSTRUMENTS
from resonforge.pipeline.types import PipelineConfig
from resonforge.transcribers.registry import (
    DEFAULT_TRANSCRIBER,
    available_specs,
)


def _parse_batch_size(value: str) -> int | tuple[tuple[str, int], ...]:
    """Accept `auto`, one width for every model, or `MODEL=N` per model.

    The per-model form exists because a symmetric width is not expressive
    enough once two models share one device: their rows cost different
    amounts, so the widths that fit the device invariant are asymmetric.
    """
    text = value.strip()
    if text.lower() == "auto":
        return 0
    if "=" in text:
        widths: dict[str, int] = {}
        for entry in text.split(","):
            model, separator, raw = entry.partition("=")
            model = model.strip().lower()
            if not separator or not model:
                raise argparse.ArgumentTypeError(
                    "per-model batch size must use MODEL=ROWS entries"
                )
            try:
                width = int(raw)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"batch size for {model} must be an integer"
                ) from error
            if width < 1:
                raise argparse.ArgumentTypeError("batch size must be at least 1")
            if model in widths:
                raise argparse.ArgumentTypeError(f"duplicate model: {model}")
            widths[model] = width
        return tuple(sorted(widths.items()))
    try:
        parsed = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "batch size must be 'auto', an integer, or MODEL=ROWS entries"
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("batch size must be at least 1")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _parse_model_widths(value: str) -> tuple[tuple[str, int], ...]:
    entries: list[tuple[str, int]] = []
    for raw_entry in value.split(","):
        name, separator, raw_width = raw_entry.strip().partition("=")
        if not separator or name not in {"small", "medium", "large"}:
            raise argparse.ArgumentTypeError(
                "model widths must use MODEL=ROWS for small, medium, or large"
            )
        try:
            width = int(raw_width)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "model width must be an integer"
            ) from error
        if width < 1:
            raise argparse.ArgumentTypeError(
                "model width must be at least 1"
            )
        entries.append((name, width))
    if len({name for name, _width in entries}) != len(entries):
        raise argparse.ArgumentTypeError("model width entries must be unique")
    return tuple(entries)


@dataclass(frozen=True)
class CliInvocation:
    configs: tuple[PipelineConfig, ...]
    max_concurrent_jobs: int


def parse_args() -> CliInvocation:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_files",
        nargs="+",
        type=Path,
        help=(
            "One or more input audio files sharing all options; non-WAV cache "
            "misses are converted temporarily "
            "to WAV under input/tmp/."
        ),
    )
    parser.add_argument(
        "--song-jobs",
        type=_positive_int,
        default=2,
        metavar="COUNT",
        help="Maximum songs processed concurrently (default: 2)",
    )
    parser.add_argument(
        "--transcribers",
        default="",
        metavar="STEM:BACKEND[.MODEL],...",
        help=(
            "per-stem MuScriptor models, for example "
            "drums:muscriptor.medium,bass:muscriptor.small; "
            f"omitted stems use {DEFAULT_TRANSCRIBER} unless default:SPEC is "
            f"set; choices: {', '.join(available_specs())}"
        ),
    )
    parser.add_argument(
        "--bs-sample-rate",
        dest="bs_sample_rate",
        type=int,
        default=44_100,
        metavar="HZ",
        help=(
            "sample rate used for the stereo WAV passed to BS-RoFormer (default: 44100)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=_parse_batch_size,
        default=0,
        metavar="auto|ROWS|MODEL=ROWS,...",
        help=(
            "Maximum independent MuScriptor rows per model generation; "
            "auto uses free VRAM and splits batches on OOM (default: auto). "
            "MODEL=ROWS forces one width per model, which is the only form "
            "that can express an asymmetric vector"
        ),
    )
    parser.add_argument(
        "--disable-muscriptor-anomaly-detection",
        dest="muscriptor_anomaly_detection",
        action="store_false",
        help="Disable rolling stall and repetition diagnostics",
    )
    parser.set_defaults(muscriptor_anomaly_detection=True)
    parser.add_argument(
        "--muscriptor-recovery-model",
        choices=("small", "medium", "large"),
        default="medium",
        help=(
            "MuScriptor model used for independent greedy Recovery B "
            "when recovery is enabled (default: medium)"
        ),
    )
    parser.add_argument(
        "--disable-muscriptor-silence-split",
        dest="muscriptor_silence_split",
        action="store_false",
        help="Transcribe each MuScriptor stem as one full-length region",
    )
    parser.set_defaults(muscriptor_silence_split=True)
    parser.add_argument(
        "--disable-muscriptor-overlap-detection",
        dest="muscriptor_overlap_detection",
        action="store_false",
        help="Disable low-overlap detection on primary chunks",
    )
    parser.set_defaults(muscriptor_overlap_detection=True)
    parser.add_argument(
        "--disable-muscriptor-recovery",
        dest="muscriptor_recovery",
        action="store_false",
        help="Disable A/B candidate generation; unsafe chunks are still discarded",
    )
    parser.set_defaults(muscriptor_recovery=True)
    parser.add_argument(
        "--muscriptor-fresh-reanchor",
        dest="muscriptor_fresh_reanchor",
        action="store_true",
        help=(
            "Enable the last-resort unprompted re-anchor taken when primary, "
            "both A/B candidates and the safe frontier all fail. Off by "
            "default: it costs a full chunk on the larger model and has never "
            "been shown to improve output"
        ),
    )
    parser.set_defaults(muscriptor_fresh_reanchor=False)
    parser.add_argument(
        "--muscriptor-runtime",
        choices=("torch-eager", "cuda-eager", "cuda-graphs"),
        default="cuda-graphs",
        help=(
            "Generation execution line: which decode step runs and which "
            "attention kernel it runs on (default: cuda-graphs = replay a "
            "captured graph over our contiguous CUDA kernel; cuda-eager = the "
            "same kernel with per-step decode, which is what a CUDA profiler "
            "can be pointed at; torch-eager = per-step decode over dense "
            "PyTorch attention, the line that needs no CUDA kernel of ours). "
            "Paging is not on this axis -- every line pages."
        ),
    )
    parser.add_argument(
        "--muscriptor-prefill-runtime",
        choices=("eager", "cuda-graphs"),
        default="cuda-graphs",
        help=(
            "How a prefill forward runs, independently of --muscriptor-runtime "
            "(default: cuda-graphs = replay a captured forward per position "
            "bucket; eager = issue it per admission). Two reasons this is a "
            "flag: a device may not support capture, and what it is worth is a "
            "property of the host -- 10.30 -> 4.63 ms at width 1 on BAIR's "
            "fp16 cards, ~2%% on a compute-6.1 fp32 card whose forward is "
            "already saturated. It carries no kernel choice; the prefill runs "
            "whichever attention backend --muscriptor-runtime selected"
        ),
    )
    parser.add_argument(
        "--log-retention",
        choices=("errors", "all", "none"),
        default="errors",
        help=(
            "Retain compressed run logs for failures only, every run, or no run "
            "(default: errors)"
        ),
    )
    parser.add_argument(
        "--other-instruments",
        default=OTHER_INSTRUMENTS,
        metavar="NAMES",
        help=(
            "comma-separated instruments the other stem may transcribe into: "
            "MuScriptor group names, family aliases (brass, reed, string, "
            "guitar, bass, keyboard, mallet, synth), or General MIDI program "
            "numbers; run 'muscriptor list-instruments' for the full set"
        ),
    )
    parser.add_argument(
        "--disable-gate-presence-filter",
        dest="gate_presence_filter",
        action="store_false",
        help=(
            "Do not analyze and skip globally absent selected stems before "
            "transcription."
        ),
    )
    parser.set_defaults(gate_presence_filter=True)
    parser.add_argument("--only-other", action="store_true")
    parser.add_argument(
        "--skip",
        dest="skip_stems",
        default="",
        metavar="STEMS",
        help=(
            "comma-separated stems to exclude from transcription and mixdown "
            "(bass,drums,guitar,piano,other,vocal; band with --combine-band)"
        ),
    )
    parser.add_argument(
        "--combine-band",
        nargs="?",
        const="bass,guitar,piano",
        default="",
        metavar="STEMS",
        help=(
            "comma-separated stems to mix and transcribe once as band "
            "(at least two); without STEMS defaults to bass,guitar,piano"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory to create each song's output directory under "
            "(default: output/). Give an experiment its own root to keep its "
            "runs and separation caches together instead of interleaved with "
            "every other run under one song name."
        ),
    )
    parser.add_argument(
        "--treat-original-as-other",
        action="store_true",
        help=(
            "Bypass BS-RoFormer and transcribe the original input as the other "
            "stem. Implies --only-other; other-stem instrument and gate "
            "options still apply."
        ),
    )
    parser.add_argument(
        "--mp3-out",
        action="store_true",
        help="Render and mix an MP3 in addition to the default final MIDI.",
    )
    parser.add_argument(
        "--no-file-output",
        dest="publish_files",
        action="store_false",
        help="Run all MuScriptor and MIDI stages in memory without publishing files",
    )
    parser.set_defaults(publish_files=True)
    parser.add_argument(
        "--clean-midi",
        dest="midi_cleanup",
        action="store_true",
        help=(
            "Enable destructive MIDI cleanup: interval normalization, "
            "quiet-onset filtering, duration clamping, and unsupported-retrigger "
            "cleanup. Onset-burst filtering is currently disabled. On by "
            "default; kept so existing commands and scripts still parse."
        ),
    )
    parser.add_argument(
        "--no-clean-midi",
        dest="midi_cleanup",
        action="store_false",
        help="Publish faithful MIDI instead of the destructive cleanup output",
    )
    parser.set_defaults(midi_cleanup=True)
    parser.add_argument(
        "--silence-threshold-dbfs",
        type=float,
        default=-50.0,
        help=(
            "Remove note onsets whose source-audio peak stays below this "
            "level near the onset (default: -50 dBFS)."
        ),
    )
    parser.add_argument(
        "--mp3-bitrate",
        type=int,
        choices=(32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
        default=128,
    )
    parser.add_argument("--mix-jobs", type=int, choices=range(1, 7), default=4)
    parser.add_argument(
        "--preprocess-jobs",
        type=int,
        choices=range(1, 7),
        default=4,
        help=(
            "Number of CPU workers for per-stem gating and presence analysis "
            "(default: 4)."
        ),
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        choices=range(1, 7),
        default=1,
        help="Number of transcription workers (default: 1).",
    )
    parser.add_argument(
        "--host-memory-budget-mib",
        type=int,
        default=4096,
        metavar="MIB",
        help=(
            "Process-global budget for decoded and working stem audio "
            "(default: 4096 MiB). Oversized stems run alone."
        ),
    )
    parser.add_argument(
        "--gpus",
        default="",
        metavar="IDS",
        help=(
            "Comma-separated GPU IDs for transcription workers (for example "
            "0,1); workers are assigned round-robin via CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument(
        "--debug-capture",
        default="",
        metavar="MEMBER[,MEMBER...]",
        help=(
            "Record extra evidence: overlap-probe (bounded reference/candidate "
            "token provenance), trace (generated-row logits), trace-hidden "
            "(logits plus final hidden states), or all. Every member only "
            "records — none changes transcription or writes durable state"
        ),
    )
    parser.add_argument(
        "--export-activations",
        action="store_true",
        help=(
            "Collect final hidden states and per-token logit distributions "
            "into the run directory, for training another model. Unlike "
            "--debug-capture this writes durable output, and it runs at full "
            "speed: the reduction happens inside the CUDA Graph capture"
        ),
    )
    parser.add_argument(
        "--gpu-telemetry-interval-ms",
        type=int,
        default=0,
        metavar="MS",
        help="Sample selected NVIDIA GPUs with NVML; 0 disables (default)",
    )
    parser.add_argument(
        "--decode-transient-per-row-mib",
        type=int,
        default=None,
        metavar="MIB",
        help=(
            "Seed the per-row decode transient before anything has measured it "
            "(default: the value measured at production width on the reference "
            "host). It is a high-water seed, so a device that needs more still "
            "gets more. 0 restores the pre-2026-08-26 behaviour of opening the "
            "first arena against a transient of zero, which spills on Windows "
            "and OOMs on Linux."
        ),
    )
    args = parser.parse_args()
    if args.decode_transient_per_row_mib is not None:
        if args.decode_transient_per_row_mib < 0:
            parser.error("--decode-transient-per-row-mib cannot be negative")
        from resonforge.scheduler.device.block_pool import (
            set_bootstrap_transient_per_row_bytes,
        )

        set_bootstrap_transient_per_row_bytes(
            args.decode_transient_per_row_mib * 1024**2
        )
    try:
        configs = build_pipeline_configs(args)
    except ConfigurationError as error:
        parser.error(str(error))
    # `--song-jobs` is now only the service's concurrency; the device divisor
    # comes from the service at dispatch, not from any request (D1-1).
    max_concurrent_jobs = min(args.song_jobs, len(configs))
    return CliInvocation(
        configs=configs,
        max_concurrent_jobs=max_concurrent_jobs,
    )


def _format_performance(metrics: PerformanceMetrics) -> str:
    return (
        f"{metrics.notes_per_second:.2f} notes/s, "
        f"{metrics.generated_tokens_per_second:.2f} tokens/s, "
        f"{metrics.processing_seconds_per_audio_minute:.2f} s/audio-min, "
        f"{metrics.audio_seconds_per_wall_second:.2f}x realtime"
    )


LOGGER = logging.getLogger(__name__)


def _enable_stack_dump_signal() -> None:
    """Let `kill -USR1 <pid>` print every thread's Python stack.

    A wedged run is diagnosed by *where* its threads are parked, and nothing
    else recovers that: the waterfall stops at the last completed span, and
    per-thread kernel state distinguishes wedged from slow without naming a
    single frame. Installing the handler changes nothing until the signal
    arrives. POSIX only -- Windows has no SIGUSR1.
    """
    handler = getattr(signal, "SIGUSR1", None)
    if handler is None:
        return
    try:
        # `chain=True` would also run SIGUSR1's previous handler, and its
        # default action is to terminate -- the first dump taken this way
        # printed the stacks and then killed the run it was diagnosing.
        faulthandler.register(handler, all_threads=True, chain=False)
    # `io.UnsupportedOperation` is the one this was found through and it is a
    # subclass of both `OSError` and `ValueError`, so naming it adds an import
    # and no coverage.
    except (OSError, ValueError, RuntimeError) as error:
        # **Best-effort, and it was not.** `faulthandler` writes to the real
        # `stderr` file descriptor, so a replaced stream without `fileno`
        # raises here -- and this runs on the first line of `main`, so the
        # whole CLI died before parsing an argument. Found on BAIR 2026-08-27,
        # where every Linux run under a captured stderr failed this way and
        # Windows never could, having no SIGUSR1 to register.
        #
        # A diagnostic that cannot be installed must not take the process with
        # it. Reported rather than swallowed: losing `kill -USR1` is the
        # difference between naming where a wedged run is parked and guessing.
        LOGGER.warning("thread stack dumps unavailable: %s", error)


def main() -> int:
    _enable_stack_dump_signal()
    invocation = parse_args()
    gpu_error = unavailable_gpu_error(invocation.configs)
    if gpu_error is not None:
        print(f"error: {gpu_error}", file=sys.stderr)
        return 2
    started = time.monotonic()
    with InProcessPipelineService(
        max_concurrent_jobs=invocation.max_concurrent_jobs
    ) as service:
        handles = [
            service.submit(JobRequest(config=config))
            for config in invocation.configs
        ]
        previous_handlers: dict[signal.Signals, object] = {}

        def request_cancel(_signum: int, _frame: object) -> None:
            for handle in handles:
                handle.cancel()

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_cancel)
        try:
            results = [handle.result() for handle in handles]
        finally:
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
    for result in results:
        if result.error is not None:
            print(
                f"run {result.session_id} failed: {result.error}",
                file=sys.stderr,
            )
            # A failure that escaped _run_pipeline's own handler (e.g. from
            # its finally) is recorded nowhere else -- the run's log already
            # closed. This is its only surface; keep it.
            trace = getattr(result.failure, "traceback_text", "")
            if trace:
                print(trace, file=sys.stderr)
    batch_performance = aggregate_performance(
        (result.performance for result in results),
        wall_seconds=time.monotonic() - started,
    )
    if batch_performance is not None:
        label = "Batch throughput" if len(results) > 1 else "Pipeline throughput"
        print(f"{label}: {_format_performance(batch_performance)}")
    return next(
        (result.exit_code for result in results if result.exit_code != 0),
        0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
