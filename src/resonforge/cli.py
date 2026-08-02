"""Command-line interface for the ResonForge pipeline."""

import argparse
from pathlib import Path

from resonforge.adaptive_volume_gate import GATE_PRESETS
from resonforge.pipeline import run_pipeline
from resonforge.pipeline.transcription import OTHER_INSTRUMENTS
from resonforge.pipeline.types import SKIPPABLE_STEMS, PipelineConfig
from resonforge.transcribers.registry import (
    DEFAULT_TRANSCRIBER,
    available_specs,
    parse_transcriber_assignments,
)


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        type=Path,
        help=(
            "Input audio file; non-WAV cache misses are converted temporarily "
            "to WAV under input/tmp/."
        ),
    )
    parser.add_argument(
        "--transcribers",
        default="",
        metavar="STEM:BACKEND[.MODEL],...",
        help=(
            "per-stem transcribers, for example "
            "drums:mt3.yptf_moe_multi,piano:basic-pitch,"
            "bass:muscriptor.small; "
            f"omitted stems use {DEFAULT_TRANSCRIBER} unless default:SPEC is "
            f"set; choices: {', '.join(available_specs())}"
        ),
    )
    parser.add_argument(
        "--bs-sample-rate",
        "--separation-sample-rate",
        dest="bs_sample_rate",
        type=int,
        default=44_100,
        metavar="HZ",
        help=(
            "sample rate used for the stereo WAV passed to BS-RoFormer (default: 44100)"
        ),
    )
    parser.add_argument("--cfg-coef", type=float, default=1.0)
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="MuScriptor beam width (1 = greedy; 2 or higher = beam search)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "MuScriptor 5-second chunks decoded together (default: 1). "
            "Values above 1 automatically disable prelude forcing."
        ),
    )
    parser.add_argument("--other-instruments", default=OTHER_INSTRUMENTS)
    parser.add_argument(
        "--adaptive-gate-stems",
        dest="adaptive_gate_presets",
        default="",
        metavar="STEM[:PRESET],...",
        help=(
            "per-stem adaptive gate presets; omitted presets use balanced, "
            "for example bass,guitar:strong,other:extreme"
        ),
    )
    parser.add_argument(
        "--no-gate-presence-filter",
        dest="gate_presence_filter",
        action="store_false",
        help=(
            "Do not analyze and skip globally absent selected stems before "
            "transcription."
        ),
    )
    parser.set_defaults(gate_presence_filter=True)
    gate_threshold = parser.add_mutually_exclusive_group()
    gate_threshold.add_argument(
        "--gate-threshold-offset-db",
        type=float,
        help="override auto threshold relative to the active median",
    )
    parser.add_argument(
        "--hard-zero-threshold-dbfs",
        "--gate-zero-below-dbfs",
        dest="hard_zero_threshold_dbfs",
        type=float,
        default=-80.0,
        help=(
            "For --hard-zero-stems, set quieter 20 ms blocks to exact zero "
            "(default: -80 dBFS)."
        ),
    )
    parser.add_argument(
        "--hard-zero-stems",
        default="",
        metavar="STEMS",
        help=(
            "comma-separated stems whose 20 ms audio blocks below the hard-zero "
            "threshold are set to exact zero before transcription"
        ),
    )
    gate_threshold.add_argument(
        "--gate-threshold-dbfs",
        type=float,
        help="override auto threshold with an absolute dBFS value",
    )
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
    parser.add_argument("--mix-only", action="store_true")
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
    parser.add_argument("--midi-out", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--clean-midi",
        dest="midi_cleanup",
        action="store_true",
        help=(
            "Enable destructive MIDI cleanup, including missing-EOS removal, "
            "interval normalization, quiet-onset and burst filtering, duration "
            "clamping, and unsupported-retrigger cleanup."
        ),
    )
    parser.set_defaults(midi_cleanup=False)
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
        "--gpus",
        default="",
        metavar="IDS",
        help=(
            "Comma-separated GPU IDs for transcription workers (for example "
            "0,1); workers are assigned round-robin via CUDA_VISIBLE_DEVICES."
        ),
    )
    args = parser.parse_args()
    try:
        assignments = parse_transcriber_assignments(args.transcribers)
    except ValueError as error:
        parser.error(f"--transcribers {error}")
    args.transcribers = assignments.overrides
    args.default_transcriber = assignments.default
    if args.treat_original_as_other:
        args.only_other = True
    if args.only_other and args.mix_only:
        parser.error("--only-other and --mix-only cannot be used together")
    if args.beam_size < 1:
        parser.error("--beam-size must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.bs_sample_rate <= 0:
        parser.error("--bs-sample-rate must be positive")
    if args.silence_threshold_dbfs > 0:
        parser.error("--silence-threshold-dbfs must be 0 or lower")
    if args.hard_zero_threshold_dbfs > 0:
        parser.error("--hard-zero-threshold-dbfs must be 0 or lower")
    try:
        args.gpus = tuple(
            int(gpu.strip()) for gpu in args.gpus.split(",") if gpu.strip()
        )
    except ValueError:
        parser.error("--gpus must be a comma-separated list of non-negative integers")
    if any(gpu < 0 for gpu in args.gpus):
        parser.error("--gpus must be a comma-separated list of non-negative integers")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must not contain duplicate GPU IDs")
    requested_skip_stems = {
        "vocals" if stem.strip().lower() == "vocal" else stem.strip().lower()
        for stem in args.skip_stems.split(",")
        if stem.strip()
    }
    requested_combine_stems = {
        "vocals" if stem.strip().lower() == "vocal" else stem.strip().lower()
        for stem in args.combine_band.split(",")
        if stem.strip()
    }
    combinable_stems = SKIPPABLE_STEMS - {"band"}
    unsupported_combine_stems = requested_combine_stems - combinable_stems
    if unsupported_combine_stems:
        parser.error(
            "--combine-band only accepts bass,drums,guitar,piano,other,vocal; "
            "unsupported: " + ",".join(sorted(unsupported_combine_stems))
        )
    if requested_combine_stems and len(requested_combine_stems) < 2:
        parser.error("--combine-band requires at least two different stems")
    unsupported_skip_stems = requested_skip_stems - SKIPPABLE_STEMS
    if unsupported_skip_stems:
        parser.error(
            "--skip only accepts bass,drums,guitar,piano,other,vocal,band; unsupported: "
            + ",".join(sorted(unsupported_skip_stems))
        )
    if args.only_other and "other" in requested_skip_stems:
        parser.error("--only-other and --skip other cannot be used together")
    if args.only_other and args.combine_band:
        parser.error("--only-other and --combine-band cannot be used together")
    adaptive_gate_presets: dict[str, str] = {}
    for entry in args.adaptive_gate_presets.split(","):
        if not entry.strip():
            continue
        parts = entry.strip().split(":", 1)
        if not parts[0].strip() or (len(parts) == 2 and not parts[1].strip()):
            parser.error(
                "--adaptive-gate-stems must use STEM or STEM:PRESET entries "
                "separated by commas"
            )
        stem = parts[0].strip().lower()
        preset = parts[1].strip().lower() if len(parts) == 2 else "balanced"
        if stem == "vocal":
            stem = "vocals"
        if stem in adaptive_gate_presets:
            parser.error("--adaptive-gate-stems contains duplicate stem: " + stem)
        if preset not in GATE_PRESETS:
            parser.error(
                "--adaptive-gate-stems preset must be one of "
                + ",".join(GATE_PRESETS)
                + "; unsupported: "
                + preset
            )
        adaptive_gate_presets[stem] = preset
    requested_adaptive_gate_stems = set(adaptive_gate_presets)
    unsupported_adaptive_gate_stems = requested_adaptive_gate_stems - SKIPPABLE_STEMS
    if unsupported_adaptive_gate_stems:
        parser.error(
            "--adaptive-gate-stems only accepts "
            "bass,drums,guitar,piano,other,vocal,band; "
            "unsupported: " + ",".join(sorted(unsupported_adaptive_gate_stems))
        )
    skipped_gates = requested_adaptive_gate_stems & requested_skip_stems
    if skipped_gates:
        parser.error(
            "cannot gate stems that are skipped: " + ",".join(sorted(skipped_gates))
        )
    if "band" in requested_skip_stems and not args.combine_band:
        parser.error("--skip band requires --combine-band")
    skipped_band_parts = requested_skip_stems & requested_combine_stems
    if requested_combine_stems and skipped_band_parts:
        parser.error(
            "--combine-band aggregates these stems, so they cannot also be "
            "skipped: " + ",".join(sorted(skipped_band_parts))
        )
    args.skip_stems = frozenset(requested_skip_stems)
    unused_transcriber_overrides = (
        dict(assignments.overrides).keys() & requested_combine_stems
    )
    if unused_transcriber_overrides:
        parser.error(
            "--combine-band aggregates these stems, so they cannot have "
            "--transcribers overrides: "
            + ",".join(sorted(unused_transcriber_overrides))
        )
    gated_band_parts = requested_adaptive_gate_stems & requested_combine_stems
    if requested_combine_stems and gated_band_parts:
        parser.error(
            "--combine-band aggregates these stems; use "
            "--adaptive-gate-stems band "
            "instead: " + ",".join(sorted(gated_band_parts))
        )
    if "band" in requested_adaptive_gate_stems and not requested_combine_stems:
        parser.error("--adaptive-gate-stems band requires --combine-band")
    args.adaptive_gate_presets = tuple(sorted(adaptive_gate_presets.items()))
    requested_hard_zero_stems = {
        "vocals" if stem.strip().lower() == "vocal" else stem.strip().lower()
        for stem in args.hard_zero_stems.split(",")
        if stem.strip()
    }
    unsupported_hard_zero_stems = requested_hard_zero_stems - SKIPPABLE_STEMS
    if unsupported_hard_zero_stems:
        parser.error(
            "--hard-zero-stems only accepts "
            "bass,drums,guitar,piano,other,vocal,band; unsupported: "
            + ",".join(sorted(unsupported_hard_zero_stems))
        )
    skipped_hard_zero = requested_hard_zero_stems & requested_skip_stems
    if skipped_hard_zero:
        parser.error(
            "cannot hard-zero stems that are skipped: "
            + ",".join(sorted(skipped_hard_zero))
        )
    hard_zero_band_parts = requested_hard_zero_stems & requested_combine_stems
    if requested_combine_stems and hard_zero_band_parts:
        parser.error(
            "--combine-band aggregates these stems; use --hard-zero-stems "
            "band instead: " + ",".join(sorted(hard_zero_band_parts))
        )
    if "band" in requested_hard_zero_stems and not args.combine_band:
        parser.error("--hard-zero-stems band requires --combine-band")
    args.hard_zero_stems = frozenset(requested_hard_zero_stems)
    args.combine_band = frozenset(requested_combine_stems)
    return PipelineConfig.from_namespace(args)


def main() -> int:
    return run_pipeline(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
