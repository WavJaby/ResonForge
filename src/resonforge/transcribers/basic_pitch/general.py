#!/usr/bin/env python
"""Experimental posteriorgram-aware Basic Pitch decoder.

Produces stock and conservative decoder outputs for pitched stems.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ._runtime import _basic_pitch_runtime
from .artifacts import (
    max_polyphony,
    note_count_by_source,
    save_baseline,
    write_csv,
    write_midi,
)
from .candidates import (
    frames_for_ms,
    generate_candidates,
)
from .postprocessing import (
    candidates_to_seconds,
    merge_same_pitch,
    suppress_weak_harmonics,
)

SUPPORTED_STEMS = ("bass", "guitar", "piano")
SKIPPED_STEMS = ("drums", "vocals", "vocal", "other")
__all__ = [
    "GeneralDecoderConfig",
    "build_general_export_command",
    "export_general_midi",
]


@dataclass(frozen=True)
class GeneralDecoderConfig:
    """Public configuration for the experimental decoder."""

    onset_threshold: float = 0.50
    frame_threshold: float = 0.30
    frame_only_threshold: float = 0.45
    onset_min_ms: float = 120.0
    frame_only_min_ms: float = 175.0
    merge_gap_ms: float = 55.0
    energy_tolerance_ms: float = 125.0
    frame_only_min_confidence: float = 0.42
    harmonic_score_ratio: float = 0.72
    harmonic_suppression: bool = True


def infer_stem_name(path: Path) -> str:
    name = path.stem.lower()
    for stem in (*SUPPORTED_STEMS, *SKIPPED_STEMS):
        if name == stem or name.endswith(f"_{stem}") or name.startswith(f"{stem}."):
            return stem
        if f"_{stem}_" in name or f".{stem}." in name:
            return stem
    return "pitched"


def discover_audio(args: Any) -> list[tuple[Path, str]]:
    requested_stems = {
        item.strip().lower() for item in args.stems.split(",") if item.strip()
    }
    unsupported = requested_stems - set(SUPPORTED_STEMS)
    if unsupported:
        raise ValueError(
            "--stems currently supports only bass,guitar,piano; unsupported: "
            + ",".join(sorted(unsupported))
        )

    discovered: list[tuple[Path, str]] = []
    for path in args.audio:
        resolved = path.resolve()
        stem = args.stem or infer_stem_name(resolved)
        if stem in SKIPPED_STEMS:
            print(f"Skipping non-target stem: {resolved.name} ({stem})", flush=True)
            continue
        discovered.append((resolved, stem))

    if args.stem_dir is not None:
        stem_dir = args.stem_dir.resolve()
        for stem in sorted(requested_stems):
            matches = sorted(stem_dir.glob(f"*_{stem}.wav"))
            direct = stem_dir / f"{stem}.wav"
            if direct.is_file():
                matches.append(direct)
            unique_matches = dict.fromkeys(path.resolve() for path in matches)
            for path in unique_matches:
                discovered.append((path, stem))

    unique: dict[Path, str] = {}
    for path, stem in discovered:
        if not path.is_file():
            raise FileNotFoundError(f"audio file not found: {path}")
        unique[path] = stem
    if not unique:
        raise FileNotFoundError("no matching pitched stems were found")
    return list(unique.items())


def process_audio(
    audio_path: Path,
    stem: str,
    output_dir: Path,
    model: object,
    args: Any,
) -> dict[str, object]:
    print(f"\nPredicting {stem:<7} {audio_path.name}", flush=True)
    output = _basic_pitch_runtime().run_inference(audio_path, model)
    prefix = output_dir / f"{stem}.{audio_path.stem}"

    def artifact_path(suffix: str) -> Path:
        return prefix.with_name(prefix.name + suffix)

    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_count: int | None = None
    baseline_path: Path | None = None
    if not args.no_baseline:
        baseline_path = artifact_path(".basic-pitch.baseline.mid")
        baseline_count = save_baseline(
            output,
            baseline_path,
            onset_threshold=args.onset_threshold,
            frame_threshold=args.frame_threshold,
            onset_min_ms=args.onset_min_ms,
        )
        print(f"  baseline -> {baseline_path}", flush=True)

    generated = generate_candidates(
        output,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        frame_only_threshold=args.frame_only_threshold,
        onset_min_frames=frames_for_ms(args.onset_min_ms),
        frame_only_min_frames=frames_for_ms(args.frame_only_min_ms),
        tolerance_frames=frames_for_ms(args.energy_tolerance_ms),
        frame_only_min_confidence=args.frame_only_min_confidence,
    )
    merged = merge_same_pitch(
        generated,
        merge_gap_frames=frames_for_ms(args.merge_gap_ms),
        onset_threshold=args.onset_threshold,
        onset_min_frames=frames_for_ms(args.onset_min_ms),
    )
    if args.no_harmonic_suppression:
        accepted, harmonic_rejected = merged, []
    else:
        accepted, harmonic_rejected = suppress_weak_harmonics(
            merged,
            score_ratio=args.harmonic_score_ratio,
            onset_threshold=args.onset_threshold,
        )
    notes_seconds = candidates_to_seconds(
        accepted,
        int(output["note"].shape[0]),
    )

    midi_path = artifact_path(".basic-pitch.general.mid")
    csv_path = artifact_path(".basic-pitch.general.csv")
    stats_path = artifact_path(".basic-pitch.general.json")
    write_midi(notes_seconds, midi_path, stem)
    write_csv(notes_seconds, csv_path)

    pitches = [candidate.pitch for candidate in accepted]
    stats: dict[str, object] = {
        "audio": str(audio_path),
        "stem": stem,
        "baseline_note_count": baseline_count,
        "raw_candidate_count": len(generated),
        "merged_candidate_count": len(merged),
        "general_note_count": len(accepted),
        "general_notes_by_source": note_count_by_source(accepted),
        "harmonic_rejected_count": len(harmonic_rejected),
        "pitch_min": min(pitches) if pitches else None,
        "pitch_max": max(pitches) if pitches else None,
        "max_polyphony": max_polyphony(notes_seconds),
        "outputs": {
            "baseline_midi": str(baseline_path) if baseline_path else None,
            "general_midi": str(midi_path),
            "notes_csv": str(csv_path),
            "stats_json": str(stats_path),
            "model_output_npz": (
                str(artifact_path(".basic-pitch.model-output.npz"))
                if args.save_model_output
                else None
            ),
        },
        "parameters": {
            "onset_threshold": args.onset_threshold,
            "frame_threshold": args.frame_threshold,
            "frame_only_threshold": args.frame_only_threshold,
            "onset_min_ms": args.onset_min_ms,
            "frame_only_min_ms": args.frame_only_min_ms,
            "merge_gap_ms": args.merge_gap_ms,
            "energy_tolerance_ms": args.energy_tolerance_ms,
            "frame_only_min_confidence": args.frame_only_min_confidence,
            "harmonic_score_ratio": args.harmonic_score_ratio,
            "harmonic_suppression": not args.no_harmonic_suppression,
        },
        "harmonic_rejected": [asdict(candidate) for candidate in harmonic_rejected],
    }
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.save_model_output:
        np.savez_compressed(
            artifact_path(".basic-pitch.model-output.npz"),
            onset=output["onset"],
            note=output["note"],
            contour=output["contour"],
        )

    print(f"  general  -> {midi_path}", flush=True)
    print(
        "  notes: "
        f"baseline={baseline_count if baseline_count is not None else 'disabled'}, "
        f"general={len(accepted)}, "
        f"frame-only={note_count_by_source(accepted)['frame']}, "
        f"harmonics-removed={len(harmonic_rejected)}, "
        f"max-polyphony={stats['max_polyphony']}",
        flush=True,
    )
    return stats


def build_general_export_command(
    basic_pitch_python: Path | str,
    audio_path: Path | str,
    *,
    stem: str,
    output_dir: Path | str,
    config: GeneralDecoderConfig | None = None,
    save_baseline: bool = True,
    save_model_output: bool = False,
    model_path: Path | str | None = None,
) -> list[str]:
    """Build the subprocess command intended for ResonForge's ProcessRunner."""
    normalized_stem = stem.strip().lower()
    if normalized_stem not in SUPPORTED_STEMS:
        raise ValueError(
            "stem must be one of bass,guitar,piano; "
            f"received: {normalized_stem or '<empty>'}"
        )
    decoder_config = config or GeneralDecoderConfig()
    command = [
        os.fspath(Path(basic_pitch_python)),
        "-u",
        os.fspath(Path(__file__).resolve()),
        os.fspath(Path(audio_path)),
        "--stem",
        normalized_stem,
        "--output-dir",
        os.fspath(Path(output_dir)),
        "--onset-threshold",
        str(decoder_config.onset_threshold),
        "--frame-threshold",
        str(decoder_config.frame_threshold),
        "--frame-only-threshold",
        str(decoder_config.frame_only_threshold),
        "--onset-min-ms",
        str(decoder_config.onset_min_ms),
        "--frame-only-min-ms",
        str(decoder_config.frame_only_min_ms),
        "--merge-gap-ms",
        str(decoder_config.merge_gap_ms),
        "--energy-tolerance-ms",
        str(decoder_config.energy_tolerance_ms),
        "--frame-only-min-confidence",
        str(decoder_config.frame_only_min_confidence),
        "--harmonic-score-ratio",
        str(decoder_config.harmonic_score_ratio),
    ]
    if not decoder_config.harmonic_suppression:
        command.append("--no-harmonic-suppression")
    if not save_baseline:
        command.append("--no-baseline")
    if save_model_output:
        command.append("--save-model-output")
    if model_path is not None:
        command.extend(["--model", os.fspath(Path(model_path))])
    return command


def export_general_midi(
    audio_path: Path | str,
    *,
    stem: str,
    output_dir: Path | str,
    config: GeneralDecoderConfig | None = None,
    model_path: Path | str | None = None,
    model: object | None = None,
    save_baseline: bool = True,
    save_model_output: bool = False,
) -> dict[str, object]:
    """Export baseline and general-decoder MIDI for one pitched stem.

    This is the stable public entry point intended for later pipeline
    integration. Importing this module does not import Basic Pitch; the
    dependency is loaded only when this function runs.
    """
    normalized_stem = stem.strip().lower()
    if normalized_stem not in SUPPORTED_STEMS:
        raise ValueError(
            "stem must be one of bass,guitar,piano; "
            f"received: {normalized_stem or '<empty>'}"
        )
    audio = Path(audio_path).resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")
    destination = Path(output_dir).resolve()
    decoder_config = config or GeneralDecoderConfig()
    runtime = _basic_pitch_runtime()
    resolved_model_path = (
        Path(model_path).resolve()
        if model_path is not None
        else runtime.default_model_path
    )
    if model is None:
        if not resolved_model_path.is_file():
            raise FileNotFoundError(
                f"Basic Pitch model not found: {resolved_model_path}"
            )
        model = runtime.Model(resolved_model_path)

    options = SimpleNamespace(
        **asdict(decoder_config),
        no_harmonic_suppression=not decoder_config.harmonic_suppression,
        no_baseline=not save_baseline,
        save_model_output=save_model_output,
    )
    return process_audio(
        audio,
        normalized_stem,
        destination,
        model,
        options,
    )
