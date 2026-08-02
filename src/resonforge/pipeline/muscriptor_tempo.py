"""MuScriptor beat-grid analysis and cross-stem tempo consensus."""

from __future__ import annotations

import math
from collections import Counter
from statistics import median

from ..tempo_types import MuscriptorTempoReport, TempoResult
from .transcription import split_transcriber_spec, transcriber_for_stem
from .types import PipelineContext, StemTask


def _finite_float(
    value: object,
    default: float | None = None,
) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def analyze_muscriptor_tempo(
    task: StemTask,
    *,
    model: str | None,
) -> MuscriptorTempoReport:
    """Run MuScriptor beat-grid detection for one stem."""
    report: MuscriptorTempoReport = {
        "stem": task.name,
        "method": "muscriptor_detect_grid",
        "backend": "muscriptor",
        "model": model or "default",
        "status": "failed",
    }
    if not task.audio.is_file():
        report["reason"] = "missing_audio_file"
        return report

    try:
        from muscriptor.utils.audio import load_audio
        from muscriptor.utils.beats import BeatDetectionError, detect_grid

        wav = load_audio(task.audio, target_sr=16_000)
        beat_grid = detect_grid(wav, 16_000, checkpoint="final0")
    except FileNotFoundError:
        report["reason"] = "audio_file_not_found"
        return report
    except BeatDetectionError as error:
        report["reason"] = str(error)
        if error.fit is not None:
            report.update(
                {
                    "candidate_bpm": round(error.fit.bpm, 6),
                    "residual_seconds": round(error.fit.residual_seconds, 6),
                    "beat_count": error.fit.beat_count,
                    "outlier_fraction": round(error.fit.outlier_fraction, 4),
                }
            )
        return report
    except Exception as error:  # noqa: BLE001  # pragma: no cover
        report["reason"] = str(error)
        return report

    report.update(
        {
            "residual_seconds": round(beat_grid.residual_seconds, 6),
            "beat_count": beat_grid.beat_count,
            "downbeat_count": beat_grid.downbeat_count,
            "coverage_ratio": round(beat_grid.coverage_ratio, 4),
            "meter_agreement": beat_grid.meter_agreement,
            "outlier_fraction": round(beat_grid.outlier_fraction, 4),
            "status": "ok",
            "bpm": round(beat_grid.bpm, 6),
            "beats_per_bar": beat_grid.beats_per_bar,
            "first_downbeat": round(beat_grid.first_downbeat, 6),
            "bar_offset": round(beat_grid.bar_offset(), 6),
            "bar_seconds": (
                round(beat_grid.bar_seconds, 6)
                if _finite_float(beat_grid.bar_seconds) is not None
                else None
            ),
        }
    )
    return report


def collect_muscriptor_tempo_reports(
    tasks: list[StemTask],
    *,
    context: PipelineContext,
) -> list[MuscriptorTempoReport]:
    """Analyze every selected MuScriptor stem and record its report."""
    reports: list[MuscriptorTempoReport] = []
    for task in tasks:
        backend, model = split_transcriber_spec(
            transcriber_for_stem(context.config, task.name)
        )
        if backend != "muscriptor":
            continue
        report = analyze_muscriptor_tempo(task, model=model)
        context.metadata["stems"].setdefault(task.name, {}).setdefault(
            "transcription",
            {},
        ).setdefault("artifacts", {})["muscriptor_tempo"] = report
        reports.append(report)
    return reports


def _valid_reports(
    reports: list[MuscriptorTempoReport],
) -> list[MuscriptorTempoReport]:
    return [
        report
        for report in reports
        if report.get("status") == "ok"
        and _finite_float(report.get("bpm")) is not None
    ]


def _bar_fields(report: MuscriptorTempoReport, bpm: float) -> dict[str, float | int]:
    meter = report.get("beats_per_bar")
    downbeat = _finite_float(report.get("first_downbeat"))
    if meter is None or downbeat is None:
        return {}
    bar_seconds = meter * 60.0 / bpm
    phase = downbeat % bar_seconds
    return {
        "beats_per_bar": meter,
        "first_downbeat": downbeat,
        "bar_seconds": round(bar_seconds, 6),
        "bar_phase_seconds": round(phase, 6),
        "bar_offset": round((-phase) % bar_seconds, 6),
    }


def _result_from_report(
    report: MuscriptorTempoReport,
    reports: list[MuscriptorTempoReport],
    *,
    method: str,
) -> TempoResult:
    bpm = _finite_float(report.get("bpm")) or _finite_float(
        report.get("candidate_bpm")
    )
    if bpm is None:
        raise ValueError("tempo report has no BPM")
    result: TempoResult = {
        "status": "ok" if report.get("status") == "ok" else "candidate_only",
        "method": method,
        "selected_bpm": round(bpm, 6),
        "selected_source": report["stem"],
        "sources_used": [report["stem"]],
        "all_reports": reports,
    }
    if report.get("status") == "ok":
        result.update(_bar_fields(report, bpm))
    return result


def _cross_stem_result(
    valid: list[MuscriptorTempoReport],
    reports: list[MuscriptorTempoReport],
) -> TempoResult:
    bpm = float(median(float(report["bpm"]) for report in valid))
    meters = [report["beats_per_bar"] for report in valid if report.get("beats_per_bar")]
    result: TempoResult = {
        "status": "ok",
        "method": "muscriptor_cross_stem_median",
        "selected_bpm": round(bpm, 6),
        "selected_source": "three_or_more_stem_consensus",
        "sources_used": [report["stem"] for report in valid],
        "all_reports": reports,
    }
    if not meters:
        return result
    meter = Counter(meters).most_common(1)[0][0]
    bar_seconds = meter * 60.0 / bpm
    candidates = [
        report
        for report in valid
        if report.get("beats_per_bar") == meter
        and _finite_float(report.get("first_downbeat")) is not None
    ]
    if not candidates:
        return result
    phases = [float(report["first_downbeat"]) % bar_seconds for report in candidates]
    phase = min(
        phases,
        key=lambda candidate: sum(
            min(abs(candidate - other), bar_seconds - abs(candidate - other))
            for other in phases
        ),
    )
    result.update(
        {
            "beats_per_bar": meter,
            "first_downbeat": float(candidates[phases.index(phase)]["first_downbeat"]),
            "bar_seconds": round(bar_seconds, 6),
            "bar_phase_seconds": round(phase, 6),
            "bar_offset": round((-phase) % bar_seconds, 6),
        }
    )
    return result


def select_tempo_grid(
    reports: list[MuscriptorTempoReport],
) -> TempoResult | None:
    """Use three-grid consensus, otherwise use the drums tempo estimate."""
    valid = _valid_reports(reports)
    if len(valid) >= 3:
        return _cross_stem_result(valid, reports)
    drums = next(
        (
            report
            for report in reports
            if report["stem"] == "drums"
            and (
                _finite_float(report.get("bpm")) is not None
                or _finite_float(report.get("candidate_bpm")) is not None
            )
        ),
        None,
    )
    if drums is None:
        return None
    return _result_from_report(drums, reports, method="muscriptor_drums_fallback")


def merged_bpm(tempo_result: TempoResult | None) -> float:
    if tempo_result is None:
        return 120.0
    return _finite_float(tempo_result.get("selected_bpm"), 120.0) or 120.0
