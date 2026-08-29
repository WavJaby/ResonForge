"""Build validated pipeline configurations from any transport's raw input.

Every rule here is a domain rule, not a parsing rule: which stem names exist,
which options contradict each other, which ranges are meaningful. It lived in
`cli.py`'s `parse_args()`, where no non-CLI caller could reach it — an HTTP
adapter would have had to duplicate all of it or accept invalid configurations
(D1-2). The CLI now parses, calls this, and turns `ConfigurationError` into
`parser.error`.

Raw input arrives as strings because that is what both transports have: argparse
produces them and a JSON body carries them. Normalisation (`vocal` -> `vocals`,
comma lists to frozensets) is therefore part of the job, not a precondition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from resonforge.pipeline.types import SKIPPABLE_STEMS, PipelineConfig
from resonforge.transcribers.registry import parse_transcriber_assignments

# `band` is the aggregate itself, so it can be skipped but never combined.
COMBINABLE_STEMS = SKIPPABLE_STEMS - {"band"}
MINIMUM_TELEMETRY_INTERVAL_MS = 100
MINIMUM_HOST_MEMORY_BUDGET_MIB = 128

# Flags the service carries to the backend without interpreting. Named here
# rather than derived, because "every field starting with muscriptor_" would
# silently change meaning the day a second backend arrives.
BACKEND_OPTION_FIELDS = (
    "muscriptor_runtime",
    "muscriptor_prefill_runtime",
    "muscriptor_anomaly_detection",
    "muscriptor_recovery_model",
    "muscriptor_silence_split",
    "muscriptor_overlap_detection",
    "muscriptor_recovery",
    "muscriptor_fresh_reanchor",
)


# One named set replaces four separate diagnostic flags (T15). Membership is
# deliberately narrow: a member may only record. `capacity-probe` is excluded
# because it forces a width *and persists the result to the calibration store*,
# so `--debug-capture all` would silently rewrite learned profiles; the reviewed
# path for that is `capacity_profiles.py --record-product-verification`.
# `waterfall` is excluded because the global waterfall database already records
# every span, and rendering per run adds wall time to whatever is being measured
# (the reason stage 1.5c/d removed it).
DEBUG_CAPTURE_MEMBERS = {
    "overlap-probe",
    "trace",
    "trace-hidden",
}
# Graph capture replays a fixed buffer set, so there is no per-step tensor to
# read; the two are mutually exclusive by construction rather than by policy.
TRACE_MEMBERS = {"trace", "trace-hidden"}


class ConfigurationError(ValueError):
    """A request that parsed but does not describe a runnable pipeline."""


def _normalize_stems(value: str | Sequence[str]) -> set[str]:
    """Split a comma list and fold the singular alias callers actually type."""
    parts = value.split(",") if isinstance(value, str) else value
    return {
        "vocals" if stem.strip().lower() == "vocal" else stem.strip().lower()
        for stem in parts
        if stem.strip()
    }


def _parse_gpu_ids(value: str | Sequence[int]) -> tuple[int, ...]:
    if not isinstance(value, str):
        gpus = tuple(int(gpu) for gpu in value)
    else:
        try:
            gpus = tuple(
                int(gpu.strip()) for gpu in value.split(",") if gpu.strip()
            )
        except ValueError:
            raise ConfigurationError(
                "--gpus must be a comma-separated list of non-negative integers"
            ) from None
    if any(gpu < 0 for gpu in gpus):
        raise ConfigurationError(
            "--gpus must be a comma-separated list of non-negative integers"
        )
    if len(set(gpus)) != len(gpus):
        raise ConfigurationError("--gpus must not contain duplicate GPU IDs")
    return gpus


def build_pipeline_configs(source: Any) -> tuple[PipelineConfig, ...]:
    """Validate and normalise one request into one config per input file.

    `source` is any attribute bag — an argparse namespace today, a decoded API
    body later. Raises `ConfigurationError` with a caller-ready message.
    """
    values = {
        field: getattr(source, field)
        for field in dir(source)
        if not field.startswith("_")
    }

    input_files = tuple(Path(path) for path in values.get("input_files") or ())
    if not input_files:
        raise ConfigurationError("at least one input file is required")

    interval = int(values.get("gpu_telemetry_interval_ms", 0))
    if interval != 0 and interval < MINIMUM_TELEMETRY_INTERVAL_MS:
        raise ConfigurationError(
            f"--gpu-telemetry-interval-ms must be 0 or at least "
            f"{MINIMUM_TELEMETRY_INTERVAL_MS}"
        )
    if int(values.get("host_memory_budget_mib", 0)) < MINIMUM_HOST_MEMORY_BUDGET_MIB:
        raise ConfigurationError(
            f"--host-memory-budget-mib must be at least "
            f"{MINIMUM_HOST_MEMORY_BUDGET_MIB}"
        )
    if int(values.get("bs_sample_rate", 0)) <= 0:
        raise ConfigurationError("--bs-sample-rate must be positive")
    if float(values.get("silence_threshold_dbfs", 0.0)) > 0:
        raise ConfigurationError("--silence-threshold-dbfs must be 0 or lower")

    try:
        assignments = parse_transcriber_assignments(values.get("transcribers", ""))
    except ValueError as error:
        raise ConfigurationError(f"--transcribers {error}") from error

    only_other = bool(values.get("only_other", False))
    if values.get("treat_original_as_other"):
        only_other = True
    publish_files = bool(values.get("publish_files", True))
    if not publish_files and values.get("mp3_out"):
        raise ConfigurationError("--mp3-out cannot be combined with --no-file-output")

    debug_capture = {
        member.strip().lower()
        for member in str(values.get("debug_capture", "") or "").split(",")
        if member.strip()
    }
    if "all" in debug_capture:
        debug_capture = set(DEBUG_CAPTURE_MEMBERS)
    unsupported_capture = debug_capture - DEBUG_CAPTURE_MEMBERS
    if unsupported_capture:
        raise ConfigurationError(
            "--debug-capture only accepts all,"
            + ",".join(sorted(DEBUG_CAPTURE_MEMBERS))
            + "; unsupported: "
            + ",".join(sorted(unsupported_capture))
        )
    if debug_capture & TRACE_MEMBERS and (
        values.get("muscriptor_runtime") == "cuda-graphs"
    ):
        # `all` is not narrowed to fit: silently dropping a member the caller
        # asked for is how a debug capture ends up missing the evidence it was
        # turned on to collect.
        raise ConfigurationError(
            "--debug-capture trace cannot be combined with "
            "--muscriptor-runtime cuda-graphs; graph replay has no per-step "
            "tensor to record. Add --muscriptor-runtime cuda-eager, which "
            "keeps the production attention kernel, or torch-eager, or ask "
            "for overlap-probe instead of trace/trace-hidden/all"
        )

    skip_stems = _normalize_stems(values.get("skip_stems", ""))
    combine_band = _normalize_stems(values.get("combine_band", ""))

    unsupported_combine = combine_band - COMBINABLE_STEMS
    if unsupported_combine:
        raise ConfigurationError(
            "--combine-band only accepts bass,drums,guitar,piano,other,vocal; "
            "unsupported: " + ",".join(sorted(unsupported_combine))
        )
    if combine_band and len(combine_band) < 2:
        raise ConfigurationError("--combine-band requires at least two different stems")
    unsupported_skip = skip_stems - SKIPPABLE_STEMS
    if unsupported_skip:
        raise ConfigurationError(
            "--skip only accepts bass,drums,guitar,piano,other,vocal,band; "
            "unsupported: " + ",".join(sorted(unsupported_skip))
        )
    if only_other and "other" in skip_stems:
        raise ConfigurationError("--only-other and --skip other cannot be used together")
    if only_other and combine_band:
        raise ConfigurationError("--only-other and --combine-band cannot be used together")
    if "band" in skip_stems and not combine_band:
        raise ConfigurationError("--skip band requires --combine-band")
    skipped_band_parts = skip_stems & combine_band
    if combine_band and skipped_band_parts:
        raise ConfigurationError(
            "--combine-band aggregates these stems, so they cannot also be "
            "skipped: " + ",".join(sorted(skipped_band_parts))
        )
    unused_overrides = dict(assignments.overrides).keys() & combine_band
    if unused_overrides:
        raise ConfigurationError(
            "--combine-band aggregates these stems, so they cannot have "
            "--transcribers overrides: " + ",".join(sorted(unused_overrides))
        )

    resolved = dict(values)
    resolved.update(
        {
            "input_file": input_files[0],
            "transcribers": assignments.overrides,
            "default_transcriber": assignments.default,
            "only_other": only_other,
            "skip_stems": frozenset(skip_stems),
            "combine_band": frozenset(combine_band),
            "gpus": _parse_gpu_ids(values.get("gpus", ())),
            "debug_capture": frozenset(debug_capture),
            "backend_options": {
                field: values[field]
                for field in BACKEND_OPTION_FIELDS
                if field in values
            },
        }
    )

    class _Resolved:
        pass

    holder = _Resolved()
    for name, value in resolved.items():
        setattr(holder, name, value)
    base = PipelineConfig.from_namespace(holder)
    return tuple(
        replace(base, input_file=input_file) for input_file in input_files
    )


def unavailable_gpu_error(configs: Sequence[PipelineConfig]) -> str | None:
    """Reject GPU IDs this host does not have.

    Device availability is a service fact, not a transport fact (D1-4), so the
    torch import belongs here rather than in `cli.py`. Deferred because a caller
    that requested no GPUs should not pay for loading torch.
    """
    requested = sorted({gpu for config in configs for gpu in config.gpus})
    if not requested:
        return None
    import torch

    available = torch.cuda.device_count()
    invalid = [gpu for gpu in requested if gpu >= available]
    if not invalid:
        return None
    available_ids = ",".join(str(gpu) for gpu in range(available)) or "none"
    return (
        "--gpus contains unavailable IDs: "
        f"{','.join(map(str, invalid))}; available GPU IDs: {available_ids}"
    )
