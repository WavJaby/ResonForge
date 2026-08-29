"""Typed scheduler telemetry schema."""

from __future__ import annotations

from typing import Any

CORE_TELEMETRY_FIELDS = (
    "sequence",
    "priority",
    "queue_wait_seconds",
    "run_seconds",
    "succeeded",
    "model_loaded",
    "pipeline_session_id",
    "batch_size",
)

TELEMETRY_CAPABILITY_FIELDS: dict[str, tuple[str, ...]] = {
    "generation": ("generation_steps", "generated_token_count", "termination_reason"),
    "persistent_sessions": (
        "hot_replacements",
        "resident_checkpoints",
        "resident_resumes",
        "resident_discards",
    ),
    "phase_widths": (
        "condition_batches_by_width",
        "prefill_batches_by_width",
        "first_token_batches_by_width",
        "active_steps_by_width",
    ),
    "phase_timing": (
        "scheduler_decision_cpu_us",
        "scheduler_decision_phase_us",
        "scheduler_action_wall_us",
        "scheduler_boundary_gap_us",
        "scheduler_boundary_gap_count",
        "phase_gpu_ms",
    ),
    "cuda_graph": (
        "cuda_graph_captures",
        "cuda_graph_replays",
        "cuda_graph_evictions",
        "cuda_graph_variants",
        "cuda_graph_cache_pool_bytes_peak",
    ),
    "scheduler_state": (
        "scheduler_observations",
        "scheduler_state_histograms",
        "scheduler_gauges",
    ),
    "generation_guard": (
        "guard_action",
        "guard_warning_findings",
        "guard_critical_findings",
        "guard_reasons",
    ),
    "model_trace_context": ("trace_context",),
}


def telemetry_capabilities(jobs: list[dict[str, Any]]) -> list[str]:
    """Report metric families that contain observed data in this run."""
    return [
        capability
        for capability, names in TELEMETRY_CAPABILITY_FIELDS.items()
        if any(
            job.get(name) not in (None, False, 0, "", {}, [], ())
            for job in jobs
            for name in names
        )
    ]
