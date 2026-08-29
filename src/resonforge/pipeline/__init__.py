"""Public API for the phase-oriented audio pipeline package."""

from .memory_postprocessing import postprocess_in_memory
from .orchestrator import (
    initialize_run,
    make_band_task,
    make_run_prefix,
    prepare_pipeline_context,
    prepare_separated_stems,
    run_pipeline,
    select_stems,
    transcribe_stems,
)
from .output_production import STEM_GAINS_DB, produce_outputs
from .preprocessing import preprocess_stems
from .types import (
    PipelineConfig,
    PipelineContext,
    PipelineOutputs,
    PipelineRunOutcome,
)

__all__ = [
    "PipelineConfig",
    "PipelineContext",
    "PipelineOutputs",
    "PipelineRunOutcome",
    "STEM_GAINS_DB",
    "initialize_run",
    "make_band_task",
    "make_run_prefix",
    "postprocess_in_memory",
    "prepare_pipeline_context",
    "prepare_separated_stems",
    "preprocess_stems",
    "produce_outputs",
    "run_pipeline",
    "select_stems",
    "transcribe_stems",
]
