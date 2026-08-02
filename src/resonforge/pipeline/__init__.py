"""Public API for the phase-oriented audio pipeline package."""

from .midi_postprocessing import postprocess_midis
from .orchestrator import (
    combine_band_stems,
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
from .preprocessing import (
    apply_adaptive_gates,
    filter_absent_stems,
    gated_audio_path,
    hard_zero_audio_path,
    preprocess_stems,
)
from .process_runner import Cancelled, ProcessRunner
from .types import PipelineConfig, PipelineContext, PipelineOutputs, StemTask

__all__ = [
    "Cancelled",
    "PipelineConfig",
    "PipelineContext",
    "PipelineOutputs",
    "ProcessRunner",
    "STEM_GAINS_DB",
    "StemTask",
    "apply_adaptive_gates",
    "combine_band_stems",
    "filter_absent_stems",
    "gated_audio_path",
    "hard_zero_audio_path",
    "initialize_run",
    "make_band_task",
    "make_run_prefix",
    "postprocess_midis",
    "prepare_pipeline_context",
    "prepare_separated_stems",
    "preprocess_stems",
    "produce_outputs",
    "run_pipeline",
    "select_stems",
    "transcribe_stems",
]
