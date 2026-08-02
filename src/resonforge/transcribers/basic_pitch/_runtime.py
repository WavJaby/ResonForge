"""Lazy Basic Pitch runtime loading shared by decoder components."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from .inference import predict_model_output


@lru_cache(maxsize=1)
def _basic_pitch_runtime() -> SimpleNamespace:
    """Load Basic Pitch lazily so decoder modules are safe to import."""
    try:
        import pretty_midi
        from basic_pitch.inference import (
            ICASSP_2022_MODEL_PATH,
            Model,
        )
        from basic_pitch.note_creation import (
            get_infered_onsets,
            model_frames_to_time,
            model_output_to_notes,
        )
    except ImportError as error:
        raise RuntimeError(
            "Basic Pitch dependencies are unavailable in this interpreter. "
            "Run the export with "
            "`uv run --project environments/basic-pitch python`."
        ) from error
    return SimpleNamespace(
        pretty_midi=pretty_midi,
        default_model_path=Path(ICASSP_2022_MODEL_PATH),
        Model=Model,
        run_inference=predict_model_output,
        get_infered_onsets=get_infered_onsets,
        model_frames_to_time=model_frames_to_time,
        model_output_to_notes=model_output_to_notes,
    )
