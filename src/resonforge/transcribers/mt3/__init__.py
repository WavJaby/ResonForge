"""MT3 transcription backend."""

from typing import Any

__all__ = ["MODELS", "output_path", "transcribe"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import transcription

        return getattr(transcription, name)
    raise AttributeError(name)
