"""MuScriptor transcription backend."""

from typing import Any

__all__ = ["output_path", "output_prefix", "transcribe"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import transcription

        return getattr(transcription, name)
    raise AttributeError(name)
