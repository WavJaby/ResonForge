"""Basic Pitch decoding and export tools."""

from typing import Any

__all__ = ["GeneralDecoderConfig", "export_general_midi"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import general

        return getattr(general, name)
    raise AttributeError(name)
