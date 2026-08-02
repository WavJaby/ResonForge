"""Reusable audio processing tools for ResonForge.

Public helpers are loaded lazily so isolated backend environments do not need
unrelated pipeline dependencies.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "GATE_PRESETS": (".adaptive_volume_gate", "GATE_PRESETS"),
    "GatePreset": (".adaptive_volume_gate", "GatePreset"),
    "GateResult": (".adaptive_volume_gate", "GateResult"),
    "adaptive_volume_gate": (".adaptive_volume_gate", "adaptive_volume_gate"),
    "get_gate_preset": (".adaptive_volume_gate", "get_gate_preset"),
    "hard_zero_audio_blocks": (
        ".adaptive_volume_gate",
        "hard_zero_audio_blocks",
    ),
    "analyze_stem_presence": (
        ".stem_presence_filter",
        "analyze_stem_presence",
    ),
    "assess_stem_presence": (
        ".stem_presence_filter",
        "assess_stem_presence",
    ),
    "convert_jsonl_to_midi": (".midi.jsonl", "convert_jsonl_to_midi"),
    "export_general_midi": (
        ".transcribers.basic_pitch.general",
        "export_general_midi",
    ),
    "mixdown_midis": (".midi.mixdown", "mixdown_midis"),
    "run_pipeline": (".pipeline", "run_pipeline"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
