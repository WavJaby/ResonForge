"""Registry and shared CLI parsing for transcription backends."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from .base import Transcriber

DEFAULT_TRANSCRIBER = "mt3.yptf_moe_multi"
TRANSCRIBABLE_STEMS = frozenset(
    {"bass", "drums", "guitar", "piano", "other", "vocals", "band"}
)
_BACKENDS: dict[str, Transcriber] | None = None


def _load_backends() -> dict[str, Transcriber]:
    """Import backend adapters lazily to keep package imports acyclic."""
    global _BACKENDS
    if _BACKENDS is None:
        from .basic_pitch.transcription import BACKEND as basic_pitch
        from .mt3.transcription import BACKEND as mt3
        from .muscriptor.transcription import BACKEND as muscriptor

        _BACKENDS = {
            backend.name: backend
            for backend in (basic_pitch, mt3, muscriptor)
        }
    return _BACKENDS


class _BackendRegistry(Mapping[str, Transcriber]):
    def __getitem__(self, key: str) -> Transcriber:
        return _load_backends()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(_load_backends())

    def __len__(self) -> int:
        return len(_load_backends())


TRANSCRIBERS: Mapping[str, Transcriber] = _BackendRegistry()


@dataclass(frozen=True)
class TranscriberSelection:
    """Parsed backend plus its optional model."""

    backend: Transcriber
    model: str | None

    @property
    def spec(self) -> str:
        return (
            self.backend.name
            if self.model is None
            else f"{self.backend.name}.{self.model}"
        )


@dataclass(frozen=True)
class TranscriberAssignments:
    """Validated default and per-stem transcriber selections."""

    default: str
    overrides: tuple[tuple[str, str], ...]


def available_specs() -> tuple[str, ...]:

    specs: list[str] = []
    for backend in TRANSCRIBERS.values():
        if backend.models:
            specs.extend(f"{backend.name}.{model}" for model in backend.models)
        else:
            specs.append(backend.name)
    return tuple(sorted(specs))


def parse_transcriber_spec(
    spec: str,
    *,
    stem: str | None = None,
    as_default: bool = False,
) -> TranscriberSelection:
    """Parse and validate one BACKEND[.MODEL] selection."""
    normalized = spec.strip().lower()
    backend_name, separator, model = normalized.partition(".")
    backend = TRANSCRIBERS.get(backend_name)
    if backend is None:
        raise ValueError(
            "transcriber must be one of " + ", ".join(available_specs())
            + f"; unsupported: {normalized}"
        )
    selected_model = model if separator else None
    if backend.models:
        if selected_model not in backend.models:
            raise ValueError(
                f"{backend.name} model must be one of "
                + ",".join(sorted(backend.models))
                + f"; unsupported: {selected_model or '(missing)'}"
            )
    elif selected_model is not None:
        raise ValueError(
            f"{backend.name} does not accept a model; unsupported: {normalized}"
        )
    if as_default and not backend.supports_default:
        raise ValueError(
            f"default:{normalized} is not supported because "
            f"{backend.name} cannot transcribe every stem"
        )
    if (
        stem is not None
        and backend.supported_stems is not None
        and stem not in backend.supported_stems
    ):
        raise ValueError(
            f"{backend.name} only supports "
            + ",".join(sorted(backend.supported_stems))
            + f"; unsupported stem: {stem}"
        )
    return TranscriberSelection(backend, selected_model)


def parse_transcriber_assignments(
    value: str,
    *,
    default: str = DEFAULT_TRANSCRIBER,
) -> TranscriberAssignments:
    """Parse STEM:BACKEND[.MODEL] entries used by the pipeline CLI."""
    overrides: dict[str, str] = {}
    default_spec = default
    default_seen = False
    for raw_entry in value.split(","):
        if not raw_entry.strip():
            continue
        raw_stem, separator, raw_spec = raw_entry.partition(":")
        if not separator or not raw_stem.strip() or not raw_spec.strip():
            raise ValueError(
                "must use STEM:BACKEND[.MODEL] entries separated by commas"
            )
        stem = raw_stem.strip().lower()
        if stem == "vocal":
            stem = "vocals"
        if stem == "default":
            if default_seen:
                raise ValueError("contains duplicate stem: default")
            default_spec = parse_transcriber_spec(
                raw_spec, as_default=True
            ).spec
            default_seen = True
            continue
        if stem not in TRANSCRIBABLE_STEMS:
            raise ValueError(
                "only accepts bass,drums,guitar,piano,other,vocal,band; "
                f"unsupported: {stem}"
            )
        if stem in overrides:
            raise ValueError(f"contains duplicate stem: {stem}")
        overrides[stem] = parse_transcriber_spec(raw_spec, stem=stem).spec
    return TranscriberAssignments(
        default=default_spec,
        overrides=tuple(sorted(overrides.items())),
    )


def get_transcriber(spec: str, *, stem: str | None = None) -> TranscriberSelection:

    return parse_transcriber_spec(spec, stem=stem)
