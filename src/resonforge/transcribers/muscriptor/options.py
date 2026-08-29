"""Backend options for the MuScriptor transcriber."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from resonforge.transcribers.base import BackendOptions


@dataclass(frozen=True)
class MuscriptorOptions(BackendOptions):
    """Everything this backend reads from the caller's configuration.

    A dozen fields, against the roughly thirty on `PipelineConfig` that used to
    be handed over whole. Eight are MuScriptor's own; `publish_files` and
    `debug_capture` are service policy the adapter must honour, carried here
    because every path that would act on them already holds this object.
    """

    batch_size: int = 0
    muscriptor_runtime: str = "cuda-graphs"
    muscriptor_prefill_runtime: str = "cuda-graphs"
    muscriptor_anomaly_detection: bool = True
    muscriptor_overlap_detection: bool = True
    muscriptor_recovery: bool = True
    muscriptor_recovery_model: str = "medium"
    # Off by default: the last-resort unprompted re-anchor costs a full chunk
    # on the larger model and has never been shown to improve output. The
    # ladder above it -- primary, A/B candidates, safe-frontier prefix
    # retention -- is unaffected. MuScriptor's own default stays on; this is
    # the product's choice, made explicitly at every call site.
    muscriptor_fresh_reanchor: bool = False
    muscriptor_silence_split: bool = True
    publish_files: bool = True
    debug_capture: frozenset[str] = frozenset()
    # Durable activation export for training data, deliberately not a
    # `debug_capture` member: that one promises to write no durable state.
    export_activations: bool = False

    @classmethod
    def from_source(cls, source: Any) -> MuscriptorOptions:
        """Read the declared fields off any attribute source.

        Duck-typed on purpose: the source may be an argparse namespace, a
        pipeline config, or a decoded API request, and none of those types may
        be imported here without pointing the dependency back at the service.
        """
        names = {option.name for option in fields(cls)}
        values = {name: getattr(source, name) for name in names if hasattr(source, name)}
        # Service-level fields (`batch_size`, `publish_files`) are attributes;
        # backend-specific ones ride in the opaque mapping the service never
        # interprets. Reading both here is what lets the service stay ignorant.
        carried = getattr(source, "backend_options", None) or {}
        values.update({k: v for k, v in carried.items() if k in names})
        return cls(**values)
