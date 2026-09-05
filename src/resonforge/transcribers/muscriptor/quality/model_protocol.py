"""What the quality layer needs from a model, stated as structural types.

Overlap verification, recovery, the guards and chunk quality all sit *above*
the model and reach into it for very little: how a token becomes an `Event`,
and what one token is worth in seconds. Naming that surface is what lets those
modules leave this package without dragging the model behind them, and what
stops the next one reaching for a member nobody declared.

Structural, not nominal -- `MT3Tokenizer` satisfies this without importing it,
so nothing in the model has to know the quality layer exists. Underscore names
are here because that is what the tokenizer calls them: a protocol records a
surface as it is, it does not rename it.

Grows one member at a time, as each caller is re-signed. Anything listed here
has a live user; a member with none does not belong in a contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import torch
from muscriptor.modules.conditioners import ConditioningAttributes
from muscriptor.tokenizer.notes import Event, NoteEvent


class TokenizerProtocol(Protocol):
    """The token/time vocabulary the quality layer decodes through."""

    #: token id -> event. Indexed by id, never iterated whole.
    _vocab: Sequence[Event]
    #: shift ticks per second -- the only token <-> absolute time conversion.
    frame_rate: int
    eos_id: int

    def tie_section_token_ids(
        self,
        open_note_keys: Iterable[tuple[int, int]],
    ) -> list[int]: ...

    def overlap_prompt_token_ids(
        self,
        open_note_keys: Iterable[tuple[int, int]],
        note_events: Iterable[NoteEvent],
        seek_time: float,
    ) -> list[int]: ...


class ModelProtocol(Protocol):
    """The model surface the chunk-scheduling loop needs beyond the tokenizer.

    Three members and a name. Generation itself is NOT here: the loop yields a
    `GenerationRequest` and the caller runs it, which is why the scheduler can
    own execution without this contract knowing it exists.

    """

    _tokenizer: TokenizerProtocol
    _device: torch.device
    #: labels which model produced a result. Read with a default -- a model
    #: that does not set one is legal, so this is not a required capability.
    _model_name: str

    def _build_conditions(
        self,
        wav: torch.Tensor,
        instrument_group: str | None = None,
    ) -> list[ConditioningAttributes]: ...
