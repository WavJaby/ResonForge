"""Overlap verification, recovery, guards and chunk quality.

The layer that decides whether a chunk is good and what to do when it is
not. `contract/` is what it shares with the executor and the model --
`generation_batch` the messages it yields, `guard_protocol` the one object
it plants inside the decode loop, `model_protocol` the whole of what it
needs from a `TranscriptionModel`. `policy/` is everything that decides;
`quality_plan` is its door, and `transcription` still reaches past it.

It lives here rather than in `muscriptor` because the scheduler that
executes its requests lives here: nothing in the model runs any of this,
and the model imports none of it. `quality_plan` is what the backend
supplies to `TranscriptionModel.transcribe` to put it in play.
"""
