"""Overlap verification, recovery, guards and chunk quality.

The layer that decides whether a chunk is good and what to do when it is
not. It sits above the model and drives it through three contracts --
`model_protocol` states the whole of what it needs from a
`TranscriptionModel`, `generation_batch` the whole of what it hands back for
someone else to run, and `guard_protocol` the one object it plants inside
that executor's decode loop. Everything else here is policy nobody outside
the package imports.

It lives here rather than in `muscriptor` because the scheduler that
executes its requests lives here: nothing in the model runs any of this,
and the model imports none of it. `quality_plan` is what the backend
supplies to `TranscriptionModel.transcribe` to put it in play.
"""
