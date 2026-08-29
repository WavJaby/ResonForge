"""The two liveness models, kept together because the pair is the contract.

  scheduler -- `enabled_actions` over the PRODUCTION state. What actually runs.
  model     -- a bounded CPU model of the same transitions, exhaustively searched by `test_bounded_reachable_states_are_deadlock_free`.

They are separate files and one obligation: the proof is only about the production scheduler while the two agree.
Since paging G, `_pool_admits` in the model is the same single comparison as `PoolView.admits`, which is what makes the device-byte half of S1 gated rather than argued.

! the model has NO page supply -- `ModelRunState` carries `arena_blocks` and a `resource_blocked` flag, so a state with free slots and zero pages cannot be expressed in it, let alone reached.
  A deadlock happened on that excluded law on 2026-08-26; treat any new admission, pause or bundle path as unproved against pages.
"""
