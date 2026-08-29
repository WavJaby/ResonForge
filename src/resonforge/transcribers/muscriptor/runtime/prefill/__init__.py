"""Turning a request and its conditions into a row decode can install.

Prefill owns one device allocation -- the state it writes its KV into -- and
`memory` is both where that allocation happens and where its cost is stated
before it is taken. Nothing here reaches into a session: the dependency runs
one way, `decode` -> `prefill`, and `PreparedGenerationRow` is the handoff.
"""
