"""MuScriptor-specific execution runtime driven by the ResonForge scheduler.

Continuous batching, KV slot residency, CUDA Graph decode, and the exactness
contract that verifies them. Model-architecture specific by construction, which
is why it sits under the backend rather than in `scheduler/`; it imports the
MuScriptor model, never the reverse.
"""
