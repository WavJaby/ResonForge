"""Device resource accounting: what one GPU holds, who owns it, and what is left.

Five modules, one subject. They were flat beside the scheduler, which read as five peers of `model_workers` rather than one accounting layer under it.

  block_pool  -- device BYTES. The one comparison admission makes (R1's first conservation law).
  ledger      -- how one device reading is divided between co-resident lanes, by role.
  lease       -- cross-process ownership of a physical card.
  allocator   -- which device a pipeline session gets.
  environment -- what this host's devices are.

Nothing here imports the scheduler -- true today (only `lease` reaches outside, to `runtime.paths`), and ARGUED, not gated:
`import_graph.py` resolves to top-level packages only, so `scheduler.device` is not a node it can be given a `--forbid` edge for.
Gating it means teaching that tool subpackage granularity. Until then the direction is held by hand.
"""
