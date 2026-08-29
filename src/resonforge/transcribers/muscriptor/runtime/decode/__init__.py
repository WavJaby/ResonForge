"""Running installed rows forward, and the memory that costs.

Two execution lines share this package -- a captured CUDA graph replayed a
quantum at a time (`graphs`) and the eager step it replaces. They differ in
memory as well as in speed, and **neither difference is priced here.**

`memory.py` used to state decode's device demand from declared shapes: a
per-row transient and a flat figure for the CUDA graph private pool. It was
deleted on 2026-08-27, uncalled by anything but its own tests, and both halves
are better answered elsewhere:

* the **transient** is measured by `scheduler.device.block_pool` against the row count
  it is spent at (`transient_bytes_for`). A computed per-row figure was tried
  and is 6x wrong in production -- fitted on a bench that holds every row at
  full length, spent on a workload whose rows are short and staggered
  (`docs/vram-accounting.md` 5b-3).
* the **graph private pool** needs no reserve at all. Its bytes are reserved
  from the driver through the caching allocator, so `mem_get_info` free has
  already excluded them by the time the pool view reads it. A flat 58 MiB
  constant here was contradicted by a measured 72 MiB and read by nothing.

What survives is the *distinction*: an eager line captures nothing and pays no
pool, which is the one place the two lines differ in memory rather than speed.
`cuda_graph_cache_pool_bytes_peak` records what a run actually held.

**Paging is not here.** KV storage, addressing and the attention kernels live

KV storage, addressing and the attention kernels live
in the MuScriptor fork (`muscriptor/modules/paged_kv.py`,
`attention_backends.py`), because a page's geometry is the *model's* business
while when to capture, how wide, and for which quantum are the *scheduler's*.
That boundary is also a repository boundary, which is why the two device-memory
ledgers -- `scheduler.device.block_pool` here, `KVBlockPool` there -- are separate.
"""
