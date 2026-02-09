# CPU Memory Leak Analysis (GitHub Issue #28726)

## Summary

vLLM suffers from continuous CPU memory growth when serving multimodal (VLM)
models with prefix caching enabled (the default). The EngineCore subprocess
RSS grows by ~1.5 GB per 1000 requests and never stabilizes, eventually
causing OOM. The issue appeared between v0.11.0 and v0.11.1.

**Root cause**: A reference cycle in `Request` objects prevents Python's
reference counting from freeing them. A GC optimization introduced in v0.11.1
reduces how often the cyclic garbage collector runs, causing these cyclic
`Request` objects (each holding megabytes of multimodal feature data) to
accumulate far faster than the GC can reclaim them.

## The Reference Cycle

In `vllm/v1/request.py`, when prefix caching is enabled, each `Request` binds
itself into a `functools.partial`:

```python
# vllm/v1/request.py:167-170 (main branch)
self.get_hash_new_full_blocks: Callable[[], list[BlockHash]] | None = None
if block_hasher is not None:
    self.get_hash_new_full_blocks = partial(block_hasher, self)  # <-- cycle
    self.block_hashes = self.get_hash_new_full_blocks()
```

This creates a **reference cycle**:

```
Request ──(self.get_hash_new_full_blocks)──> partial object
   ^                                              │
   └──────────(partial stores self as arg)────────┘
```

### Why this matters

Python uses two garbage collection mechanisms:

1. **Reference counting** (immediate): When an object's reference count drops
   to zero, it is freed instantly. This is the fast path.

2. **Cyclic garbage collector** (deferred): Periodically scans for groups of
   objects that reference each other but are unreachable from the rest of the
   program. This is the slow path, and it runs based on heuristic thresholds.

The reference cycle means that when the scheduler finishes a request and does
`del self.requests[request_id]`, the `Request` object's reference count
**does not drop to zero** -- the `partial` still holds a reference. The
`partial`'s count doesn't drop to zero either -- the `Request` still holds
it. Both objects are unreachable from the program, but neither can be freed
by reference counting. They become **cyclic garbage**, waiting for the cyclic
GC to detect and collect them.

### Why it only affects prefix caching

When prefix caching is **disabled**, `block_hasher` is `None`, so the
`partial` is never created. There is no cycle. `Request` objects are freed
immediately by reference counting when the scheduler removes them. This is
why `--no-enable-prefix-caching` prevents the leak.

### Why it only affects multimodal models visibly

Each `Request` object holds a `mm_features: list[MultiModalFeatureSpec]`
field. For vision-language models, this contains the **processed image
feature tensors** -- several megabytes per image. A text-only request has
empty `mm_features` and is only a few kilobytes. When cyclic garbage
accumulates:

- **Text-only**: 100 leaked Request objects ~ a few MB (invisible)
- **VLM with images**: 100 leaked Request objects ~ hundreds of MB to GBs
  (causes OOM)

## Why It Became a Problem in v0.11.1

The reference cycle existed since prefix caching was introduced. In v0.11.0,
it was harmless because the cyclic GC ran frequently enough to clean it up.
Two changes in v0.11.1 broke this equilibrium:

### Change 1: Fewer GC-tracked objects per request (primary cause)

**Commit `acaa2c0a4`** -- *"Reuse empty block lists whenever possible in
KVCacheBlocks to mitigate GC costs"*

This optimization replaced empty `list` objects (`[]`) with empty `tuple`
objects (`()`) in `KVCacheBlocks`. Empty tuples are **not tracked by the
cyclic GC** (CPython optimization), while empty lists are. This means each
request cycle creates fewer GC-tracked objects.

Python's cyclic GC uses a generational scheme with thresholds (default:
`(700, 10, 10)`):

- **Generation 0** collection triggers when 700+ new tracked objects
  accumulate since the last gen-0 collection.
- **Generation 1** triggers every 10 gen-0 collections.
- **Generation 2** triggers every 10 gen-1 collections (every 100 gen-0's).

With fewer tracked objects created per request, it takes longer for the
generation-0 threshold (700 objects) to be reached. This means:
- Gen-0 collections happen less often
- Gen-1 and gen-2 collections happen much less often
- Cyclic garbage from `Request` objects accumulates longer before being swept

In v0.11.0, the extra `list` objects from `KVCacheBlocks` kept the GC
running frequently. Gen-2 collections (which sweep long-lived cyclic
garbage) ran often enough that the leaked `Request` memory stabilized.
In v0.11.1, gen-2 collections became too infrequent, and memory grew
without bound.

### Change 2: Earlier gc.freeze() (secondary contributor)

**Commit `b30372cbd`** -- *"Move gc.freeze logic from EngineCoreProc to
EngineCore for better coverage"*

`gc.freeze()` moves all currently tracked objects into a permanent
generation that the GC never scans. This was moved from the end of
`EngineCoreProc.__init__()` to the end of `EngineCore.__init__()`,
freezing objects earlier. While this doesn't directly prevent collection
of new `Request` objects, the different freeze timing subtly changes the
GC's generation accounting, further reducing the frequency of collections
on unfrozen objects.

## Reproduction Results

Using `Qwen/Qwen2.5-VL-3B-Instruct` with the `lmarena-ai/VisionArena-Chat`
dataset (real user-uploaded images), 1000 prompts per round, prefix caching
enabled:

### main branch (leak present)

```
Round   Reqs     Total(GB)   EC(GB)    EC delta   EC round   Time
----------------------------------------------------------------------
idle    0        3.63        3.63      ---        ---
1       1000     10.97       10.97     +7.33      +7.33      66s
2       2000     14.34       14.34     +10.71     +3.37      57s
3       3000     15.94       15.94     +12.31     +1.60      55s
4       4000     16.91       16.91     +13.28     +0.97      58s
5       5000     17.38       17.38     +13.74     +0.47      58s

EngineCore final RSS: 14.70 GB (started at 2.40 GB)
Growth rate: +1.60 GB/round average -- NEVER STABILIZES
```

### fix-cpu-leak branch (leak fixed)

```
Round   Reqs     Total(GB)   EC(GB)    EC delta   EC round   Time
----------------------------------------------------------------------
idle    0        3.63        3.63      ---        ---
1       1000     9.86        9.86      +6.22      +6.22      67s
2       2000     10.50       10.50     +6.86      +0.64      56s
3       3000     10.55       10.55     +6.91      +0.05      57s
4       4000     10.55       10.55     +6.92      +0.01      56s
5       5000     10.64       10.64     +7.01      +0.09      56s

EngineCore final RSS: 7.51 GB (started at 2.41 GB)
Growth rate: +0.20 GB/round average -- STABLE after round 1
```

The fix reduces EngineCore memory by **half** (7.51 GB vs 14.70 GB) after
5000 multimodal requests.

## The Fix

**Break the reference cycle** by storing `block_hasher` directly without
`partial`, and passing `self` explicitly at call sites:

```python
# BEFORE (creates cycle):
self.get_hash_new_full_blocks = partial(block_hasher, self)
self.block_hashes = self.get_hash_new_full_blocks()

# AFTER (no cycle):
self._block_hasher = block_hasher
self.block_hashes = self._block_hasher(self)
```

Without the cycle, `Request` objects are freed **immediately** by reference
counting when the scheduler removes them -- no cyclic GC needed. This
eliminates the leak regardless of GC frequency or `gc.freeze()` behavior.

### Files changed

- **`vllm/v1/request.py`** -- Store `_block_hasher` instead of
  `partial(block_hasher, self)`. Update `append_output_token_ids()` to call
  `self._block_hasher(self)`.
- **`vllm/v1/core/sched/scheduler.py`** -- Update session block hash call
  site from `session.get_hash_new_full_blocks()` to
  `session._block_hasher(session)`.
- **`tests/v1/core/test_async_scheduler.py`** -- Update test call site.

### Verification (unit test)

With cyclic GC disabled (`gc.disable()`), create 100 Request objects with
prefix caching and delete all external references:

| | main (cycle) | fix (no cycle) |
|---|---|---|
| Objects alive after `del` | **100** (all leaked) | **0** (all freed) |
| Freed by `gc.collect()` | 100 | 0 (nothing to collect) |

All 137 existing tests pass (86 scheduler + 43 kv_cache_utils + 8 async
scheduler).
