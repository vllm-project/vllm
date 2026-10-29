# M2 — where to measure "lost reuse" on the hybrid (attention + GDN) model

Phase 0 of jsboige/vllm#198. Goal: for one request, know

- **(a)** the prefix hit length that the **attention groups alone** would allow;
- **(b)** the **effective** hit length once `MambaManager.find_longest_cache_hit`
  constrains it.

`(a) - (b)` = tokens the engine recomputes *only* because no Gated DeltaNet state
exists at that depth. That number is what Phase 2 (a GDN checkpoint at the radix
branch point) is supposed to bring down, and it is the acceptance measure for the
compaction-replay claim.

Code read: fork at `dcbe5da29e`, synced with upstream 2026-10-25 (PR #197), same
logic as the v0.32.0 prod image plus its tierpin overlay.

**Not implemented in the running engine.** This is a design note plus a proposed
patch guarded by an env var; it needs the overlay patch tree and a user GO.

## 1. Where the two numbers already exist

`HybridKVCacheCoordinator.find_longest_cache_hit`
(`vllm/v1/core/kv_cache_coordinator.py:796-931`) runs a fixed-point loop over
attention groups and carries, for every group:

- `hit_length_by_group` — per-group hit, filled at **:899** and truncated to the
  final length for full-attention groups at **:922**;
- `longest_hit_length` — `max()` over the iterations, **:901**. Full attention is
  evaluated first and is downward-closed (`:848-855`), so on our model (group 3 =
  full attention, groups 0-2 = GDN) it is **the attention-only hit (a)**;
- `hit_length` — the reconciled value every group agrees on, i.e. **the effective
  hit (b)**;
- **:927** `num_uncached_common_prefix_tokens = longest_hit_length - hit_length`
  → exactly `(a) - (b)`.

That third value is already returned to the caller and stored on the request:

- `KVCacheManager.get_computed_blocks` (`vllm/v1/core/kv_cache_manager.py:301-305`)
  receives `(computed_blocks, num_new_computed_tokens, num_uncached)`;
- `:321-322` turns it into `shared_prefix_boundary = num_new_computed_tokens +
  num_uncached` (the Marconi-style pin point for sparse retention), returned at
  `:326`;
- the scheduler stores it on the request — `self.kv_cache_manager.get_computed_blocks(...)`
  at `vllm/v1/core/sched/scheduler.py:1180` (inside `schedule`, declared `:677`),
  and `scheduler.py:652` (inside `_get_local_prefix_cache_hit`, declared `:642`).

So M2 needs **no new lookup and no new computation**: the instrument is a log
formatting line. That is the minimal point.

Where the GDN constraint is produced, for reference:
`MambaManager.find_longest_cache_hit` (`vllm/v1/core/single_type_kv_cache_manager.py:1466-1542`)
searches right-to-left for a cached state block and stops at the first one, then
floors the result to the alignment boundary (`:1527-1531`). The only state
checkpoints a prefill can export are the ones of
`get_mamba_prefill_checkpoint_position` (`vllm/v1/kv_cache_interface.py:1154-1166`,
last prompt boundary floored to the hash block) gated by
`is_mamba_prefill_checkpoint_valid` (`:1169-1187`, needs `query_start %
hash_block_size == 0` and `checkpoint_col > initial_state_col`), plus the running
state at the end of the request. That is why a deeper attention prefix frequently
has no GDN partner.

## 2. Proposed patch (~15 lines, one file, no behaviour change)

File: `vllm/v1/core/kv_cache_manager.py`.

1. `import os` next to the existing imports (:4-29 — `os` is not imported today)
   and, next to `logger = init_logger(__name__)` (:31):

```python
_LOG_LOST_REUSE = os.getenv("MYIA_LOG_LOST_REUSE") == "1"
```

2. In `get_computed_blocks`, right after the coordinator call (after :305:

```python
        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )
```

insert:

```python
        if _LOG_LOST_REUSE and num_uncached:
            logger.info(
                "MYIA_LOST_REUSE req=%s prompt=%d hit_attn_only=%d "
                "hit_effective=%d lost=%d",
                request.request_id,
                request.num_tokens,
                num_new_computed_tokens + num_uncached,
                num_new_computed_tokens,
                num_uncached,
            )
```

That is 12 added lines across two hunks (1 import, 1 constant, 10 in the
function). No state is mutated, no extra block-pool access, and `num_uncached == 0`
on non-hybrid models, so the flag is free when off and near-free when on.

### What the numbers mean

| field | meaning |
|---|---|
| `prompt` | `request.num_tokens`, the prompt length the lookup was bounded by (`max_cache_hit_length` is `num_tokens - 1`, :299-300) |
| `hit_effective` | (b) tokens actually reused (blocks pinned, no recompute) |
| `hit_attn_only` | (a) deepest hit of the full-attention group |
| `lost` | `(a)-(b)` tokens recomputed solely for want of a GDN state at that depth |

One line per request that has a gap; a request whose two hits coincide logs
nothing.

### Known imprecision (state it, do not paper over it)

- `num_uncached` is `longest - effective`, i.e. **"deepest single-group hit minus
  reconciled hit"**, not literally "attention hit minus mamba hit". In production
  order (full attention first, downward-closed) it *is* the attention hit. The
  inverse case exists in code — attention tail evicted while a deeper Mamba state
  survives, described at `kv_cache_manager.py:333-341` — and would then count the
  reverse direction. Distinguishing the two directions costs a second lookup
  (`HybridKVCacheCoordinator.find_longest_cache_hit_per_group`,
  `kv_cache_coordinator.py:933-977`, which evaluates each group independently and
  is already used by the connector path at `:364`): call it **only** when
  `MYIA_LOG_LOST_REUSE=1` and log `hit_attn = hits[full_attention_group_id]`,
  `hit_gdn = min(other groups)`. That is the exact-attribution variant; it doubles
  the lookup only on the instrumented path, so it must never be enabled in steady
  state.
- The attention hit is itself floored by `max_cache_hit_length = num_tokens - 1`
  and by hash-block alignment, so `lost` is a lower bound on the recompute if a
  GDN state existed at the true fork.
- GDN hits only land on positions that own a state block; `lost` therefore
  includes ordinary quantisation of the mamba hit to the alignment boundary, not
  just "compaction replay". For the Phase 2 acceptance number, correlate `lost`
  with the request's prefix-hit source (a compaction replay starts from a fresh
  prompt whose socle hits on attention only).

### Cost

- Off (default): one module-level string compare at import; no per-call cost.
- On: one `logger.info` per request **with** a gap. At the measured rhythm
  (~1,300 requests per 2 h, #191) that is at most a few hundred lines per hour,
  emitted on the scheduler thread. vLLM's logger is synchronous; a
  multi-hundred-line-per-hour burst is negligible against the scheduler's own
  logging, but if the volume ever matters, sample with
  `if num_uncached and _rng.random() < 0.1`.
- No GPU work, no KV allocation, no behaviour change: a rejected store still
  means a recompute, exactly as today.

### Reading it back

The line lands in the engine container log. `docker logs --since` is unreliable
on this host (log split-brain, memory note) — use `docker logs --tail N`, and
date any event by the engine log, not by the watchdog.

```bash
docker logs --tail 20000 myia_vllm-medium-swift15-27b 2>&1 | grep MYIA_LOST_REUSE
```

## 3. Why not the other candidate points

- **Inside `MambaManager.find_longest_cache_hit`** (`single_type_kv_cache_manager.py:1466`):
  sees only mamba's own hit, never the attention one; no request identity.
- **Inside the coordinator** (`kv_cache_coordinator.py:901/927`): both numbers are
  there and the attribution is exact, but the coordinator receives only
  `block_hashes` — no `request_id`, no prompt length, so the log line could not be
  tied to a request. (This is where the exact-attribution variant would stash
  `hit_length_by_group` for the caller, if ever needed.)
- **Scheduler `schedule` (`scheduler.py:1180`)**: `request.shared_prefix_boundary`
  is already on the request object there — a zero-computation alternative if the
  overlay should not touch `kv_cache_manager.py`. It measures the same sum
  (`num_new_computed_tokens + num_uncached`, pinned at `:321-322`) minus the
  ability to tell the pinned boundary from a genuine divergence, and it is set
  only on the admission path, not on the connector re-lookup.
