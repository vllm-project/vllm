# MP Observability — Debug Recipes

How to derive L1-only / L2-only / blend hit rates from the counters in
[METRICS.md](METRICS.md), and what to send to a maintainer when reporting
a hit-rate regression.

---

## Common parameters

All from `GET <lmcache-url>/status`:

| Field | Source | Example |
|---|---|---|
| `chunk_size` | top-level | 256 |
| L1 capacity (bytes) | `storage_manager.l1_manager.memory_total_bytes` | `5 * 2**30` |
| L1 used (bytes) | `storage_manager.l1_manager.memory_used_bytes` | varies |

## Token-level hit rate (L1 + L2 combined)

```
lmcache_mp_lookup_hit_tokens_total / lmcache_mp_lookup_requested_tokens_total
```

Fraction of chunk-aligned tokens that came back from cache anywhere.
Reported per `(model_name, cache_salt)`.

## L2-only hit rate

L2's prefetch lookups carry per-key counts, not per-token:

```
L2_hit_tokens_total = increase(lmcache_mp_l2_prefetch_hit_chunks_total) * chunk_size
L2_hit_rate         = L2_hit_tokens_total
                    / increase(lmcache_mp_lookup_requested_tokens_total)
```

The keys-to-tokens conversion is exact — every L2 hit is a full chunk.

## L1-only hit rate (derived)

Total minus L2:

```
L1_hit_tokens_total = increase(lmcache_mp_lookup_hit_tokens_total)
                    - increase(lmcache_mp_l2_prefetch_hit_chunks_total) * chunk_size
L1_hit_rate         = L1_hit_tokens_total
                    / increase(lmcache_mp_lookup_requested_tokens_total)
```

Sanity check: `L1_hit_tokens_total >= 0`.  A sustained negative value
indicates a metric-emission bug.

## Blend total hit rate

```
lmcache_blend_lookup_hit_tokens_total / lmcache_blend_lookup_requested_tokens_total
```

Reported per CacheBlend lookup (`CB_LOOKUP_END`).

## Blend L1 vs L2 — *not separable today*

The blend lookup consults L1 (fingerprint table) and L2 (storage
prefetch) as a single pipelined operation, and `lmcache_blend.lookup_storage_hits`
aggregates them.  Splitting requires per-chunk attribution inside the
blend lookup; tracked as an open follow-up.

## Did eviction fire during my run?

```
ticks      = increase(lmcache_mp_l1_eviction_loop_ticks_total)
triggered  = increase(lmcache_mp_l1_eviction_loop_triggered_total)
fire_ratio = triggered / ticks   # 0.0 = never fired; 1.0 = fired every cycle
```

If `triggered == 0` over a thrash test, the benchmark completed faster
than the 1Hz polling interval.  Lower `--eviction-trigger-watermark` or
raise `--eviction-ratio` (or run the workload longer) to see eviction
engage.

## Self-service checklist

Attach these four artifacts when reporting a hit-rate regression:

1. `GET <lmcache-url>/status` (config + state snapshot).
2. `GET <lmcache-url>/metrics` snapshots taken **before** and **after** the run.
3. The bench's `bench_summary.json` and `bench_results.csv` (TTFT per request).
4. The LMCache server's stdout/stderr (eviction-trigger logs are at INFO level).

(1)–(3) plus the formulas above let a maintainer compute L1, L2, blend,
and combined hit rates; the eviction-loop counters confirm whether the
LRU machinery actually engaged.
