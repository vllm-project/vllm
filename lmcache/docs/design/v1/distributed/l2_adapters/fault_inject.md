# Fault-Injecting L2 Adapter

`lmcache/v1/distributed/l2_adapters/fault_inject_l2_adapter.py`

A **test/diagnostic-only** L2 adapter that wraps a real inner adapter (e.g.
`fs_native`) and deterministically drops keys on the load read, simulating
partial L2 retrieve failures. Its purpose is to exercise the **segmented**
code paths — gapped found-set → segmented prefetch → segmented scatter /
attention — that a healthy cache never produces, so CacheBlend's
segmented-prefix handling can be tested without an actual faulty L2.

## Fault model

It faults the **load read primitive only**. A dropped key is reported
**present at lookup** but its **load fails** — the faithful "L2 retrieve
error". The lookup-miss case is *intentionally not modeled*: a key absent at
lookup merely shortens the found-set, which the `PREFIX` trim policy
(`count_leading_ones`) already covers. Modeling load-failure-after-lookup is
what produces a *gapped* found-set (hit … miss … hit), the input the
segmented path is built to handle.

When a load fails, **no unlock is issued here** — the prefetch controller
releases the load-failed read locks itself via the trim mask.

## Drop selection

A load position is dropped (`_drop_positions` / `_should_drop_key`) if **any**
of the following hold:

| knob | meaning | use |
|---|---|---|
| `gap_tail_ratios: list[float]` | positions given as **distance-from-tail ÷ load-length** in `[0,1]` (`0.0`=last, `0.5`=middle, `1.0`=first); the dropped slot `round((1-ratio)·(n-1))` is computed from the `n`-key load batch the server actually receives. **Workload-agnostic and self-scaling** — the server needs no advance knowledge of the stored content, and the same fraction picks the right chunk at any context length. | primary lever for segmented repros (drop the mid chunk → a stable gap at any length) |
| `gap_indices: list[int]` | exact head-relative task-positions to always drop | precise single-gap unit tests |
| `rate: float` + `seed: int` | per-key drop probability in `[0,1]` via a stable seeded hash of the key, bucketed by `rate` | randomized resilience sweeps; deterministic given the seed |

Defaults are pass-through (`rate=0.0`, empty `gap_indices`/`gap_tail_ratios`).

> **Why a ratio and not a chunk hash?** An earlier `drop_chunk_hashes` knob keyed
> drops on `ObjectKey.chunk_hash` — content-addressed, so the *same* chunk failed
> on every leg. But it forced the test author (and thus the server config) to know
> a hash computed from a not-yet-stored workload, which inverted the dependency
> (the server "knowing" a future workload). `gap_tail_ratios` is a positional rule
> the server evaluates from the load batch alone, so no content needs to be known
> in advance. Trade-off: the drop is **per-load-batch**, not content-identical
> across legs — for the segmented-prefix repro the gap lives in the prefix leg's
> load, which is exactly what's exercised.

## Configuration

Activated through `--l2-adapter` (or the engine L2 spec). The required
`inner` sub-dict is a full adapter spec — dispatched through the L2 adapter
config registry — so any inner adapter type works and keeps its own
eviction/persist/serde config:

```json
{
  "type": "fault_inject",
  "inner": { "type": "fs_native", "base_path": "/dev/shm/cb_l2" },
  "gap_tail_ratios": [0.5],
  "rate": 0.0,
  "seed": 0,
  "gap_indices": []
}
```

`"gap_tail_ratios": [0.5]` drops the middle chunk of every load — the stable
mid-prefix gap the segmented repro needs, at any context length.

At startup it logs `FaultInjectL2Adapter ACTIVE (rate=… seed=… gap_indices=…
gap_tail_ratios=[…]) wrapping <inner> -- test/diagnostic use only.`

## Data flow

```
submit_load_task(keys)         → delegate to inner; stash keys under task_id
query_load_result(task_id)     → bitmap = inner.query_load_result(task_id)
                                 for i in _drop_positions(keys): bitmap.clear(i)
                                 return bitmap        # dropped bits now read "miss"
```

`submit_load_task` records the per-task key list so `query_load_result` can map
the dropped bit positions back to keys; the result query is **non-idempotent**
(a non-`None` bitmap is returned once per task), matching the inner contract.

## Delegation

Everything other than the load-result query passes straight through to the
inner adapter: store, lookup-and-lock, unlock, delete, the event fds,
listener registration, usage, and global-eviction support. The fault layer
holds no data of its own. `report_status()` returns the inner status annotated
with a `fault_inject` sub-dict (`rate`, `seed`, `gap_indices`,
`gap_tail_ratios`) so the active fault config is visible in diagnostics.

## Scope and safety

Test/diagnostic only — never enable in production. It silently makes reads
fail, so any served KV is intentionally incomplete. The primary consumer is
the CacheBlend **segmented-prefix** validation (the `ci_v3_l2_hit`
`segmented_prefix` workload sets `gap_tail_ratios=[0.5]` to force a gapped
prefix, so V3 loads prefix+tail and recomputes only the gap).
