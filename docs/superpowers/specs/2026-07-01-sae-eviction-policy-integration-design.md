# SAE Eviction Policy Integration into vLLM CPU Offload

**Date**: 2026-07-01
**Branch**: `session_aware_eviction`
**Status**: Draft — awaiting approval

## Summary

Integrate the Session-Aware Eviction (SAE) policy into current vLLM as a
third `CachePolicy` alongside `lru` and `arc` under
`vllm/v1/kv_offload/cpu/policies/`. The policy is selectable via
`kv_connector_extra_config["eviction_policy"] = "sae"` — the same
selection mechanism LRU and ARC already use.

The SAE algorithm was previously delivered as an out-of-tree plugin
package (`sae_kv_offload`, targeting vLLM 0.18.0) at
`/Users/iklamer/ai-native-systems/VSCodeProjects/sae_kv_offload`, where
it registered a full `OffloadingManager` subclass via
`OffloadingSpecFactory`. In current vLLM, `CPUOffloadingManager`
delegates replacement decisions to a pluggable `CachePolicy`; SAE
slots into that seam.

We also add four per-policy cache-effectiveness counters
(`cpu_block_lookup_total`, `cpu_block_hit_total`,
`cpu_block_miss_total`, `block_eviction_total`) with a `policy` label
so all three policies emit uniform metrics on vLLM's existing
`/metrics` endpoint. Counter plumbing lives in `CPUOffloadingManager`,
not in the policies — no changes to the `CachePolicy` interface.

## Goals

- SAE selectable next to LRU and ARC through the existing
  `extra_config["eviction_policy"]` mechanism.
- Zero behavioral change for LRU and ARC.
- No new methods on the `CachePolicy` ABC — SAE implements exactly
  the same interface LRU and ARC do.
- Four cache-effectiveness counters emitted by every CPU-offload
  policy with a `policy` label so one dashboard covers all three.
- One INFO log line at startup identifying the active policy.
- Fail-fast validation of `eviction_policy` and SAE-specific tunables
  at server startup.

## Non-goals

- Environment-variable or TOML config layer for SAE tunables. SAE
  keys live under `kv_connector_extra_config` only, matching how
  other CPU offload knobs (`cpu_bytes_to_use`, `store_threshold`,
  etc.) are configured today.
- Benchmark launcher, comparison harness, or plotting.
- Backwards compatibility with vLLM 0.18. The in-tree integration
  targets current vLLM; the older plugin package remains a separate
  artifact.

## Semantic differences from the reference algorithm

The reference `SessionAwareEvictionManager` (in the plugin package)
was written against vLLM 0.18's `OffloadingManager` interface, which
handed the manager whole batches of block hashes on `lookup`,
`prepare_store`, etc. Current vLLM's `CachePolicy` is a strictly
per-key surface with `touch(keys)` as the only batch method. Two
semantic differences follow — both preserve the algorithm's intent
while fitting the current interface:

**1. Session boundaries are reconstructed from the call sequence.**
The reference algorithm identified a session as "the set of keys
handed to a single `prepare_store` call." SAE now identifies a
session as the run of consecutive `insert` calls arriving between
manager-level events (`touch`, `evict`, `remove`, `clear`). The
first `insert` after any of those opens a new session; subsequent
`insert`s that arrive without an intervening event join it. This
yields the same session grouping the reference produced under the
current interface, because `CPUOffloadingManager.prepare_store`
already calls exactly the same interleaving of hooks (see
[manager.py:151-237](../../../vllm/v1/kv_offload/cpu/manager.py)).

**2. Position-within-batch weighting on lookup is dropped.** The
reference weighted per-key ghost updates by `1/log2(pos + 2)` where
`pos` was the block's index in a per-request lookup batch. The
current scheduler calls `manager.lookup(key, req_context)` one key
at a time and breaks on the first miss
([scheduler.py:391-410](../../../vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py)),
so per-batch position isn't recoverable inside the policy. SAE
drops the position weighting — ghost scores accumulate plain
`ghost_hit_weight` on hits and `ghost_miss_weight` on misses.
Relative ordering of session scores is preserved; absolute
magnitudes shift, and the tunables' defaults are chosen to match
the reference's effective behavior on a representative workload
after this simplification.

## Architecture

### Policy seam

Add `SAECachePolicy` at `vllm/v1/kv_offload/cpu/policies/sae.py`,
implementing the existing `CachePolicy` ABC from
`vllm/v1/kv_offload/cpu/policies/base.py`. Register `"sae"` in
`_CACHE_POLICIES` in `vllm/v1/kv_offload/cpu/manager.py`. No
changes to `CachePolicy`, `CPUOffloadingManager`'s call sequence,
or `CPUOffloadingSpec`'s selection logic.

### How SAE maps onto the existing `CachePolicy` interface

| SAE need                             | Existing hook                                                     |
|--------------------------------------|-------------------------------------------------------------------|
| Ghost score updates on hit / miss    | `get(key)` — called for every lookup key                          |
| Per-session hit accounting           | `touch(keys)` — batch of the request's hit prefix                 |
| Periodic decay                       | Every N-th call to `get` (SAE tracks a global counter internally) |
| Open a session for new keys          | `insert(key, block)` — attach to the currently-open session       |
| Session boundary detection           | `touch`, `evict`, `remove`, `clear` all close the open session    |
| Admission gate (deny the store)      | `evict(n, protected)` returns `None`                              |
| Seed `initial_hits` from ghost sum   | Inside `evict` (before returning success), for the pending sid    |
| Worst-first eviction                 | `evict(n, protected)` walks sessions by SAE's score function      |
| Session bookkeeping on removal       | `remove(key)`, `mark_evictable`, `mark_non_evictable`             |

There is one subtle case that needs explicit handling: **a store
that fits without eviction never calls `evict`**. Under the
reference algorithm, that store still opened a new session and
seeded its `initial_hits`. SAE handles this by seeding
`initial_hits` inside `insert` itself, using the ghost sum of the
keys inserted into the currently-open session up to that point.
The seeding runs once per session — on the first `insert` that
opens the session — using only the ghost score of that first key.
Subsequent inserts into the same session add their ghost score to
the session's `hits` field. This yields the same total as the
reference algorithm's "seed from ghost sum of `to_store`" step:
the sum is just built up incrementally rather than in one shot.

The admission gate still needs somewhere to run. Since the gate
compares a new session's hypothetical score to the worst
incumbent's, it only matters when the cache is full — which is
exactly when `evict` is called. When no eviction is needed, the
new session displaces nothing, so the gate is trivially satisfied.
This matches the reference algorithm's `needed > 0` guard.

### Metrics seam

Add four counter definitions to
`CPUOffloadingSpec.build_metric_definitions()` unconditionally (they
exist for every CPU-offload policy):

| Metric name                       | Type    | Labels                    |
|-----------------------------------|---------|---------------------------|
| `vllm:cpu_block_lookup_total`     | counter | (existing) + `policy`     |
| `vllm:cpu_block_hit_total`        | counter | (existing) + `policy`     |
| `vllm:cpu_block_miss_total`       | counter | (existing) + `policy`     |
| `vllm:block_eviction_total`       | counter | (existing) + `policy`     |

`CPUOffloadingManager` maintains four in-memory delta counters
(`_lookups_delta`, `_hits_delta`, `_misses_delta`,
`_evictions_delta`), incremented in `lookup()` and `prepare_store()`
and flushed to the connector stats payload in `get_stats()` via
`stats.increase_counter(name, delta, labelvalues=(self._policy_name,))`.

The identity `hits + misses == lookups` holds by construction because
all three are incremented at the single classification point in
`lookup()`. `HIT_PENDING` counts as a hit (matches the reference
plugin's convention). `RETRY` is neither a hit nor a miss and does
not increment `lookups`; this is a documented exception to the
invariant.

## Components

### New files

**`vllm/v1/kv_offload/cpu/policies/sae.py`**

`SAECachePolicy(CachePolicy)`. Runtime state (all in-memory):

- `blocks: dict[OffloadKey, BlockStatus]` — the standard block table.
- `sid_to_keys: dict[int, list[OffloadKey]]` — keys owned by each
  session, insertion-ordered.
- `key_to_sid: dict[OffloadKey, int]` — reverse index.
- `sid_stats: dict[int, {"hits": int, "last_touch": int,
  "start_pos": int}]` — per-session bookkeeping.
- `_key_ghost: dict[OffloadKey, float]` — per-key ghost score.
  Decayed periodically; pruned when a key is non-resident and its
  score falls below a small threshold (`0.01`, matching the
  reference).
- `_logical_timer: int` — increments on `get`, `touch`, and `insert`
  boundaries; drives `last_touch`.
- `_sid_counter: int` — monotonically increasing session id source.
- `_lookup_count: int` — number of `get` calls; drives the decay
  tick.
- `_open_sid: int | None` — the currently-open session for
  aggregating consecutive `insert` calls. `None` when no session is
  open.
- `_last_event: str | None` — most recent event kind (`"get"`,
  `"touch"`, `"insert"`, `"evict"`, `"remove"`, `"clear"`, `"init"`);
  used to detect session boundaries.
- `_evictable_keys: OrderedDict[OffloadKey, None]` — keys with
  `ref_cnt == 0`, kept for fast worst-first eviction candidate
  scans. Populated by `mark_evictable`, drained by
  `mark_non_evictable` / `evict` / `remove`.

Configurable tunables (received via constructor kwargs from the
manager): `decay_interval`, `decay_factor`, `ghost_hit_weight`,
`ghost_miss_weight`, `ghost_norm`.

Method overrides (all part of the existing `CachePolicy` ABC — no
new methods):

- `get(key) -> BlockStatus | None` — looks up `key` in `blocks`;
  increments `_lookup_count`; updates `_key_ghost[key]` by
  `ghost_hit_weight` when the key is resident and ready, by
  `ghost_miss_weight` otherwise; on every `decay_interval`-th call,
  scales all session `hits` and all `_key_ghost` values by
  `decay_factor` and prunes ghost entries below the threshold. Sets
  `_last_event = "get"`. Does NOT increment `_logical_timer`
  (session `last_touch` values only advance on real touches, matching
  the reference).
- `insert(key, block)` — if `_last_event` is not `"insert"`, opens a
  new session: assigns `sid = self._sid_counter++`, stores in
  `_open_sid`, seeds `sid_stats[sid]["hits"]` from
  `_key_ghost.get(key, 0.0) / ghost_norm`. Otherwise reuses
  `_open_sid` and adds `_key_ghost.get(key, 0.0) / ghost_norm` to
  the existing `hits`. Appends `key` to `sid_to_keys[sid]`, records
  `key_to_sid[key] = sid`, stores `blocks[key] = block`. Sets
  `_last_event = "insert"`.
- `remove(key)` — cleans up `blocks`, `key_to_sid`,
  `sid_to_keys[sid]` (removing the empty list clears the sid), and
  `_evictable_keys`. Closes the open session (`_open_sid = None`).
  Sets `_last_event = "remove"`.
- `touch(keys)` — increments `_logical_timer`; for each key in
  `keys`, if `key_to_sid[key]` exists, bumps
  `sid_stats[sid]["hits"] += 1` and sets `sid_stats[sid]
  ["last_touch"] = _logical_timer`. Closes the open session. Sets
  `_last_event = "touch"`.
- `evict(n, protected)` — the heart of the algorithm. Closes the
  open session. Runs the admission gate: for the *would-be* new
  session (which the *next* `insert` will open), computes a
  hypothetical score using the ghost sum of the pending
  `keys_to_store`. **But `evict` doesn't know what those keys are.**
  See the "Admission gate" subsection below for the resolution.
  Then walks `sid_stats` sorted by score-worst-first; for each
  session, yields idle keys (`ref_cnt == 0`, not in `protected`)
  from the tail until `n` are collected. Returns `None` if fewer
  than `n` are collectable. Sets `_last_event = "evict"`.
- `clear()` — resets all state; sets `_last_event = "clear"`.
- `mark_evictable(key)` — adds `key` to `_evictable_keys`.
- `mark_non_evictable(key)` — removes `key` from `_evictable_keys`.

**Admission gate resolution.** `evict(n, protected)` is called by
`CPUOffloadingManager.prepare_store` at
[manager.py:200-201](../../../vllm/v1/kv_offload/cpu/manager.py)
with `n = num_blocks_to_evict` and
`protected = set(keys)` — the full input batch (after
`store_threshold` filtering but before removing already-stored keys).
So `protected` is a superset of `keys_to_store` and matches the batch
the reference algorithm's `prepare_store` received in v0.18. The
admission-gate ghost sum is
`sum(_key_ghost.get(k, 0.0) for k in protected) / ghost_norm`.
Because `protected` includes keys that are already stored, ghost
contributions from resident keys still count toward the new
session's hypothetical score — same as the reference.

### Modified files

**`vllm/v1/kv_offload/cpu/policies/base.py`**

No changes. SAE uses only the existing methods.

**`vllm/v1/kv_offload/cpu/manager.py`**

- Register `"sae": SAECachePolicy` in `_CACHE_POLICIES`.
- Widen `cache_policy: Literal["lru", "arc"]` to
  `Literal["lru", "arc", "sae"]`.
- Accept an optional `policy_kwargs: dict[str, Any] | None = None`
  constructor kwarg; pass through to `policy_cls(cache_capacity=...,
  **policy_kwargs)`. Empty dict for LRU/ARC.
- Store `self._policy_name: str = cache_policy`.
- Add delta counters (`_lookups_delta`, `_hits_delta`,
  `_misses_delta`, `_evictions_delta`), initialized to `0`.
- In `lookup()`: increment `_lookups_delta`; on HIT / HIT_PENDING
  increment `_hits_delta`; on MISS increment `_misses_delta`; on
  RETRY increment neither. No new call to the policy — the existing
  `self._policy.get(key)` already fires and is where SAE hooks in.
- In `prepare_store()`: after a successful eviction, add
  `len(evicted)` to `_evictions_delta`. No new call to the policy.
- In `get_stats()`: emit each of the four counter deltas with
  `stats.increase_counter(name, delta, labelvalues=(self._policy_name,))`;
  zero the deltas.

**`vllm/v1/kv_offload/cpu/spec.py`**

- Add the four counter definitions to `build_metric_definitions`.
  Their `labelnames` include `"policy"`.
- Validate `eviction_policy`: must be one of `lru`/`arc`/`sae`;
  otherwise raise `ValueError` naming the offending value and the
  supported set.
- When `eviction_policy != "sae"`, scan `extra_config` for any key
  beginning with `sae_`; if any, raise a single `ValueError`
  listing every offending key and the active policy.
- When `eviction_policy == "sae"`, extract and validate the five
  `sae_*` tunables into a dict; range violations raise `ValueError`
  naming the offending key. Stash on `self._sae_policy_kwargs`.
- In `get_manager()`, pass `policy_kwargs=self._sae_policy_kwargs`
  (empty dict for lru/arc) to `CPUOffloadingManager`.
- Log a single INFO line in `__init__`:
  `logger.info("CPU offload: eviction_policy=%s", self.eviction_policy)`.

### New tests

**`tests/v1/kv_offload/cpu/test_sae_policy.py`**

Pure algorithm tests against `SAECachePolicy` in isolation (no
`CPUOffloadingManager`):

- Session opens on first `insert`; subsequent `insert`s without an
  intervening event join it; `insert` after `touch`/`evict`/
  `remove`/`clear` opens a new session.
- `initial_hits` seeding: session `hits` after all `insert`s equals
  `sum(ghost_scores) / ghost_norm` for the inserted keys.
- Decay tick at `decay_interval`: session `hits` and `_key_ghost`
  values scale by `decay_factor`.
- Ghost pruning: non-resident keys with score `< 0.01` are dropped
  after decay; resident keys are kept regardless.
- Admission gate: when the would-be new session's score is below
  the worst incumbent's, `evict(n, protected)` returns `None`.
- Eviction order: `evict(n, protected)` walks sessions worst-first;
  respects `protected`; skips keys with `ref_cnt != 0`.
- `remove` cleans `sid_to_keys` and empties the sid entry when the
  last key is removed.
- `mark_evictable` / `mark_non_evictable` maintain `_evictable_keys`
  correctly across `ref_cnt` transitions.
- Tunable overrides via constructor kwargs take effect.

**`tests/v1/kv_offload/cpu/test_manager_policy_metrics.py`**

Parametrized over `["lru", "arc", "sae"]`. Drives a small
sequence of `lookup` / `prepare_store` / `complete_store` calls
against `CPUOffloadingManager` and asserts:

- `get_stats()` emits the four counters with `labelvalues=(policy,)`.
- `hits + misses == lookups` after each call (with `HIT_PENDING`
  counted as a hit, `RETRY` counted as neither).
- Eviction counter increments by exactly the number of evicted keys.
- Deltas are cleared each `get_stats()` cycle (repeated calls with
  no activity emit zero-delta counters — no double-counting).

**`tests/v1/kv_offload/cpu/test_spec_config_validation.py`**

- Unknown `eviction_policy` value → `ValueError` at
  `CPUOffloadingSpec.__init__`.
- `sae_*` key present when `eviction_policy != "sae"` → single
  `ValueError` naming every offending key.
- Out-of-range SAE tunable (e.g. `sae_decay_factor=1.5`) →
  `ValueError` naming the offending key.
- Valid SAE config: spec constructs, `get_manager()` returns a
  `CPUOffloadingManager` whose `_policy` is a `SAECachePolicy`
  with the requested tunables.

### Untouched

- `CachePolicy` ABC in `vllm/v1/kv_offload/cpu/policies/base.py`.
- `LRUCachePolicy`, `ARCCachePolicy`.
- `OffloadingConnectorStats`, `OffloadPromMetrics` — the labelled
  counter pipeline already works via `stats.increase_counter(...,
  labelvalues=...)` and the `labelnames` on the metric definition.

## Config surface

All keys under `kv_connector_extra_config`:

| Key                     | Type  | Default | Validation             |
|-------------------------|-------|---------|------------------------|
| `eviction_policy`       | str   | `"lru"` | one of `lru`/`arc`/`sae` |
| `sae_decay_interval`    | int   | `500`   | `>= 1`                 |
| `sae_decay_factor`      | float | `0.9`   | `0.0 < x <= 1.0`       |
| `sae_ghost_hit_weight`  | float | `12.0`  | `>= 0.0`               |
| `sae_ghost_miss_weight` | float | `1.0`   | `>= 0.0`               |
| `sae_ghost_norm`        | float | `12.0`  | `> 0.0`                |

Defaults match the reference algorithm's constants (recognizing the
"Semantic differences" section above: absolute score magnitudes
shift because position weighting is dropped, but the defaults are
the same starting point as the reference and can be re-tuned from
benchmark data separately).

Validation runs in `CPUOffloadingSpec.__init__`; all violations
raise `ValueError` at server startup with the offending key named.

Example `kv-transfer-config`:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": 17179869184,
    "eviction_policy": "sae",
    "sae_decay_interval": 500,
    "sae_decay_factor": 0.9
  }
}
```

## Data flow (policy=`sae`)

1. `CPUOffloadingSpec.__init__` reads `eviction_policy`; when
   `"sae"`, extracts and validates the five `sae_*` tunables into
   `self._sae_policy_kwargs`. Logs
   `"CPU offload: eviction_policy=sae"` at INFO.
2. `CPUOffloadingSpec.get_manager()` calls
   `CPUOffloadingManager(cache_policy="sae",
   policy_kwargs=self._sae_policy_kwargs, ...)`.
3. `CPUOffloadingManager.__init__` instantiates
   `SAECachePolicy(cache_capacity=num_blocks,
   **self._sae_policy_kwargs)`.
4. On each `manager.lookup(key, req_context)`: manager classifies
   `LookupResult`, increments the appropriate delta counters, and
   calls `self._policy.get(key)` as it does today. SAE's `get`
   updates its ghost score for `key` and runs the decay pass on
   the `decay_interval`-th call.
5. On `manager.prepare_store(keys, req_context)`: manager filters
   already-stored keys and calls `self._policy.evict(n, protected)`
   when eviction is needed. SAE's `evict` runs the admission gate;
   on failure returns `None` (manager returns `None` from
   `prepare_store`). On success returns the eviction list. Then
   the manager calls `self._policy.insert(key, block)` for each
   `key` in `keys_to_store`; SAE's `insert` opens a new session
   (unless the previous event was also `insert`) and seeds
   `initial_hits` incrementally from ghost scores.
6. On `manager.touch(keys, req_context)`: manager calls
   `self._policy.touch(keys)`. SAE bumps session hits and
   `last_touch`.
7. On `manager.get_stats()`: the four counter deltas are emitted
   with `labelvalues=(self._policy_name,)` and reset.

## Error handling

All errors are fail-fast at `CPUOffloadingSpec.__init__` (server
startup):

- `eviction_policy` not in the supported set → `ValueError` naming
  the offending value and listing supported values.
- Any `sae_*` key present when `eviction_policy != "sae"` → single
  `ValueError` listing every offending key and the active policy
  name.
- Individual SAE tunable out of range → `ValueError` naming the
  offending key and its expected shape. First offender wins.

No new runtime error paths. When SAE's admission gate denies a
store, `evict` returns `None` and the manager returns `None` from
`prepare_store` — the same "eviction failed" contract LRU and ARC
already produce today.

## Docs

- Docstring on `SAECachePolicy` summarizing the algorithm (sessions
  reconstructed from the call sequence, ghost scores, decay,
  admission gate, no position weighting) and pointing at the
  reference implementation in the plugin package.
- Docstrings on the four new counter definitions in
  `build_metric_definitions` describing what each counts and the
  `hits + misses == lookups` invariant.
- Small addition to the existing CPU-offload doc page mentioning
  `"sae"` as an accepted `eviction_policy` value alongside `"lru"`
  and `"arc"`, and pointing at the tunables. No standalone SAE
  doc page.

## Out of scope

- Environment-variable or TOML config layer for SAE tunables.
- Benchmark launcher, comparison harness, or plotting.
- Custom logging plumbing beyond the one INFO line at startup.
- Any change to LRU or ARC behavior.
- Any change to the `CachePolicy` ABC.
- Any change to `OffloadingConnectorStats` or `OffloadPromMetrics`.
- Backwards compatibility with vLLM 0.18.
