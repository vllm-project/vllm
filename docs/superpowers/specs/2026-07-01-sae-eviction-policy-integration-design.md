# SAE Eviction Policy Integration into vLLM CPU Offload

**Date**: 2026-07-01
**Branch**: `session_aware_eviction`
**Status**: Draft — awaiting approval

## Summary

Integrate the Session-Aware Eviction (SAE) policy into current vLLM as a
third `CachePolicy` alongside `lru` and `arc` under
`vllm/v1/kv_offload/cpu/policies/`. The policy is selectable via
`kv_connector_extra_config["eviction_policy"] = "sae"`. In addition,
add four per-policy cache-effectiveness counters
(`cpu_block_lookup`, `cpu_block_hit`, `cpu_block_miss`,
`block_eviction`) with a `policy` label so all three policies emit
uniform metrics on vLLM's existing `/metrics` endpoint.

The SAE algorithm was previously delivered as an out-of-tree plugin
package (`sae_kv_offload`, targeting vLLM 0.18.0) at
`/Users/iklamer/ai-native-systems/VSCodeProjects/sae_kv_offload`. That
package registered a full `OffloadingManager` subclass via
`OffloadingSpecFactory`. In current vLLM, `CPUOffloadingManager` has
been refactored to delegate replacement decisions to a pluggable
`CachePolicy`; SAE fits into that seam.

## Goals

- SAE selectable next to LRU and ARC through the existing
  `extra_config["eviction_policy"]` mechanism — no new selection surface.
- Zero behavioral change for LRU and ARC.
- Four cache-effectiveness counters emitted by every CPU-offload
  policy with a `policy` label so a single dashboard covers all three.
- One INFO log line at startup identifying the active policy.
- Fail-fast validation of `eviction_policy` and SAE-specific tunables
  at server startup.

## Non-goals

- Environment-variable or TOML config layer for SAE tunables. SAE keys
  live under `kv_connector_extra_config` only, matching how other CPU
  offload knobs (`cpu_bytes_to_use`, `store_threshold`, etc.) are
  configured today.
- Benchmark launcher, comparison harness, or plotting. The original
  plugin shipped these; they are out of scope for the in-tree
  integration.
- Backwards compatibility with vLLM 0.18. The in-tree integration
  targets whatever version this branch is cut from; the older plugin
  package remains a separate artifact.

## Architecture — three seams

The integration lands cleanly on three existing extension points:

### Policy seam

Add `SAECachePolicy` at `vllm/v1/kv_offload/cpu/policies/sae.py`,
implementing the `CachePolicy` ABC from
`vllm/v1/kv_offload/cpu/policies/base.py`. Register `"sae"` in
`_CACHE_POLICIES` in `vllm/v1/kv_offload/cpu/manager.py` alongside
`"lru"` and `"arc"`. No changes to how `CPUOffloadingSpec` selects
the policy — it already reads `extra_config["eviction_policy"]`
(default `"lru"`).

### Interface extension

Extend `CachePolicy` with two optional hooks, defaulted to no-op so
LRU and ARC are unaffected:

```python
def on_lookup(
    self, key: OffloadKey, result: LookupResult, req_context: ReqContext
) -> None:
    """Called after each per-key lookup. Policies use this for
    behavioral bookkeeping (ghost scores, session state)."""
    return

def on_prepare_store(
    self, keys_to_store: Collection[OffloadKey], req_context: ReqContext
) -> bool:
    """Called from prepare_store after already-stored keys are filtered
    and before eviction. Return False to deny admission (manager returns
    None from prepare_store, same as an eviction failure)."""
    return True
```

`CPUOffloadingManager.lookup` calls `on_lookup` immediately after
classifying the `LookupResult` (and after updating the internal
counters, so counter accounting is independent of policy hooks).
`CPUOffloadingManager.prepare_store` calls `on_prepare_store` after
the `keys_to_store` filter step and before eviction; on `False`, it
returns `None`. LRU and ARC inherit the default implementations
and their behavior is byte-identical to before.

### Metrics seam

Add four counter definitions to
`CPUOffloadingSpec.build_metric_definitions()` unconditionally (they
exist for every CPU-offload policy):

| Metric name                   | Type    | Labels                    |
|-------------------------------|---------|---------------------------|
| `vllm:cpu_block_lookup_total` | counter | (existing) + `policy`     |
| `vllm:cpu_block_hit_total`    | counter | (existing) + `policy`     |
| `vllm:cpu_block_miss_total`   | counter | (existing) + `policy`     |
| `vllm:block_eviction_total`   | counter | (existing) + `policy`     |

`CPUOffloadingManager` maintains four in-memory delta counters
(`_lookups_delta`, `_hits_delta`, `_misses_delta`,
`_evictions_delta`), incremented in `lookup()` and `prepare_store()`
and flushed to the connector stats payload in `get_stats()` via
`stats.increase_counter(name, delta, labelvalues=(self._policy_name,))`.

The identity `hits + misses == lookups` holds by construction because
all three are incremented at the single classification point in
`lookup()`; `HIT_PENDING` counts as a hit (matches the reference
plugin's convention). `RETRY` is neither a hit nor a miss and does
not increment `lookups` (documented as the invariant's stated
exception).

## Components

### New files

**`vllm/v1/kv_offload/cpu/policies/sae.py`**

`SAECachePolicy(CachePolicy)`. Ports the algorithmic core of
`SessionAwareEvictionManager` from the plugin package into the
`CachePolicy` shape. Runtime state (all in-memory):

- `blocks: dict[OffloadKey, BlockStatus]` — the standard block table.
- `sid_to_keys: dict[int, list[OffloadKey]]` — keys owned by each
  session, insertion-ordered.
- `key_to_sid: dict[OffloadKey, int]` — reverse index.
- `sid_stats: dict[int, dict]` — per-session `hits`, `last_touch`,
  `start_pos`.
- `_key_ghost: dict[OffloadKey, float]` — per-key ghost score,
  decayed periodically, pruned when a key is non-resident and the
  score falls below a small threshold.
- `_pos_weights: list[float]` — precomputed `1/log2(i+2)` for
  `i in range(1024)`.
- `_logical_timer: int`, `_sid_counter: int`, `_lookup_count: int`.
- `_last_lookup_sid: int`, `_last_lookup_count: int`,
  `_last_lookup_timer: int` — for merge detection.
- `_req_state: dict[<req_key>, {"pos": int, "hits": int, "last": LookupResult}]`
  — per-request cursor reconstructing batch/position semantics from
  the per-key scheduler interface. `<req_key>` is the identity
  extracted from `req_context` (specifically the job/request
  identifier; exact field pinned during implementation).

Configurable tunables (received via constructor kwargs from the
manager): `decay_interval`, `decay_factor`, `ghost_hit_weight`,
`ghost_miss_weight`, `ghost_norm`. Defaults match the reference
algorithm at `sae_kv_offload/manager.py`.

Overrides:

- `get / insert / remove / touch / evict / clear` — the standard
  `CachePolicy` interface. `evict(n, protected)` walks sessions
  sorted worst-first by SAE's score function
  (`last_touch + 1500·hits + 30000/(1 + start_pos/8)`), yielding
  idle keys from each session's tail until `n` are collected;
  returns `None` if fewer than `n` are collectable.
- `on_lookup(key, result, req_context)` — updates the per-request
  cursor, ghost score for `key` weighted by position (hit weight
  for prefix hits, miss weight afterwards), session stats when the
  key belongs to a resident session. On every `decay_interval`
  lookups (tracked globally), decays session hits and ghost scores
  by `decay_factor` and prunes ghost entries below a small
  threshold.
- `on_prepare_store(keys_to_store, req_context)` — computes
  `is_merging` from `_last_lookup_*` and the batch's `start_pos`;
  when not merging, runs the admission gate: if the new session's
  hypothetical score is worse than the worst incumbent session, the
  gate returns `False`. Otherwise it seeds `initial_hits` for the
  new session from the sum of ghost scores of `keys_to_store`
  divided by `ghost_norm`, records the new sid in `sid_stats`, and
  returns `True`. Records `_last_lookup_*` from the terminating
  request state.
- `mark_evictable / mark_non_evictable` — no-op. SAE's `evict`
  walks sessions in worst-first order and filters candidates by
  `block.ref_cnt == 0` at eviction time, so it does not need a
  separately-tracked evictable list like LRU's `evictable_blocks`
  OrderedDict. The manager still calls these hooks and the base
  class's default implementations return `None`, matching current
  behavior for policies that don't need the signal.

### Modified files

**`vllm/v1/kv_offload/cpu/policies/base.py`**

Add `on_lookup` and `on_prepare_store` to `CachePolicy` with no-op
default implementations (returning `None` and `True` respectively).
Update the ABC's class docstring to describe the two hooks.

**`vllm/v1/kv_offload/cpu/manager.py`**

- Register `"sae": SAECachePolicy` in `_CACHE_POLICIES`.
- Widen `cache_policy: Literal["lru", "arc"]` to
  `Literal["lru", "arc", "sae"]`.
- Accept an optional `policy_kwargs: dict[str, Any] | None = None`
  constructor kwarg; pass through to `policy_cls(cache_capacity=...,
  **policy_kwargs)`.
- Store `self._policy_name: str = cache_policy`.
- Add delta counters (`_lookups_delta`, `_hits_delta`,
  `_misses_delta`, `_evictions_delta`), initialized to `0`.
- In `lookup()`: increment `_lookups_delta`; on HIT / HIT_PENDING
  increment `_hits_delta`; on MISS increment `_misses_delta`; on
  RETRY increment neither. Call `self._policy.on_lookup(key,
  result, req_context)` at the end.
- In `prepare_store()`: after the `keys_to_store` filter and before
  the eviction path, call `self._policy.on_prepare_store(
  keys_to_store, req_context)`; on `False`, return `None`. After a
  successful eviction, add `len(evicted)` to `_evictions_delta`.
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

Pure algorithm tests against `SAECachePolicy` in isolation (a
minimal fake context object; no `CPUOffloadingManager`):

- Session creation and `initial_hits` seeding from ghost sum.
- Decay tick at `sae_decay_interval`: session hits and ghost scores
  both scale by `decay_factor`.
- Ghost pruning: non-resident keys with score `< 0.01` are dropped
  after decay; resident keys are kept regardless.
- Admission gate: when the new session's score is below the worst
  incumbent's, `on_prepare_store` returns `False`.
- Eviction order: `evict(n, protected)` walks sessions worst-first;
  respects `protected`.
- Merge detection: consecutive lookup + prepare_store with matching
  `start_pos` merges into `_last_lookup_sid`.
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

- `LRUCachePolicy`, `ARCCachePolicy` — inherit the new no-op hooks
  unchanged.
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

Defaults match the reference algorithm. Validation runs in
`CPUOffloadingSpec.__init__`; all violations raise `ValueError` at
server startup with the offending key named.

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
   `LookupResult`, increments the appropriate delta counters, then
   calls `policy.on_lookup(key, result, req_context)`. SAE
   maintains a per-request cursor (position within the current
   scan, running hit count, direction) keyed by `req_context`'s
   request identity, and applies ghost-score updates and session
   stats accordingly. On the `decay_interval`th lookup globally,
   the policy runs its decay pass.
5. On `manager.prepare_store(keys, req_context)`: manager filters
   already-stored keys, then calls
   `policy.on_prepare_store(keys_to_store, req_context)`. SAE
   runs the admission gate; on `False` the manager returns `None`.
   On `True`, SAE seeds `initial_hits` from the ghost sum and
   records the new session. The manager then proceeds through
   eviction and allocation as normal; eviction candidates come
   from `policy.evict(n, protected)`, which SAE implements by
   walking sessions worst-first.
6. On `manager.get_stats()`: the four counter deltas are emitted
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
store, `on_prepare_store` returns `False` and the manager returns
`None` from `prepare_store` — the same "eviction failed" contract
LRU and ARC already produce today.

## Docs

- Docstring on `SAECachePolicy` summarizing the algorithm (sessions,
  ghost scores, decay, admission gate, merge detection) and pointing
  at the reference implementation.
- Docstrings on the two new `CachePolicy` hooks describing when they
  fire and what the boolean return of `on_prepare_store` means.
- Docstrings on the four new counter definitions in
  `build_metric_definitions` describing what each counts and the
  `hits + misses == lookups` invariant.
- Small addition to the existing CPU-offload doc page mentioning
  `"sae"` as an accepted `eviction_policy` value alongside `"lru"`
  and `"arc"`, and pointing at the tunables via their docstrings.
  No standalone SAE doc page.

## Out of scope

- Environment-variable or TOML config layer for SAE tunables.
- Benchmark launcher, comparison harness, or plotting.
- Custom logging plumbing beyond the one INFO line at startup.
- Any change to LRU or ARC behavior.
- Any change to `OffloadingConnectorStats` or `OffloadPromMetrics`.
- Backwards compatibility with vLLM 0.18.
