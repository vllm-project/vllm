# Event Metadata Contracts

Each `EventType` has a documented metadata schema.  Producers **must** populate
these keys; subscribers **may** rely on them being present.

For the full list of event types see `event.py`.  For metrics derived from
these events see [METRICS.md](METRICS.md).

---

## L1Manager Events

| EventType | Metadata keys | Types |
|---|---|---|
| `L1_READ_RESERVED` | `keys` | `list[ObjectKey]` |
| `L1_READ_FINISHED` | `keys` | `list[ObjectKey]` |
| `L1_WRITE_RESERVED` | `keys` | `list[ObjectKey]` |
| `L1_WRITE_FINISHED` | `keys` | `list[ObjectKey]` |
| `L1_WRITE_FINISHED_AND_READ_RESERVED` | `keys` | `list[ObjectKey]` |
| `L1_KEYS_EVICTED` | `keys` | `list[ObjectKey]` |
| `L1_EVICTION_LOOP_TICK` | `usage`, `watermark`, `triggered` | `float`, `float`, `bool` |

`L1_EVICTION_LOOP_TICK` fires once per `L1EvictionController.eviction_loop`
iteration (default ~1Hz).  `triggered` is `True` when `usage >= watermark`
and the policy ran this cycle, `False` otherwise.

---

## L1 Failure Events

Published at the caller of the L1 API — **not** inside `L1Manager` — so the
caller context (`during`) can be attached without leaking caller identity
into the manager. See LM-291 for the health-monitoring rationale.

Producers split keys by `(during[, reason])` and publish one event per
non-empty bucket; `keys` is the list of ObjectKeys that failed for that
bucket. Subscribers bucket by `ObjectKey.model_name` to emit per-model
counter increments.

| EventType | Metadata keys | Types | Vocabulary |
|---|---|---|---|
| `L1_ALLOCATION_FAILED` | `during`, `keys` | `str`, `list[ObjectKey]` | `during` ∈ {`l1_store`, `l2_prefetch`} |
| `L1_READ_FAILED` | `during`, `reason`, `keys` | `str`, `str`, `list[ObjectKey]` | `during` ∈ {`l2_store`, `l1_retrieve`}; `reason` ∈ {`not_found`, `write_locked`} |

`L1_READ_FAILED` is a **post-lookup anomaly** event, not a cache-miss
event: in MP mode every `reserve_read` / `unsafe_read` that raises one of
these errors represents a lookup/reserve race or unexpected eviction. The
counter should stay near zero in healthy operation.

Producers:
- `L1_ALLOCATION_FAILED(during=l1_store)` — `StorageManager.reserve_write`, on `L1Error.OUT_OF_MEMORY`.
- `L1_ALLOCATION_FAILED(during=l2_prefetch)` — `PrefetchController._transition_to_load_phase`, on `L1Error.OUT_OF_MEMORY` from L1 reservation.
- `L1_READ_FAILED(during=l1_retrieve)` — `StorageManager.read_prefetched_results` (`unsafe_read` failure). `not_found` = `KEY_NOT_EXIST`, `write_locked` = `KEY_NOT_READABLE` (read lock lost).
- `L1_READ_FAILED(during=l2_store)` — `StoreController._process_new_keys` (`reserve_read` failure on keys that just finished L1 write). `not_found` = `KEY_NOT_EXIST`, `write_locked` = `KEY_NOT_READABLE`.

---

## StorageManager Events

| EventType | Metadata keys | Types |
|---|---|---|
| `SM_READ_PREFETCHED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `SM_READ_PREFETCHED_FINISHED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `SM_WRITE_RESERVED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `SM_WRITE_FINISHED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |

---

## L2 Store Controller Events

| EventType | Metadata keys | Types |
|---|---|---|
| `L2_STORE_SUBMITTED` | `adapter_index`, `task_id`, `l2_name`, `key_count`, `total_bytes`, `key_count_per_salt` | `int`, `int`, `str`, `int`, `int`, `dict[str, int]` |
| `L2_STORE_COMPLETED` | `adapter_index`, `task_id`, `l2_name`, `bytes_transferred`, `succeeded_count`, `failed_count`, `key_count_per_salt` | `int`, `int`, `str`, `int`, `int`, `int`, `dict[str, int]` |

`key_count_per_salt` maps each `cache_salt` to its key count, pre-grouped
at the emit site so the drain thread iterates tenants (O(T)) not keys (O(N)).
Store tasks can batch keys from multiple tenants; `key_count_per_salt`
enables per-tenant attribution on the subscriber side.
On the failure path of `L2_STORE_COMPLETED`, `key_count_per_salt` is absent
(no succeeded keys to attribute).

---

## L2 Prefetch Controller Events

| EventType | Metadata keys | Types |
|---|---|---|
| `L2_PREFETCH_LOOKUP_SUBMITTED` | `request_id`, `key_count`, `adapter_count`, `key_count_per_salt` | `int`, `int`, `int`, `dict[str, int]` |
| `L2_PREFETCH_LOOKUP_COMPLETED` | `request_id`, `prefix_hit_count` | `int`, `int` |
| `L2_PREFETCH_LOAD_SUBMITTED` | `request_id`, `key_count`, `adapter_count`, `key_count_per_salt` | `int`, `int`, `int`, `dict[str, int]` |
| `L2_PREFETCH_LOAD_COMPLETED` | `request_id`, `loaded_count`, `failed_count`, `key_count_per_salt` | `int`, `int`, `int`, `dict[str, int]` |
| `L2_LOAD_TASK_SUBMITTED` | `request_id`, `adapter_index`, `task_id`, `l2_name`, `key_count`, `total_bytes` | `int`, `int`, `int`, `str`, `int`, `int` |
| `L2_LOAD_TASK_COMPLETED` | `request_id`, `adapter_index`, `task_id`, `l2_name` | `int`, `int`, `int`, `str` |

`key_count_per_salt` on `L2_PREFETCH_LOOKUP_SUBMITTED`,
`L2_PREFETCH_LOAD_SUBMITTED`, and `L2_PREFETCH_LOAD_COMPLETED` enables
per-tenant metric attribution. `key_count_per_salt` on load-completed
covers only loaded (succeeded) keys.

`L2_LOAD_TASK_*` events fire once per `(request_id, adapter_index)` pair
— unlike the request-level `L2_PREFETCH_LOAD_*` events above, which
aggregate across adapters.  Throughput subscribers that need per-adapter
attribution (e.g. `L2ThroughputSubscriber`) consume these task-level
events; key-count counters continue to consume the request-level events.

---

## L2 Eviction Controller Events

| EventType | Metadata keys | Types |
|---|---|---|
| `L2_KEYS_EVICTED` | `key_count`, `key_count_per_salt` | `int`, `dict[str, int]` |

Published by `L2EvictionController._execute_eviction_action` after
`adapter.delete()` completes. Only emitted when at least one key was
evicted. `key_count_per_salt` enables per-tenant eviction dashboards.

---

## L2 Failure Events

Health-monitoring event for the L2 prefetch path. See LM-291.

| EventType | Metadata keys | Types | Vocabulary |
|---|---|---|---|
| `L2_PREFETCH_FAILED` | `reason`, `keys` | `str`, `list[ObjectKey]` | `reason` ∈ {`l1_oom`, `not_found`} |

Producers (both in `PrefetchController`):
- `reason=l1_oom` — emitted when `reserve_write` into L1 returns `OUT_OF_MEMORY` during the transition-to-load phase. Published in parallel with `L1_ALLOCATION_FAILED(during=l2_prefetch)`.
- `reason=not_found` — emitted in `_finalize_load` for keys reserved in L1 but missing from the adapter's load bitmap (L2 reported the key present at lookup but produced no data).

The third reason `serde_failure` will be added as an additive, non-breaking
extension once the serde PR lands and adapters can distinguish
deserialization errors from missing objects. No dashboard migration
needed when that happens.

---

## Timeout Events

Cross-component health event. Published by `LMCacheTimeoutError.__init__`
(see `lmcache/v1/mp_observability/errors.py`) every time the exception is
constructed, gated by `is_observability_enabled()` so it is a no-op outside
the MP server process. The `session_id` on the `Event` dataclass correlates
the timeout with the originating request when the raise site has it; it is
empty otherwise.

| EventType | Metadata keys | Types |
|---|---|---|
| `TIMEOUT_RAISED` | `message`, `exception_type`, `stacktrace` | `str`, `str`, `str` |

- `message` — the exception's message string (`str(exc)`).
- `exception_type` — the class name (`LMCacheTimeoutError` or a subclass).
- `stacktrace` — the formatted construction stack (`traceback.format_stack()`
  minus the `__init__` frame), surfaced as the OTel `exception.stacktrace`
  attribute by the tracing subscriber.

Consumed by `TimeoutMetricsSubscriber` (counter), `TimeoutLoggingSubscriber`
(warning log), and `TimeoutTracingSubscriber` (OTel span with an `exception`
event + ERROR status).

---

## MP Server Lifecycle Sentinels

CPU-synchronous sentinels published by `server.py` to bracket request scope.
Published via `EventBus.publish()` (not `publish_on_stream`) so the drain
thread processes them in strict order before any GPU-callback events.

| EventType | Metadata keys | Types | Published by / when |
|---|---|---|---|
| `MP_REQUEST_START` | *(none)* | — | `MPServer.handle_request` — at request arrival, before any GPU work |
| `MP_STORE_SUBMITTED` | `device` | `str` | `MPServer.store` — CPU-synchronous, before the GPU store is enqueued |
| `MP_RETRIEVE_SUBMITTED` | `device` | `str` | `MPServer.retrieve` — CPU-synchronous, before the GPU retrieve is enqueued |
| `MP_REQUEST_END` | *(none)* | — | `MPServer.handle_request` — after all CPU work; may precede GPU callbacks |

---

## MP Server Events

These events use `session_id` on the `Event` dataclass (not in `metadata`)
to correlate START/END pairs.

| EventType | Metadata keys | Types |
|---|---|---|
| `MP_STORE_START` | `device`, `engine_id`, `model_name` | `str`, `int`, `str` |
| `MP_STORE_END` | `device`, `stored_count`, `engine_id`, `model_name`, `total_bytes`, `num_tokens` | `str`, `int`, `int`, `str`, `int`, `int` |
| `MP_RETRIEVE_START` | `device`, `engine_id`, `model_name` | `str`, `int`, `str` |
| `MP_RETRIEVE_END` | `device`, `retrieved_count`, `engine_id`, `model_name`, `cache_salt`, `total_bytes`, `num_tokens` | `str`, `int`, `int`, `str`, `str`, `int`, `int` |
| `MP_LOOKUP_PREFETCH_START` | *(none)* | — |
| `MP_LOOKUP_PREFETCH_END` | `found_count`, `requested_tokens`, `hit_tokens`, `model_name`, `cache_salt` | `int`, `int`, `int`, `str`, `str` |
| `MP_LOOKUP` | `request_id`, `chunk_hashes`, `model_name`, `chunk_size`, `seq_len`, `dtypes`, `shapes` | `str`, `list[str]`, `str`, `int`, `int`, `list[str]`, `list[list[int]]` |
| `MP_VLLM_BLOCK_ALLOCATION` | `instance_id`, `model_name`, `records` | `int`, `str`, `list[BlockAllocationRecord]` (each has `req_id: str`, `new_block_ids: list[int]`, `new_token_ids: list[int]`) |
| `MP_VLLM_END_SESSION` | `request_id` | `str` |

### `num_tokens` on `MP_STORE_END` / `MP_RETRIEVE_END`

Denormalized token count (`num_chunks * chunk_size`) so subscribers need
not know `chunk_size`.  Fail-closed, matching `total_bytes`: `0` when the
store committed nothing (`stored_count == 0`) or the retrieve did not
fully succeed.

### `MP_LOOKUP_PREFETCH_END` metadata

`found_count` is the contiguous prefix hit at chunk granularity, already
divided by `world_size` at the emit site.  `requested_tokens` and
`hit_tokens` are denormalized token-level counts so subscribers need not
know `chunk_size`:

- `requested_tokens = len(chunk_hashes) * chunk_size` on the happy path; `0`
  on the two early-exit paths in `lookup()` (no matching GPU context,
  empty `chunk_hashes`).  Sub-chunk trailing tokens are excluded — they
  cannot hit at chunk granularity.
- `hit_tokens = found_count * chunk_size`.
- `model_name` and `cache_salt` are captured at lookup time from
  `IPCCacheServerKey` and surface as OTel attributes on the
  `lmcache_mp.lookup_*_tokens` counters so the hit rate can be sliced
  per model and per tenant / isolation domain on the dashboard.
  `cache_salt` may have high cardinality (e.g. one entry per tenant);
  operators can drop the label at scrape time with a `metric_relabel_configs`
  rule if storage cost matters.

Together they drive the `lmcache_mp.lookup_*_tokens` counters used to
compute the L1+L2 token-level hit rate.  See
[L1_L2_HIT_RATE_PLAN.md](L1_L2_HIT_RATE_PLAN.md) for the design.

---

## Trace Recording Events

A single unified event used by the `@enable_tracing` decorator (see
[trace.md](trace.md)). All instrumented call sites publish the same
`EventType` regardless of which method or layer; the `qualname` field
inside `metadata` discriminates ops.

| EventType | Metadata keys | Types |
|---|---|---|
| `TRACE_CALL` | `qualname`, `args` | `str`, `dict[str, Any]` (codec-encoded; see `lmcache.v1.mp_observability.trace.codecs`) |

---

## Blend Server Lifecycle Sentinels

CPU-synchronous sentinels published by `blend_server_v2.py` to bracket
request scope and guard GPU callback races.  Published via `EventBus.publish()`
(not `publish_on_stream`).

| EventType | Metadata keys | Types | Published by / when |
|---|---|---|---|
| `CB_REQUEST_START` | *(none)* | — | `BlendEngineV2.cb_lookup_pre_computed` — at request arrival |
| `CB_STORE_PRE_COMPUTED_SUBMITTED` | `instance_id` | `int` | `BlendEngineV2.cb_store_pre_computed` — before GPU store enqueue |
| `CB_RETRIEVE_SUBMITTED` | `instance_id` | `int` | `BlendEngineV2.cb_retrieve_pre_computed` — before GPU retrieve enqueue |
| `CB_STORE_FINAL_SUBMITTED` | `instance_id` | `int` | `BlendEngineV2.cb_store_final` — before GPU store enqueue |
| `CB_REQUEST_END` | *(none)* | — | `BlendEngineV2.cb_lookup_pre_computed` (early return: no matches or no GPU context) **or** `BlendEngineV2.cb_store_final` — after SUBMITTED, before GPU work |

---

## Blend Server Events

These events use `session_id` on the `Event` dataclass (sourced from
`IPCCacheServerKey.request_id`) to correlate START/END pairs.

| EventType | Metadata keys | Types |
|---|---|---|
| `CB_LOOKUP_START` | `num_tokens` | `int` |
| `CB_LOOKUP_END` | `num_tokens`, `requested_tokens`, `hit_tokens`, `fingerprint_hits`, `storage_hits`, `stale_chunks`, `no_gpu_context` | `int`, `int`, `int`, `int`, `int`, `int`, `bool` |
| `CB_STORE_PRE_COMPUTED_START` | `instance_id`, `num_tokens` | `int`, `int` |
| `CB_STORE_PRE_COMPUTED_END` | `instance_id`, `num_tokens`, `stored_chunks`, `success` | `int`, `int`, `int`, `bool` |
| `CB_RETRIEVE_START` | `instance_id`, `num_chunks` | `int`, `int` |
| `CB_RETRIEVE_END` | `instance_id`, `num_chunks`, `success` | `int`, `int`, `bool` |
| `CB_STORE_FINAL_START` | `instance_id`, `num_tokens` | `int`, `int` |
| `CB_STORE_FINAL_END` | `instance_id`, `num_tokens`, `stored_chunks`, `success` | `int`, `int`, `int`, `bool` |
| `CB_FINGERPRINTS_REGISTERED` | `num_chunks`, `num_tokens` | `int`, `int` |
| `CB_CHUNKS_EVICTED` | `num_chunks` | `int` |
