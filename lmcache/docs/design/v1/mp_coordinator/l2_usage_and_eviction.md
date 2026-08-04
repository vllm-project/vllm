# Fleet-Wide L2 Usage Tracking and Eviction

A coordinator-level capability that gives fleet-wide visibility into per-tenant
L2 cache usage and enforces per-``cache_salt`` byte quotas via LRU eviction.
MP servers **report store/lookup events** to the coordinator; the coordinator
aggregates usage, manages quotas, and periodically selects LRU keys to evict
when a tenant exceeds its quota. It is **opt-in** (gated by
``event_reporting`` in ``CoordinatorConfig``, shared with the key-directory stream) and **additive** (the existing
per-server eviction is unchanged).

Code: `lmcache/v1/mp_coordinator/cache_control/` (coordinator side),
`lmcache/v1/mp_coordinator/http_apis/cache_api.py` (REST endpoints),
`lmcache/v1/mp_coordinator/schemas.py` (wire types),
`lmcache/v1/multiprocess/http_server.py` (MP-server wiring).

## Why

L2 eviction today is **local to each MP server**: the ``IsolatedLRUEvictionPolicy``
tracks only what that server stored and enforces quotas within that scope. With
a shared L2 backend (e.g. S3), multiple servers write to the same storage, but
no single server has a fleet-wide view of total per-tenant usage. The coordinator
centralizes usage accounting and quota enforcement so limits apply to the
aggregate, not per-replica.

## Architecture

```
MP server (store/lookup)
  L2 adapter fires on_l2_keys_stored / on_l2_keys_accessed
        │
        ▼
  L2EventListener (L2AdapterListener)
    converts ObjectKey → EncodedObjectKey, buffers UsageEvents
        │  flush every event_flush_interval (default 1s)
        │
        ▼
  POST /quota/events ──▶ Coordinator
                        ├─ L2UsageManager: per-salt byte accounting
                        ├─ L2EvictionManager: per-salt LRU
                        └─ QuotaManager: per-salt byte limits

  Coordinator background loop (every eviction_check_interval, default 5s)
        │
        ▼
  execute_evictions():
    for each tracked salt:
      limit = quota (default 0 → evict all)
      if usage ≥ watermark·limit → select LRU keys,
        fire-and-forget DELETE /cache/objects to a holder
```

## Wire types (`schemas.py`)

- **``EncodedObjectKey``** — torch-free wire shape of ``ObjectKey`` (owned by
  ``lmcache.v1.distributed.api``); ``chunk_hash`` is hex-encoded instead of raw
  bytes. The coordinator rebuilds the canonical ``ObjectKey`` via
  ``key.to_object_key()``.
- **``EventType``** — ``str`` enum: ``STORE``, ``LOOKUP``, ``DELETE``.
- **``UsageEvent``** — ``type: EventType``, ``key: EncodedObjectKey``,
  ``bytes: int``.
- **``ReportUsageRequest``** — ``instance_id``, ``seq``, ``events:
  list[UsageEvent]``, ``tier`` (data, default ``l2``).

The ``ObjectKey`` → ``EncodedObjectKey`` conversion happens at the MP-server
boundary (``obj.to_encoded_object_key()`` in ``event_listener.py``), so the
coordinator never imports ``torch``.

## Coordinator components (`cache_control/`)

### L2UsageManager (`usage_manager.py`)

Thread-safe per-salt byte counter. Two operations:

- ``record_stored(cache_salt, n_bytes)`` — increment.
- ``record_evicted(cache_salt, n_bytes)`` — decrement (clamped at zero).

Exposes ``get(salt)``, ``get_all()``, ``get_total()`` for the status endpoints.

### QuotaManager (reused from ``lmcache.v1.distributed.quota_manager``)

Thread-safe in-memory quota registry (``dict[str, int]`` + lock). CRUD via
``set_quota``, ``get_limit_bytes``, ``delete_quota``, ``list_quotas``.
Quotas are set in GiB at the API and stored as bytes internally.

Unregistered salts resolve through a configurable **default limit**
(``set_default_limit_bytes`` / ``effective_limit_bytes``), which starts as ``None``:

- ``None`` (boot default) — unregistered salts are **exempt** from coordinator
  eviction. Quotas live in memory, so a restarted coordinator has an empty
  table until the external quota controller re-syncs it; the exempt default
  makes that window safe (an empty table cannot mass-evict unknown tenants).
- ``0`` — strict allowlist: all bytes under unregistered salts become
  evictable next cycle. The external controller sets this via
  ``PUT /quota/config`` **after** re-registering every per-salt quota — it is
  the explicit signal that arms fleet-wide eviction of unquota'd data.
- ``> 0`` — every unregistered salt gets that byte budget.

The MP server's local eviction controller keeps the legacy
``get_limit_bytes`` path (unregistered ⇒ 0) and is unaffected by the default.
Because the server-local and coordinator quota registries are independent and
never synchronized, a deployment must register quotas through **one** of the
two ``/quota`` APIs, never both — the server-side enforcer fully evicts any
salt missing from its own table, so it would fight quotas registered only on
the coordinator (see the warning in ``docs/source/mp/coordinator.rst``).

### L2EvictionManager (`eviction_manager.py`)

Per-``cache_salt`` LRU for the coordinator process. It delegates the ordering to
a coordinator-side ``IsolatedLRUEvictionPolicy`` instance, keyed by the canonical
``ObjectKey`` (rebuilt from the wire ``EncodedObjectKey``). Per-salt byte
accounting lives in ``L2UsageManager``; the eviction manager only tracks order.

- ``on_store(key)`` — register the key in the LRU
  (``policy.on_keys_created``). The paired byte increment is the caller's job
  (``L2UsageManager.record_stored``).
- ``on_lookup(key)`` — touch (``policy.on_keys_touched``, move to MRU end).
- ``on_remove(key)`` — drop from the LRU (``policy.on_keys_removed``); the paired
  byte decrement is the caller's job (``L2UsageManager.record_evicted``).
- ``compute_eviction_plan() -> dict[str, list[ObjectKey]]`` — **pure**: for each
  tracked salt, fire when ``usage ≥ trigger_watermark · quota`` (quota 0 ⇒ evict
  all), selecting ``eviction_ratio`` of the salt's LRU keys via
  ``policy.get_eviction_actions``. Salts without an explicit quota use the
  registry's default limit; while it is unset (``None``) they are skipped
  entirely — see QuotaManager above. No network, no mutation.
- ``execute_evictions(registry, http_client)`` — computes the plan and
  **fire-and-forget** ``DELETE /cache/objects`` to a holder MP server for each
  salt's victims; on confirmed deletion ``on_remove`` drops them from tracking.
  The plan's keys are split into chunks of at most ``MAX_DELETE_BATCH`` (imported
  from the MP server's ``object_service``) and each chunk is dispatched as its
  own request, because the endpoint rejects any single delete over that cap with
  HTTP 400 — a full-salt eviction (quota dropped to 0) routinely exceeds it.

## REST endpoints (`quota_api.py`)

| Method | Path | Description |
| --- | --- | --- |
| ``PUT`` | ``/quota/config`` | Set the default limit for unquota'd salts (``null`` ⇒ exempt; ``0`` arms allowlist eviction) |
| ``GET`` | ``/quota/config`` | Read the default limit |
| ``PUT`` | ``/quota/{cache_salt}`` | Set quota (GiB) |
| ``DELETE`` | ``/quota/{cache_salt}`` | Remove quota |
| ``GET`` | ``/quota/{cache_salt}`` | Quota + usage for one salt |
| ``GET`` | ``/quota`` | Quota + usage for all salts |
| ``POST`` | ``/quota/events`` | Ingest batch of store/lookup/delete events |

The ``/quota/config`` routes are declared before ``/quota/{cache_salt}`` so the
literal ``config`` segment is not captured as a salt. Expected controller flow
after a coordinator (re)start: ``PUT /quota/{salt}`` for every tenant, then
``PUT /quota/config {"default_limit_gb": 0}`` to arm eviction of everything
else.

These quota/usage-accounting endpoints live in the ``/quota`` group (mirroring
the MP server's node-local ``/quota``); warm-prefetch dispatch is the only thing
left on the coordinator's ``/cache/*`` surface (``cache_api.py``). Paths are
tier-neutral; the tier is request data (`tier`, default `l2`). Status responses
report usage in GiB only (no raw bytes in the API).

## MP-server event listener (`event_listener.py`)

``L2EventListener`` implements ``L2AdapterListener`` and is registered
on all L2 adapters via ``StorageManager.register_l2_listener()``. It:

1. Receives ``on_l2_keys_stored(keys, sizes)``, ``on_l2_keys_accessed(keys)``,
   and ``on_l2_keys_deleted(keys)`` callbacks from the L2 adapter (any thread).
2. Converts each ``ObjectKey`` to ``EncodedObjectKey`` (hex-encodes
   ``chunk_hash``).
3. Buffers ``UsageEvent``s under a lock.
4. Flushes the buffer to ``POST /quota/events`` on a timer
   (``event_flush_interval``, default 1s). Failures are logged and the
   batch is dropped to prevent unbounded growth.

``on_l2_keys_deleted`` buffers a ``DELETE`` event so the coordinator can drop
the key from its usage accounting and LRU tracking.

## Configuration

### Coordinator side (`MPCoordinatorConfig`)

| Field | Default | Description |
| --- | --- | --- |
| ``eviction_check_interval`` | ``5.0`` | Seconds between eviction cycles (0 disables) |
| ``eviction_ratio`` | ``0.2`` | Fraction of a salt's LRU keys (by count) to evict per cycle |
| ``trigger_watermark`` | ``1.0`` | Eviction fires when usage reaches this fraction of the quota |

### MP-server side (`CoordinatorConfig`)

| Field | Default | Env var | Description |
| --- | --- | --- | --- |
| ``event_reporting`` | ``False`` | ``LMCACHE_COORDINATOR_EVENT_REPORTING`` | Enable event reporting (also gates the key-directory stream) |
| ``event_flush_interval`` | ``1.0`` | ``LMCACHE_COORDINATOR_EVENT_FLUSH_INTERVAL`` | Seconds between flushes |

Both also accept CLI flags (``--coordinator-event-reporting``,
``--coordinator-event-flush-interval``).

## Failure modes

| Event | Effect | Handling |
| --- | --- | --- |
| Coordinator down | Events not delivered | Flush fails → batch dropped, logged; MP server unaffected |
| Coordinator restart | Usage/LRU state lost | Rebuilt from incoming events; stale until servers report |
| Flush timeout | One batch delayed | Next flush sends new batch; no retry of old batch |
| Usage accounting drift | Quota enforcement imprecise | Self-correcting as new events arrive |

## Scope

Additive: no change to per-server eviction, L2 adapter store/lookup paths, or
the coordinator's membership/health loop. Composes via the ``L2AdapterListener``
interface and the ``http_apis`` auto-discovery — a new router reading
``app.state``, plus the opt-in event listener — with no edits to existing
controllers or adapters beyond passing ``sizes`` through to
``on_l2_keys_stored``.
