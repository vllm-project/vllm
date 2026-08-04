# L2 Per-User Quota Design

This document describes the per-user quota mechanism for L2 adapters: how
per-user storage limits are enforced, how user identity propagates through the
system, and what changes are needed across the codebase.

## Motivation

Currently, L2 eviction operates on **aggregate** storage usage. A single
`trigger_watermark` governs whether eviction fires, and the LRU policy evicts
globally without regard to who stored what. In a multi-tenant serving
environment (multiple users sharing a single vLLM + LMCache deployment), one
user's burst of traffic can fill L2 and push out other users' cached KV data.

Per-user quotas add a second eviction dimension: **each user has an independent
storage budget**. When a user exceeds their budget, only that user's
least-recently-used keys are evicted — other users' cached data remains
untouched.

## Design Overview

```
vLLM API Server
  │  sends cache_salt directly on IPCCacheServerKey
  ▼
LMCache MP Server
  │  ipc_key_to_object_keys(key, chunk_hashes)
  │    reads key.cache_salt internally and propagates
  ▼
ObjectKey(chunk_hash, model_name, kv_rank, cache_salt="alice")
  │                                         ▲
  │                            cache_salt IS part of key identity
  │                            (participates in __eq__ / __hash__)
  │                            same tokens + different user = different key
  ▼
L1 Manager → StoreController → L2 Adapter
  │                                │
  │                    _notify_keys_stored(keys, sizes)
  │                      → base class updates _total_bytes_used
  │                        and _bytes_by_cache_salt
  ▼
Listeners:  L2EvictionPolicy bridge → IsolatedLRUEvictionPolicy
            ┌──────────────────────┐
            │ "alice" → OrderedDict │
            │ "bob"   → OrderedDict │
            └──────────────────────┘

L2EvictionController (every 1s):
  for each adapter state:
    usage = adapter.get_usage()  → AdapterUsage dataclass
    if policy.is_user_level:
      for cache_salt, bytes in usage.bytes_by_cache_salt:
        if bytes > watermark * quota(cache_salt):
          policy.get_eviction_actions(ratio, cache_salt=cache_salt)
    else:
      if usage.usage_fraction ≥ watermark:
        policy.get_eviction_actions(ratio)
```

## Key Design Decisions

### 1. Strict User Isolation — `cache_salt` as ObjectKey Identity Field

User identity is a **full identity field** on `ObjectKey`, participating in
`__eq__` and `__hash__`:

```python
@dataclass(frozen=True)
class ObjectKey:
    chunk_hash: bytes
    model_name: str
    kv_rank: int
    cache_salt: str = ""
```

Two ObjectKeys with the same (chunk_hash, model_name, kv_rank) but different
`cache_salt` values are **different keys**. If Alice and Bob send identical token
sequences, they produce separate ObjectKeys and store separate copies in both
L1 and L2.

**Why strict isolation?**

- **No cross-user interference:** Evicting Alice's keys never affects Bob's
  cache hits. Each user's cached data is fully independent.
- **Simple ownership:** Every key belongs to exactly one user — no ambiguity,
  no "first-storer-wins" races, no shared-ownership accounting.
- **Clean retrieval:** Lookup and retrieve naturally scope to the correct
  user because `cache_salt` is part of the key. No special filtering needed.
- **Predictable quotas:** Per-user byte accounting is exact. A user's usage
  equals the sum of their keys' sizes, with no shared entries to split.

**Trade-off — storage duplication:** The same token sequence cached by N
users consumes N times the storage (in both L1 and L2). This is the cost of
strict isolation. In practice, this is acceptable for multi-tenant deployments
where isolation guarantees outweigh storage efficiency, and where distinct
users typically have distinct prompts anyway.

### 2. Cache Salt Propagation: API → vLLM → LMCache

User identity is derived from vLLM's **`cache_salt`** field — a per-request
string already supported by the OpenAI-compatible API for prefix cache
isolation. No custom `kv_transfer_params` or `extra_args` are needed.

**Note:** In the context of per-user quotas, **one `cache_salt` value =
one user**. All requests sharing the same `cache_salt` are treated as
belonging to the same user and share a single quota and LRU list.
Different `cache_salt` values are fully isolated from each other.

**API caller** sets `cache_salt` per request:

```json
{
  "model": "llama-3-8b",
  "messages": [...],
  "cache_salt": "alice"
}
```

`cache_salt` already flows through vLLM as a first-class field:
`request body → vLLM input processor → Request.cache_salt`. The LMCache
MP connector (`lmcache_mp_connector.py` in the vLLM repo) stores
`cache_salt` on the request tracker and passes it through metadata to
both the scheduler and worker adapters.

**Propagation path:**

Both the scheduler adapter (LOOKUP) and worker adapter (STORE/RETRIEVE)
receive `cache_salt` from the vLLM connector via
`LMCacheMPRequestMetadata`. Both set `cache_salt` on `IPCCacheServerKey`
directly. No server-side session caching is needed.

```
API request body
  │  cache_salt = "alice"
  ▼
vLLM Request.cache_salt = "alice"     (first-class vLLM field)
  ▼
lmcache_mp_connector.py               (vLLM's LMCache connector)
  │  LMCacheMPRequestTracker stores request.cache_salt
  │  LMCacheMPRequestMetadata carries cache_salt to worker
  ▼
Scheduler path (LOOKUP):
  scheduler_adapter.maybe_submit_lookup_request(
      request_id, token_ids, cache_salt=tracker.cache_salt)
  → _create_key(..., cache_salt="alice")
  → IPCCacheServerKey(cache_salt="alice", ...)  ──LOOKUP──►  MP Server
                                                           │
Worker path (STORE/RETRIEVE):                              │
  worker_adapter.batched_submit_store_requests(            │
      request_ids, ops, event, cache_salts=["alice", ...])    │
  → _create_key(..., cache_salt="alice")                      │
  → IPCCacheServerKey(cache_salt="alice", ...) ──STORE──►  MP Server
                                                           │
                                              key.cache_salt = "alice"
                                              ipc_key_to_object_keys(key, hashes)
                                                — reads key.cache_salt
                                              → ObjectKey(cache_salt="alice", ...)
```

Because both scheduler and worker set `cache_salt` directly on the IPC key,
the server simply reads `key.cache_salt` in all code paths. No session-based
fallback is needed.

`cache_salt` is added as an identity field on `IPCCacheServerKey` (with
`compare=True`, unlike `request_id`). `request_id` is ephemeral session
metadata — two requests with different `request_id`s but the same tokens
should hit the same cache. `cache_salt` is the opposite: same tokens from
different users must **not** match. This is consistent with `cache_salt`
participating in `ObjectKey.__eq__` / `__hash__`.

```python
@dataclass(order=True, frozen=True)
class IPCCacheServerKey:
    model_name: str
    world_size: int
    worker_id: int | None
    token_ids: tuple[int, ...]
    start: int
    end: int
    request_id: str = field(compare=False)   # position 7 — unchanged
    cache_salt: str = ""                         # position 8 — appended at end
```

`cache_salt` is placed **after** `request_id` (at the end) to preserve
msgspec wire compatibility. `IPCCacheServerKey` is serialized positionally
via `msgspec.msgpack`; appending `cache_salt` at the end means an old client
sending 7 fields to a new server will decode correctly with `cache_salt`
defaulting to `""`. No changes to existing field positions.

`no_worker_id_version()` must be updated to preserve `cache_salt` when
copying the key (currently reconstructs explicitly and would drop the
new field).

### 3. Dynamic Per-User Quotas via `QuotaManager`

Per-user quotas are **dynamic** — they can be created, updated, and deleted
at runtime via the HTTP API. A `QuotaManager` holds the per-user limit
registry and is queried by the eviction controller each cycle.

**Quota lookup rules:**

1. If a cache_salt has an explicit entry in the quota registry → use that limit.
2. If a cache_salt has **no** entry → effective limit is **0 bytes**.
   - Stores are still allowed (we never reject writes on the hot path).
   - At the next eviction cycle (~1s), the controller sees
     `usage > 0 > limit 0` and triggers eviction.
   - The user's keys are evicted, freeing the space.

This means the system is **allowlist-based**: only users with an explicit
quota can retain cached data. Unknown users get temporary write access, but
their data is cleaned up within one eviction cycle.

Per-user quotas are enabled by choosing the `IsolatedLRU` eviction policy.
If the operator does not want per-user quotas, they simply use the `LRU`
policy — no special "disable" flag is needed.

### 4. HTTP API for Quota Management

The existing FastAPI HTTP server (`lmcache/v1/multiprocess/http_server.py`)
already serves `/healthcheck`, `/status`, and `/cache/clear`.
Add quota management endpoints:

```
PUT    /quota/{cache_salt}          Set/update quota for a user
GET    /quota/{cache_salt}          Get quota and current usage for a user
DELETE /quota/{cache_salt}          Remove quota (user's data evicted next cycle)
GET    /quota                    List all quotas and per-user usage
```

**`_default` sentinel:** Empty strings cannot be URL path parameters. Use
`_default` as the `cache_salt` in the URL to refer to the `cache_salt=""`
namespace (anonymous / un-isolated traffic). For example,
`PUT /quota/_default` sets the quota for `cache_salt=""`.

**`PUT /quota/{cache_salt}`** — Set or update a user's quota.
`limit_gb` is required.

```json
// Request body
{"limit_gb": 2.0}

// Response
{"cache_salt": "alice", "limit_gb": 2.0, "status": "ok"}
```

**`GET /quota/{cache_salt}`** — Get quota and current usage.

```json
// Response
{
  "cache_salt": "alice",
  "limit_gb": 2.0,
  "current_usage_gb": 1.3,
  "exists": true
}
```

**`DELETE /quota/{cache_salt}`** — Remove quota entry. The user's cached
data will be evicted at the next eviction cycle (effective limit becomes 0).

```json
// Response
{"cache_salt": "alice", "status": "removed"}
```

**`GET /quota`** — List all registered quotas with per-user usage.

```json
// Response
{
  "users": {
    "alice": {"limit_gb": 2.0, "current_usage_gb": 1.3},
    "bob":   {"limit_gb": 5.0, "current_usage_gb": 4.1}
  }
}
```

### 5. Policy Selection

Set `eviction_policy: "IsolatedLRU"` in the adapter's eviction config to
enable per-user quotas. `"LRU"` retains existing aggregate-only behavior.
See the **Configuration** section for the full JSON example.

## Component Changes

### 1. `ObjectKey` and `ipc_key_to_object_keys()`

**File:** `lmcache/v1/distributed/api.py`

Add `cache_salt: str = ""` to `ObjectKey` (as shown in section 1).
`ipc_key_to_object_keys()` reads `ipc_key.cache_salt` directly and
propagates it to each constructed `ObjectKey` — no separate parameter,
so callers cannot accidentally drop the salt.

### 2. Server — `cache_salt` is carried by `IPCCacheServerKey`

**File:** `lmcache/v1/multiprocess/server.py`

Since both the scheduler and worker adapters set `cache_salt` on
`IPCCacheServerKey`, the server simply calls `ipc_key_to_object_keys(...)`
and the salt flows through automatically:

```python
obj_keys = ipc_key_to_object_keys(key, chunk_hashes)
```

**`session.py` is unchanged** — no `cache_salt` field needed on `Session`.

### 3. vLLM Connector & Adapter Layer

**File (vLLM repo):**
`vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py`

The MP connector stores `cache_salt` on the request tracker and
propagates it through metadata to both scheduler and worker:

```python
# LMCacheMPRequestTracker.__init__:
self.cache_salt: str = request.cache_salt or ""

# LMCacheMPRequestMetadata: add field
cache_salt: str = ""

# GetStoreMetadata / GetRetrieveMetadata: copy from tracker
cache_salt=tracker.cache_salt

# Scheduler LOOKUP call:
self.scheduler_adapter.maybe_submit_lookup_request(
    request.request_id,
    token_ids=list(request.all_token_ids),
    cache_salt=tracker.cache_salt,
)

# Worker start_load_kv / wait_for_save:
# Pass cache_salts alongside request_ids
cache_salts = [meta.cache_salt for meta in metadata.requests if ...]
self.worker_adapter.batched_submit_store_requests(
    request_ids, ops, event, cache_salts=cache_salts)
self.worker_adapter.batched_submit_retrieve_requests(
    request_ids, ops, event, cache_salts=cache_salts)
```

**File (LMCache repo):**
`lmcache/integration/vllm/vllm_multi_process_adapter.py`

**Scheduler adapter** (LOOKUP path):

- `maybe_submit_lookup_request(request_id, token_ids, cache_salt="")`
- `free_lookup_locks(..., cache_salt="")`
- `_create_key(..., cache_salt="")` — passes `cache_salt` to `IPCCacheServerKey`

**Worker adapter** (STORE/RETRIEVE path):

- `batched_submit_store_requests(request_ids, ops, event, cache_salts=None)`
- `batched_submit_retrieve_requests(request_ids, ops, event, cache_salts=None)`
- `submit_store_request(request_id, op, event, cache_salt="")`
- `submit_retrieve_request(request_id, op, event, cache_salt="")`
- `_create_key(..., cache_salt="")` — passes `cache_salt` to `IPCCacheServerKey`

Both adapters' `_create_key()` passes `cache_salt` through:

```python
def _create_key(self, token_ids, start, end, request_id, cache_salt=""):
    return IPCCacheServerKey(
        model_name=self.model_name,
        world_size=self.world_size,
        worker_id=...,
        token_ids=tuple(token_ids),
        start=start, end=end,
        request_id=request_id,
        cache_salt=cache_salt,
    )
```

**Non-MP adapters (`vllm_v1_adapter.py`, lookup clients) are unchanged.**
Per-user quota is an MP-mode-only feature.

### 4. Unified Usage Tracking in the Adapter Base Class

**File:** `lmcache/v1/distributed/l2_adapters/base.py`

All byte tracking — both aggregate and per-user — lives in the base class
and is updated exclusively in `_notify_keys_stored` / `_notify_keys_deleted`.
Adapters do not maintain their own byte counters. They pass `sizes` to
the `_notify_*` helpers and the base class does all accounting.

A new `AdapterUsage` dataclass replaces both the old `get_usage()` (which
returned a `tuple[float, float]`) and `get_per_user_usage()` (which
returned a `dict`) with a single structured report:

```python
@dataclass(frozen=True)
class AdapterUsage:
    """Unified usage report for an L2 adapter."""

    total_bytes_used: int
    """Aggregate bytes across all users."""

    total_capacity_bytes: int
    """Adapter's maximum capacity. 0 means unknown/unlimited."""

    bytes_by_cache_salt: dict[str, int]
    """Bytes used per cache_salt. Only entries with positive usage."""

    @property
    def usage_fraction(self) -> float:
        """Aggregate usage as a fraction in [0, 1]. -1 if capacity unknown."""
        if self.total_capacity_bytes <= 0:
            return -1.0
        return self.total_bytes_used / self.total_capacity_bytes
```

Base class changes:

```python
class L2AdapterInterface(ABC):
    def __init__(self, max_capacity_bytes: int = 0):
        self._listeners: list[L2AdapterListener] = []
        self._max_capacity_bytes = max_capacity_bytes
        self._total_bytes_used: int = 0
        self._bytes_by_cache_salt: dict[str, int] = {}
        self._usage_lock = threading.Lock()

    @property
    def supports_global_eviction(self) -> bool:
        """Whether this adapter supports eviction.

        True when the adapter declared a positive capacity via
        max_capacity_bytes. Adapters that don't support eviction
        (e.g., FSL2Adapter) pass 0 and this returns False.
        """
        return self._max_capacity_bytes > 0

    def _notify_keys_stored(
        self, keys: list[ObjectKey], sizes: list[int]
    ) -> None:
        with self._usage_lock:
            for key, size in zip(keys, sizes, strict=True):
                self._total_bytes_used += size
                self._bytes_by_cache_salt[key.cache_salt] = (
                    self._bytes_by_cache_salt.get(key.cache_salt, 0) + size
                )
        for listener in self._listeners:
            listener.on_l2_keys_stored(keys)

    def _notify_keys_deleted(
        self, keys: list[ObjectKey], sizes: list[int]
    ) -> None:
        with self._usage_lock:
            for key, size in zip(keys, sizes, strict=True):
                self._total_bytes_used -= size
                self._bytes_by_cache_salt[key.cache_salt] = (
                    self._bytes_by_cache_salt.get(key.cache_salt, 0) - size
                )
        for listener in self._listeners:
            listener.on_l2_keys_deleted(keys)

    def get_usage(self) -> AdapterUsage:
        with self._usage_lock:
            return AdapterUsage(
                total_bytes_used=self._total_bytes_used,
                total_capacity_bytes=self._max_capacity_bytes,
                bytes_by_cache_salt={
                    k: v for k, v in self._bytes_by_cache_salt.items()
                    if v > 0
                },
            )
```

`_notify_keys_accessed(keys)` is unchanged — it does not affect byte
counts. The external `L2AdapterListener` interface still receives only
keys — `sizes` is internal to the adapter/base-class boundary.

### 5. Adapter Implementations

Each adapter passes `max_capacity_bytes` to `super().__init__()` and
fires `_notify_keys_stored(keys, sizes)` / `_notify_keys_deleted(keys,
sizes)`. The base class handles all byte accounting.

**What each adapter removes:**
- `_current_size_bytes` counter
- `get_usage()` override
- `get_per_user_usage()` override (if any)

**What each adapter adds/keeps:**
- `super().__init__(max_capacity_bytes=...)` in `__init__`
- `_notify_keys_stored(stored_keys, sizes=stored_sizes)` after store
- `_notify_keys_deleted(deleted_keys, sizes=deleted_sizes)` after delete

| Adapter | `max_capacity_bytes` source | `supports_global_eviction` |
|---------|-----------------------------|---------------------|
| MockL2Adapter | `int(config.max_size_gb * 1024**3)` | `True` |
| NixlStoreL2Adapter | Pool total size | `True` |
| RawBlockL2Adapter | RawBlockCore usable capacity | `True` |
| NativeConnectorL2Adapter | `int(max_capacity_gb * 1024**3)` | `True` |
| FSL2Adapter | 0 | `False` |

`L2AdapterEvictionState` is only created for adapters where both
`eviction_config is not None` AND `adapter.supports_global_eviction` are true.
Adapters that don't support eviction are excluded from the eviction loop
entirely — no runtime checks needed.

Future adapters inherit all tracking automatically — just call
`super().__init__(max_capacity_bytes=...)` and fire `_notify_*` with
sizes.

### 6. `EvictionPolicy` — `is_user_level` property

**File:** `lmcache/v1/distributed/eviction.py`

Add a property to the abstract base class. The eviction controller uses
this to decide whether to check per-user quotas or aggregate usage —
no `isinstance` checks.

```python
class EvictionPolicy:
    @property
    def is_user_level(self) -> bool:
        """Whether this policy supports per-user eviction.

        When True, the controller checks per-user usage and passes
        cache_salt to get_eviction_actions(). When False, the
        controller uses aggregate usage only.
        """
        return False
```

| Policy | `is_user_level` |
|--------|----------------|
| `LRUEvictionPolicy` | `False` (inherits default) |
| `NoOpEvictionPolicy` | `False` (inherits default) |
| `IsolatedLRUEvictionPolicy` | `True` (override) |

### 7. `IsolatedLRUEvictionPolicy` — Per-user LRU tracking

**File (new):** `lmcache/v1/distributed/eviction_policy/isolated_lru.py`

Overrides `is_user_level` to return `True`. `get_eviction_actions` gains
an optional `cache_salt` parameter. When set, eviction is scoped to that
user's LRU list. When `None` (default), eviction is global.

```python
class IsolatedLRUEvictionPolicy(EvictionPolicy):

    @property
    def is_user_level(self) -> bool:
        return True

    def __init__(
        self,
        default_destination: EvictionDestination = EvictionDestination.DISCARD,
    ):
        self._lock = threading.Lock()
        self._per_user_order: dict[str, OrderedDict[ObjectKey, None]] = {}
        self._default_destination = default_destination

    def on_keys_created(self, keys: list[ObjectKey]):
        with self._lock:
            for key in reversed(keys):
                cache_salt = key.cache_salt
                if cache_salt not in self._per_user_order:
                    self._per_user_order[cache_salt] = OrderedDict()
                user_order = self._per_user_order[cache_salt]
                if key in user_order:
                    user_order.move_to_end(key)
                else:
                    user_order[key] = None

    def on_keys_touched(self, keys: list[ObjectKey]):
        with self._lock:
            for key in reversed(keys):
                cache_salt = key.cache_salt
                user_order = self._per_user_order.get(cache_salt)
                if user_order and key in user_order:
                    user_order.move_to_end(key)

    def on_keys_removed(self, keys: list[ObjectKey]):
        with self._lock:
            for key in keys:
                cache_salt = key.cache_salt
                user_order = self._per_user_order.get(cache_salt)
                if user_order:
                    user_order.pop(key, None)
                    if not user_order:
                        del self._per_user_order[cache_salt]

    def get_eviction_actions(
        self,
        expected_ratio: float,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        """Select victims, optionally scoped to a user.

        Args:
            expected_ratio: Fraction of keys to evict.
            cache_salt: If set, evict from this user's list only.
                If None, evict globally across all users.
        """
        with self._lock:
            if cache_salt is not None:
                order = self._per_user_order.get(cache_salt)
                if not order:
                    return []
                pool = list(order.keys())
            else:
                pool = []
                for user_order in self._per_user_order.values():
                    pool.extend(user_order.keys())

            if not pool:
                return []

            expected_ratio = max(0.0, min(1.0, expected_ratio))
            target = int(len(pool) * expected_ratio)
            if expected_ratio > 0 and target == 0 and len(pool) > 0:
                target = 1
            if target == 0:
                return []

            return [EvictionAction(
                keys=pool[:target],
                destination=self._default_destination,
            )]
```

**`EvictionPolicy` abstract class** — add `cache_salt: str | None = None`
to `get_eviction_actions`. Existing implementations (`LRUEvictionPolicy`,
`NoOpEvictionPolicy`) accept and ignore it — backward compatible.

### 8. L2 Eviction Controller — Per-user eviction trigger

**File:** `lmcache/v1/distributed/storage_controllers/eviction_controller.py`

The controller receives a reference to the `QuotaManager`. Each eviction
cycle it finds **all** users who violate their watermark threshold and
evicts from each of them. After one cycle, no user should be violating.

```python
class L2EvictionController(StorageControllerInterface):
    def __init__(
        self,
        l2_adapter_states: list[L2AdapterEvictionState],
        quota_manager: QuotaManager,
    ):
        self._adapter_states = l2_adapter_states
        self._quota_manager = quota_manager
        ...

    def _check_and_evict(self, state: L2AdapterEvictionState):
        watermark = state.eviction_config.trigger_watermark
        eviction_ratio = state.eviction_config.eviction_ratio
        policy = state.eviction_policy
        usage = state.adapter.get_usage()

        if policy.is_user_level and self._quota_manager:
            # Per-user watermark check
            for cache_salt, user_bytes in usage.bytes_by_cache_salt.items():
                limit = self._quota_manager.get_limit_bytes(cache_salt)
                if user_bytes <= watermark * limit:
                    continue
                effective_ratio = 1.0 if limit == 0 else eviction_ratio
                actions = policy.get_eviction_actions(
                    effective_ratio, cache_salt=cache_salt
                )
                for action in actions:
                    self._execute_eviction_action(state.adapter, action)
        else:
            # Global aggregate watermark
            if usage.usage_fraction < watermark:
                return
            actions = policy.get_eviction_actions(eviction_ratio)
            for action in actions:
                self._execute_eviction_action(state.adapter, action)
```

The controller uses `policy.is_user_level` (not `isinstance`) to branch:
- **`is_user_level=True`**: reads `usage.bytes_by_cache_salt`, checks each user
  against `watermark * quota`. Unregistered users (quota=0) get ratio=1.0.
- **`is_user_level=False`**: reads `usage.usage_fraction` for global check.

## Configuration

### Example: L2 adapter with per-user quota

```json
{
  "type": "mock",
  "max_size_gb": 10,
  "mock_bandwidth_gb": 4,
  "eviction": {
    "eviction_policy": "IsolatedLRU",
    "trigger_watermark": 0.8,
    "eviction_ratio": 0.2
  }
}
```

| Field               | Type    | Default | Description                                           |
|---------------------|---------|---------|-------------------------------------------------------|
| `eviction_policy`   | string  | —       | `"LRU"`, `"IsolatedLRU"`, or `"noop"`. Required.         |
| `trigger_watermark` | float   | `0.8`   | Usage fraction to trigger eviction. For `LRU`: against aggregate capacity. For `IsolatedLRU`: against each user's quota. |
| `eviction_ratio`    | float   | `0.2`   | Fraction of keys to evict each cycle.                 |

### Usage examples

```bash
# Send a request with user identity
curl -X POST http://vllm:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3-8b", "messages": [...], "cache_salt": "alice"}'

# Manage quotas at runtime
curl -X PUT http://localhost:8000/quota/alice \
  -H "Content-Type: application/json" -d '{"limit_gb": 2.0}'
curl http://localhost:8000/quota/alice
curl -X DELETE http://localhost:8000/quota/alice
curl http://localhost:8000/quota
```

## Behavioral Notes

- **No cache_salt from API:** When the API caller doesn't set `cache_salt`,
  `request.cache_salt` is `None`, which maps to `cache_salt=""`.
  `IPCCacheServerKey.cache_salt` defaults to `""`. All such keys share the
  same (anonymous) namespace.
- **`eviction_policy: "LRU"`:** Per-user quota logic is not active. The
  watermark is applied against aggregate capacity as before. Existing
  behavior is fully unchanged.
- **ObjectKey equality:** `cache_salt` is part of identity (eq/hash). Two
  `ObjectKey`s with different salts are distinct, preventing cross-user
  collisions. With the default `cache_salt=""` all un-salted traffic
  hashes identically (unchanged from the pre-cache_salt adapter).
- **Serialization format — trailing salt:** ``cache_salt`` is appended as
  a 4th field when non-empty; unsalted keys use the 3-field shape, which
  is bit-identical to the pre-cache_salt format:

  ```python
  def _object_key_to_string(key: ObjectKey) -> str:
      base = f"{key.model_name}@{key.kv_rank:08x}@{key.chunk_hash.hex()}"
      if key.cache_salt:
          return f"{base}@{key.cache_salt}"
      return base
  ```

  Because un-salted wire keys and filenames are unchanged, existing
  cache directories and remote stores need **no migration**. Parsers
  split on ``@`` and dispatch by field count (3 vs 4); both
  ``model_name`` and ``cache_salt`` are forbidden from containing
  ``@`` so the parse is unambiguous.

- **Listener interface:** `L2AdapterListener` method signatures are unchanged.
  `cache_salt` flows through `ObjectKey.cache_salt`, not through callback
  parameters.

## PR Plan

The implementation is split into 6 PRs to keep each reviewable and
independently mergeable. **Merge order matters** for wire compatibility.

```
PR1a (LMCache adapter) ──► PR1b (vLLM connector)
                                  │
PR2  (LMCache data model) ────────┤
                                  ├──► PR5 (per-user LRU)
PR3  (LMCache adapter interface) ─┤
                                  │
PR4  (LMCache eviction policy) ───┘
```

PR1a, PR2, PR3, PR4 are independent — can merge in any order.
PR1b depends on PR1a (LMCache must accept `cache_salt` before vLLM sends it).
PR5 depends on all others.

### PR1a — LMCache: Adapter `cache_salt` plumbing (LMCache repo)

Add `cache_salt=""` defaults to scheduler and worker adapter interfaces.
No-op with defaults — safe to merge independently.

| File | Change |
|------|--------|
| `lmcache/integration/vllm/vllm_multi_process_adapter.py` | Scheduler: `cache_salt=""` on `maybe_submit_lookup_request`, `free_lookup_locks`, `_create_key`. Worker: `cache_salt=""` on `submit_store_request`, `submit_retrieve_request`, `batched_submit_*`, `_create_key`. |

### PR1b — vLLM: Connector propagates `cache_salt` (vLLM repo)

Populate `cache_salt` from `request.cache_salt` and pass to adapters.

| File | Change |
|------|--------|
| `vllm/.../lmcache_mp_connector.py` | Store `cache_salt` on `LMCacheMPRequestTracker`; add `cache_salt` to `LMCacheMPRequestMetadata`; pass to scheduler + worker adapter methods |

### PR2 — LMCache: `cache_salt` on data model + server (LMCache repo)

Add `cache_salt` to `ObjectKey` and `IPCCacheServerKey`. Server passes
it through. Update serialization. No behavioral change with `cache_salt=""`.

| File | Change |
|------|--------|
| `lmcache/v1/distributed/api.py` | `cache_salt: str = ""` on `ObjectKey` with `__post_init__` validation (`@`, `/`, `\`, NUL, length cap on `cache_salt`; `@` rejected on `model_name`); `ipc_key_to_object_keys()` reads `ipc_key.cache_salt` directly (no separate param) |
| `lmcache/v1/multiprocess/custom_types.py` | `cache_salt: str = ""` on `IPCCacheServerKey` with `__post_init__` validation; update `no_worker_id_version()`, `from_token_ids()` |
| `lmcache/v1/multiprocess/server.py` / `blend_server_v2.py` | No code changes — existing `ipc_key_to_object_keys(key, chunk_hashes)` calls now carry salt automatically |
| `lmcache/integration/vllm/vllm_multi_process_adapter.py` | Scheduler + worker `_create_key()` now forward `cache_salt` to `IPCCacheServerKey` |
| `lmcache/v1/distributed/l2_adapters/native_connector_l2_adapter.py` | `_object_key_to_string()` appends trailing `@<cache_salt>` when salted; un-salted output is unchanged |
| `lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py` | `_object_key_to_filename()` / `_filename_to_object_key()` accept 3-field (unsalted) or 4-field (salted) shapes |
| `csrc/storage_backends/fs/connector.cpp` | `key_to_filename()` splits on `@` and dispatches by field count |

### PR3 — LMCache: Adapter interface refactor (LMCache repo)

`AdapterUsage` dataclass, `supports_global_eviction` property, unified
`get_usage()`, `_notify_*` with `sizes`. Purely internal refactor —
existing LRU eviction behavior unchanged.

| File | Change |
|------|--------|
| `lmcache/v1/distributed/l2_adapters/base.py` | `AdapterUsage` dataclass; `max_capacity_bytes` + `supports_global_eviction`; `_notify_*` with `sizes`; unified `get_usage() -> AdapterUsage` |
| `lmcache/v1/distributed/l2_adapters/mock_l2_adapter.py` | Pass `max_capacity_bytes` to super; pass `sizes` to `_notify_*`; remove `_current_size_bytes`, `get_usage()` |
| `lmcache/v1/distributed/l2_adapters/nixl_store_l2_adapter.py` | Same |
| `lmcache/v1/distributed/l2_adapters/native_connector_l2_adapter.py` | Same |
| `lmcache/v1/distributed/l2_adapters/mooncake_store_l2_adapter.py` | Same (if fires `_notify_*` directly) |
| `lmcache/v1/distributed/storage_controllers/eviction_controller.py` | Use `AdapterUsage.usage_fraction` instead of tuple |
| `lmcache/v1/distributed/storage_manager.py` | Filter `L2AdapterEvictionState` by `adapter.supports_global_eviction` |

### PR4 — LMCache: Eviction policy interface (LMCache repo)

`is_user_level` property, `cache_salt` param on `get_eviction_actions`.
No new policies yet — just the interface extension.

| File | Change |
|------|--------|
| `lmcache/v1/distributed/eviction.py` | `is_user_level` property (default `False`); `cache_salt` param on `get_eviction_actions()` |
| `lmcache/v1/distributed/eviction_policy/lru.py` | Accept (ignore) `cache_salt` |
| `lmcache/v1/distributed/eviction_policy/noop.py` | Accept (ignore) `cache_salt` |

### PR5 — LMCache: Per-user LRU eviction (LMCache repo)

The feature PR. Depends on PR1a + PR1b + PR2 + PR3 + PR4.

| File | Change |
|------|--------|
| `lmcache/v1/distributed/eviction_policy/isolated_lru.py` (new) | `IsolatedLRUEvictionPolicy` |
| `lmcache/v1/distributed/quota_manager.py` (new) | `QuotaManager` |
| `lmcache/v1/distributed/eviction_policy/factory.py` | Register `"IsolatedLRU"` |
| `lmcache/v1/distributed/eviction_policy/__init__.py` | Export `IsolatedLRUEvictionPolicy` |
| `lmcache/v1/distributed/config.py` | Add `"IsolatedLRU"` to literal |
| `lmcache/v1/distributed/l2_adapters/config.py` | Add `"IsolatedLRU"` to allowed values |
| `lmcache/v1/distributed/storage_controllers/eviction_controller.py` | `QuotaManager`; per-user branch using `is_user_level` + `bytes_by_cache_salt` |
| `lmcache/v1/distributed/storage_manager.py` | Create `QuotaManager`; wire to controller + HTTP |
| `lmcache/v1/multiprocess/http_server.py` | Quota CRUD endpoints |
| `tests/v1/distributed/test_isolated_lru_eviction_policy.py` (new) | Unit tests |
| `tests/v1/distributed/test_quota_manager.py` (new) | Unit tests |
| `tests/v1/distributed/test_per_user_l2_eviction.py` (new) | Integration tests |
