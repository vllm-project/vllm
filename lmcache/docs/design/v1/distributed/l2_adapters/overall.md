# L2 Store and Prefetch Controller Design

This document describes how the StoreController and PrefetchController interact
with L2 adapters, the invariants they maintain, and the assumptions they rely on.
It is intended for developers implementing new L2 adapters or modifying the
controller logic.

## Architecture Overview

```
                    ┌────────────────────────┐
                    │    StorageManager       │
                    │  submit_prefetch_task   │
                    │  query_prefetch_status  │
                    │  reserve/finish_write   │
                    └────┬──────────┬─────────┘
                         │          │
              ┌──────────┘          └──────────┐
              ▼                                ▼
   ┌────────────────────┐           ┌────────────────────┐
   │  StoreController   │           │ PrefetchController  │
   │  (background thread)│          │  (background thread) │
   │                    │           │                     │
   │  L1 write done     │           │  external submit    │
   │   → store to L2    │           │   → lookup L2       │
   │   → release locks  │           │   → plan + load     │
   └────┬───────────────┘           │   → read-lock L1    │
        │                           └──────┬──────────────┘
        │                                  │
        ▼                                  ▼
   ┌─────────────────────────────────────────────┐
   │           L2AdapterInterface(s)             │
   │   store / lookup_and_lock / load / unlock   │
   │                                             │
   │  Each adapter has 3 distinct event fds:     │
   │   store_efd, lookup_efd, load_efd           │
   └─────────────────────────────────────────────┘
```

Both controllers run a single background thread each, using `select.poll()`
on eventfds for event-driven I/O. They share the same set of L2 adapter
instances (thread-safe by contract) but use different eventfds.

## L2 Adapter Interface

`L2AdapterInterface` (`l2_adapters/base.py`) provides the non-blocking I/O
primitives that both controllers call. All operations follow a
**submit → poll eventfd → query result** pattern.

### Event Fds

Each adapter exposes **three distinct** eventfds:

| Method                         | Used by            | Signaled when                |
|--------------------------------|--------------------|------------------------------|
| `get_store_event_fd()`         | StoreController    | A store task completes       |
| `get_lookup_and_lock_event_fd()` | PrefetchController | A lookup task completes    |
| `get_load_event_fd()`          | PrefetchController | A load task completes        |

**Critical invariant:** All event fds across all adapters must be globally
unique. The controllers build `fd → adapter_index` maps; duplicate fds would
silently misroute events.

### Store Operations

```
submit_store_task(keys, objects) -> L2TaskId
pop_completed_store_tasks() -> dict[L2TaskId, L2StoreResult]
```

- **Caller provides buffers:** The `objects` list contains `MemoryObj` references
  managed by the caller (StoreController holds L1 read locks on them).
- **Coarse-grained errors:** A store task either fully succeeds or fully fails.
  The completion dict maps each task id to an `L2StoreResult`
  (`lmcache.v1.distributed.internal_api.L2StoreResult`) that encodes both
  the success flag and the bytes actually transferred. Use
  `result.is_successful()` to check the outcome and
  `result.bytes_transferred()` to read the real byte count written to L2
  (always `0` on failure). Adapters that fast-path duplicate keys (e.g.
  skip the write when the key already exists in the backend) should
  report the real, non-skipped byte count here so the L2 throughput
  histogram reflects actual work — not submitted-but-skipped bytes.
- **Pop semantics:** `pop_completed_store_tasks()` drains all completed tasks.
  Each task appears exactly once.

### Lookup and Lock Operations

```
submit_lookup_and_lock_task(keys) -> L2TaskId
query_lookup_and_lock_result(task_id) -> Bitmap | None
submit_unlock(keys) -> None
```

- **Locking:** `lookup_and_lock` atomically checks which keys exist and acquires
  L2-side locks on found keys. This prevents L2 eviction between lookup and load.
- **Fine-grained results:** Returns a `Bitmap` where bit `i` is set if `keys[i]`
  was found and locked.
- **One-shot query:** `query_lookup_and_lock_result` returns `None` while pending,
  then the `Bitmap` exactly once. Subsequent calls return `None`.
- **Unlock contract:** `submit_unlock` is fire-and-forget. The adapter **must**
  guarantee eventual success (retry internally if needed). The caller will never
  retry.

### Load Operations

```
submit_load_task(keys, objects) -> L2TaskId
query_load_result(task_id) -> Bitmap | None
```

- **Caller provides buffers:** The `objects` list contains pre-allocated L1 write
  buffers. The adapter writes loaded data directly into these buffers.
- **Fine-grained results:** Returns a `Bitmap` where bit `i` is set if `keys[i]`
  was successfully loaded.
- **One-shot query:** Same semantics as lookup — returns the Bitmap exactly once.

### Thread Safety

The adapter must be safe for concurrent calls from the StoreController thread
and the PrefetchController thread. In practice, the store operations and
lookup/load operations use separate internal state, so this is usually
straightforward with per-operation locks or lock-free queues.

### Task ID Scope

`L2TaskId` values are only unique **within a single adapter**. When tracking
tasks across multiple adapters, use the composite key `(adapter_index, task_id)`.

## StoreController

**Purpose:** Asynchronously replicate L1 data to L2 after writes complete.

**Source:** `storage_controllers/store_controller.py`

### Lifecycle

```
StorageManager.__init__
  → StoreController(l1_manager, l2_adapters, descriptors, policy)
  → controller.start()        # spawns background thread
  ...
StorageManager.close()
  → controller.stop()         # joins thread, releases locks
```

### Event-Driven Loop

The StoreController's background thread polls on:

1. **StoreListener eventfd** — fired by L1Manager when `finish_write()` completes.
   The listener is an `L1ManagerListener` registered with L1Manager.
2. **Per-adapter store eventfds** — fired when L2 store tasks complete.

### Data Flow

```
L1 finish_write()
  │
  ▼ (L1Manager listener callback, inside L1 lock — must be non-blocking)
StoreListener.on_l1_keys_write_finished(keys)
  │  appends keys + signals eventfd
  ▼
_store_loop: poll wakes up
  │
  ▼
_process_new_keys(keys)
  │
  ├─ 1. Group keys by shape (today: (model_name, kv_rank); L1 is a shared
  │     pool so one drain may span models/parallelism configs with different
  │     KV shapes, and each submit_store_task must see uniform (shape, dtype)).
  │
  ├─ 2. For each per-shape group:
  │     StorePolicy.select_store_targets(group_keys, adapters)
  │       → dict[adapter_index, list[ObjectKey]]
  │
  ├─ 3. For each adapter target:
  │     L1Manager.reserve_read(target_keys)  → get MemoryObj + read lock
  │     adapter.submit_store_task(keys, objs)
  │     Track as InFlightStoreTask
  │
  ▼ (later, for each adapter whose store_efd signaled)
_drain_l2_store_completions(signaled_adapters)
  │  adapter.pop_completed_store_tasks() → deposit L2StoreResult
  │  (success flag + bytes_transferred) on each InFlightStoreTask
  │
  ▼
_advance_request(task_key, task)  [state transition]
  │  skip if l2_store_result still None
  │
  ▼
_finalize_store(task_key, task)  [terminal execution]
  │
  ├─ 4. L1Manager.finish_read(read_locked_keys)  → release read locks
  │
  ├─ 5. If success: StorePolicy.select_l1_deletions(keys) → delete from L1
  │     If failure: log warning (best-effort, no retry)
  │
  ▼
Done. Keys remain in L1 unless the policy deletes them.
```

### Lock Invariants

| Phase             | L1 Lock State | L2 Lock State |
|-------------------|---------------|---------------|
| Before store      | Unlocked      | N/A           |
| During store      | Read-locked   | N/A           |
| After store       | Unlocked      | N/A           |

- **Read locks during store** prevent eviction from removing L1 data while the
  adapter is reading it.
- **Always released:** `stop()` calls `_cleanup_in_flight_tasks()` which releases
  all in-flight read locks, even if tasks haven't completed.

### StorePolicy

The policy decides two things:

1. **`select_store_targets(keys, adapters) → dict[int, list[ObjectKey]]`**
   Which adapters get which keys. A key can go to multiple adapters.
   `DefaultStorePolicy`: all keys → all adapters.

2. **`select_l1_deletions(keys) → list[ObjectKey]`**
   Which keys to evict from L1 after successful L2 store.
   `DefaultStorePolicy`: never delete (empty list).

Policies are selected by name via `--l2-store-policy` (default: `"default"`).
New policies self-register with `register_store_policy(name, cls)` at import
time and are auto-discovered by `storage_controllers/__init__.py`.

## PrefetchController

**Purpose:** Asynchronously load KV cache data from L2 into L1 ahead of a
serving request. Called by `StorageManager.submit_prefetch_task()` for keys
not already in L1.

**Source:** `storage_controllers/prefetch_controller.py`

### Lifecycle

```
StorageManager.__init__
  → PrefetchController(l1_manager, l2_adapters, descriptors, policy)
  → controller.start()        # spawns background thread
  ...
StorageManager.close()
  → controller.stop()         # joins thread, releases all locks
```

### External API (Thread-Safe)

```python
# Called from the serving thread
request_id = controller.submit_prefetch_request(keys, layout_desc)

# Polled by the serving thread
result = controller.query_prefetch_result(request_id)  # int | None
```

- `submit_prefetch_request` enqueues the request and signals the background thread
  via an eventfd. Returns immediately.
- `query_prefetch_result` returns `None` while in-progress, then the **prefix hit
  count** exactly once (pop semantics).

### Prefix-Only Loading

**Key invariant:** Only the **contiguous prefix** of found keys is loaded.

If L2 has keys `{0, 1, 3, 4}` but not key `2`, only keys `{0, 1}` are loaded.
The gap at index 2 means the vLLM engine cannot use keys 3 and 4 (it needs a
contiguous prefix of computed KV cache). Loading them would waste I/O bandwidth
and L1 memory.

This is enforced by `trim_load_plan_to_prefix()` after the policy computes the
raw load plan.

### Event-Driven Loop

The PrefetchController's background thread polls on:

1. **Submission eventfd** — signaled by `submit_prefetch_request()`.
2. **Per-adapter lookup eventfds** — signaled when lookup tasks complete.
3. **Per-adapter load eventfds** — signaled when load tasks complete.

### Request State Machine

Each request goes through two phases:

```
LOOKUP ──────────────────────────► PLAN_AND_LOAD ──────────► COMPLETED
  │                                    │
  │ submit lookup_and_lock             │ compute load plan
  │ to ALL adapters                    │ reserve L1 write buffers
  │                                    │ submit load tasks
  │ wait for all lookups               │ wait for all loads
  │ to complete                        │ finalize
  ▼                                    ▼
```

### Data Flow

```
submit_prefetch_request(keys, layout_desc)
  │
  ▼ (cross-thread: submission queue + eventfd signal)
_drain_submission_queue → _pending_queue
  │
  ▼ (if below max_in_flight)
_start_lookup_phase(request_id, keys, layout_desc)
  │
  ├─ Submit lookup_and_lock_task(keys) to EVERY adapter
  │
  ▼ (wait for all adapter lookups to complete)
_advance_request(request, signaled_adapters)  [LOOKUP branch]
  │  _poll_lookup_results(request, signaled_adapters[LOOKUP])
  │  when all_lookups_done():
  │
  ▼
_transition_to_load_phase(request)
  │
  ├─ 1. PrefetchPolicy.select_load_plan(keys, lookup_results, adapters)
  │     → dict[adapter_index, Bitmap]
  │
  ├─ 2. trim_load_plan_to_prefix()
  │     → only keep contiguous prefix keys
  │
  ├─ 3. L1Manager.reserve_write(keys, is_temporary=True, mode="new")
  │     → allocate L1 write buffers
  │
  ├─ 4. Re-trim plan to only successfully reserved keys
  │
  ├─ 5. Phase 1 unlock: unlock L2 keys locked in lookup but NOT in load plan
  │
  ├─ 6. Submit load_task(keys, objs) per adapter
  │
  ▼ (wait for all adapter loads to complete)
_advance_request(request, signaled_adapters)  [PLAN_AND_LOAD branch]
  │  _poll_load_results(request, signaled_adapters[PLAN_AND_LOAD])
  │  when all_loads_done():
  │
  ▼
_finalize_load(request)
  │
  ├─ 7. Phase 2 unlock: unlock all L2 keys in the load plan
  │
  ├─ 8. L1Manager.finish_write_and_reserve_read(loaded_keys)
  │     → atomically: write unlock + read lock
  │
  ├─ 9. L1Manager.finish_write(failed_keys) + delete(failed_keys)
  │     → clean up partial failures
  │
  ├─ 10. Release read locks for loaded keys beyond the prefix
  │      (partial load failures can create gaps)
  │
  ▼
_complete_request(request_id, prefix_hits)
  │  store result, remove from in-flight tracking
```

### Lock Invariants

| Phase             | L1 Lock State                    | L2 Lock State               |
|-------------------|----------------------------------|-----------------------------|
| Lookup            | None                             | Found keys are locked       |
| After plan        | Write-locked (reserved buffers)  | Plan keys locked; others unlocked (phase 1) |
| During load       | Write-locked                     | Plan keys locked            |
| After load        | **Read-locked** (prefix keys)    | All unlocked (phase 2)      |
| After finalize    | Read-locked (prefix only)        | All unlocked                |

- **L1 write buffers are temporary:** `is_temporary=True` allows the eviction
  controller to reclaim them if needed, although normally they are short-lived.
- **Atomic write→read transition:** `finish_write_and_reserve_read()` ensures no
  eviction window between write completion and read lock acquisition.
- **Post-finalize read locks:** The prefix keys remain read-locked in L1 so the
  serving engine can consume them. The caller (`StorageManager`) releases these
  via `finish_read_prefetched()` after use.

### L2 Lock Management

L2 locks prevent adapter-side eviction between lookup and load. They must be
released in all cases — success, failure, and shutdown.

**Phase 1 unlock** (`_unlock_unneeded_keys`): After the load plan is computed,
keys that were locked during lookup but are NOT in the final load plan are
unlocked immediately. This happens when:
- The policy assigned a key to a different adapter.
- The key was trimmed by prefix trimming.
- L1 write reservation failed.

**Phase 2 unlock** (`_unlock_all_plan_keys`): After load completes (regardless
of success or failure), all keys in the load plan are unlocked.

**Shutdown cleanup** (`_cleanup_in_flight_requests`): Releases all held L1 and
L2 locks for any in-flight requests.

### PrefetchPolicy

```python
select_load_plan(keys, lookup_results, adapters) → dict[int, Bitmap]
```

Receives lookup bitmaps from all adapters and produces a non-overlapping
assignment of keys to adapters. Each key appears in at most one adapter's bitmap.

`DefaultPrefetchPolicy`: For each key, assign it to the first (lowest-indexed)
adapter that has it. This is a simple greedy approach.

Policies are selected by name via `--l2-prefetch-policy` (default: `"default"`).
New policies self-register with `register_prefetch_policy(name, cls)` at import
time and are auto-discovered by `storage_controllers/__init__.py`.

### Max In-Flight Limiting

The controller limits concurrent prefetch requests to `max_in_flight` (default: 8).
Requests beyond this limit are queued in `_pending_queue` and dequeued as
in-flight requests complete.

Note: This is a simple count-based limit. A future improvement would use a
dynamic admission controller based on L1 memory usage of in-flight requests.

## Integration: StorageManager

`StorageManager` (`storage_manager.py`) is the top-level entry point that wires
everything together.

### Prefetch Flow (from the serving engine's perspective)

```python
# 1. Submit: check L1 first, then delegate remainder to L2
handle = sm.submit_prefetch_task(keys, layout_desc)

# 2. Poll: busy-wait for completion
while True:
    found_count = sm.query_prefetch_status(handle)
    if found_count is not None:
        break

# 3. Read: access the prefetched data (holds read locks)
with sm.read_prefetched_results(keys[:found_count]) as objs:
    # use objs ...
    pass

# 4. Release: drop read locks
sm.finish_read_prefetched(keys[:found_count])
```

### PrefetchHandle

```python
@dataclass(frozen=True)
class PrefetchHandle:
    request_id: int          # -1 if no L2 request needed
    l1_prefix_hit_count: int # leading keys already in L1
    total_requested_keys: int
    submit_time: float       # for latency logging
```

`submit_prefetch_task` first checks L1 for a contiguous prefix of hits:
- If all keys hit L1: returns handle with `request_id=-1` (no L2 work).
- If some keys miss: submits the **remaining** keys to PrefetchController.

`query_prefetch_status` combines L1 hits with L2 results:
`total_hits = l1_prefix_hit_count + l2_prefix_hits`.

## Assumptions and Invariants Summary

1. **All eventfds are globally unique** across all adapters and all operation
   types. Violating this corrupts the poll-based dispatch.

2. **L2 task IDs are per-adapter, not global.** Use `(adapter_index, task_id)`
   as composite keys.

3. **Query results are one-shot.** Both `query_lookup_and_lock_result()` and
   `query_load_result()` return a non-None value exactly once per task.

4. **`submit_unlock` must eventually succeed.** The controllers will never
   retry. The adapter must handle retries internally.

5. **Prefix-only loading.** Only the contiguous prefix of found keys is loaded
   from L2. Gaps break the prefix.

6. **Listener callbacks run inside L1Manager's lock.** `StoreListener` must be
   non-blocking (append + eventfd signal only). It must never call L1Manager
   methods (deadlock).

7. **L1 write buffers for prefetch are temporary.** Allocated with
   `is_temporary=True` to allow eviction if needed.

8. **Atomic write→read transition.** `finish_write_and_reserve_read()` prevents
   eviction between completing a prefetch write and acquiring the read lock
   for the serving engine.

9. **Both controllers release all locks on shutdown.** `stop()` always cleans
   up in-flight tasks, regardless of completion state.

10. **L2 adapters are thread-safe.** Concurrent calls from the StoreController
    thread and PrefetchController thread are expected.

## Implementing a New L2 Adapter

### Pure-Python Adapters

Implement `L2AdapterInterface` directly. See `mock_l2_adapter.py` for a
reference implementation for an in-memory adapter, or
[`raw_block.md`](raw_block.md) for a durable local-device adapter.
**No existing files need to be modified.** Create a new module
(e.g., `my_l2_adapter.py`) in the `l2_adapters/` package and self-register at
module level:

```python
# At the bottom of your module:
register_l2_adapter_type("my_type", MyL2AdapterConfig)
register_l2_adapter_factory("my_type", _create_my_l2_adapter)
```

The `__init__.py` uses `pkgutil.iter_modules()` to discover all
`*_l2_adapter.py` modules automatically, but imports them **lazily** — a
module (and its third-party dependencies) is only loaded when the
corresponding adapter type is actually requested at runtime.

### Persist / Recover (Optional)

Adapters that support persisting cached data across restarts use
`PersistConfig` (parsed from the JSON key `"persist_enabled"`, defaulting
to ``True``). Lookup always checks secondary storage on miss. There is
no dedicated interface method — adapters integrate persist into their
existing `close()` (to keep data on disk) path. See
`nixl_store_dynamic_l2_adapter.py` for a reference implementation and
[`nixl_store.md`](nixl_store.md) for design details.

### Native (C++/Rust) Storage Backends

For high-performance backends written in C++ or Rust, use the shared native
connector framework. A single C++ connector implementation works in **both**
non-MP mode (via `ConnectorClientBase`) and MP mode (via
`NativeConnectorL2Adapter`).

**Full guide:** [`csrc/storage_backends/README.md`](../../../../../csrc/storage_backends/README.md)

The `NativeConnectorL2Adapter` (`native_connector_l2_adapter.py`) bridges any
pybind-wrapped `IStorageConnector` to the `L2AdapterInterface`:

- Creates 3 Python eventfds from the connector's single eventfd
- Runs a background demux thread that routes completions by operation type
- Handles `ObjectKey` serialization and `MemoryObj` buffer extraction
- Implements client-side locking (refcount dict) for remote backends

**Reference implementation:** The Redis (RESP) connector in
`csrc/storage_backends/redis/` demonstrates all 5 steps of the integration
guide.

## Implementing a New Store or Prefetch Policy

Both store and prefetch policies use a name-based registry with automatic
module discovery. **To add a new policy, create a single file in
`storage_controllers/` — no changes to any existing file are needed.**

### Store Policy

1. Create a new file (e.g., `storage_controllers/store_policy_tiered.py`).
2. Subclass `StorePolicy` and implement `select_store_targets()` and
   `select_l1_deletions()`.
3. Call `register_store_policy("tiered", TieredStorePolicy)` at module level.

```python
from lmcache.v1.distributed.storage_controllers.store_policy import (
    StorePolicy,
    AdapterDescriptor,
    register_store_policy,
)
from lmcache.v1.distributed.api import ObjectKey


class TieredStorePolicy(StorePolicy):
    def select_store_targets(self, keys, adapters):
        # custom logic ...
        ...

    def select_l1_deletions(self, keys):
        return []


register_store_policy("tiered", TieredStorePolicy)
```

The policy is now available via `--l2-store-policy tiered`.

### Prefetch Policy

Same pattern: subclass `PrefetchPolicy`, implement `select_load_plan()`,
and call `register_prefetch_policy("name", cls)`.

### How Discovery Works

`storage_controllers/__init__.py` uses `pkgutil.iter_modules()` to import
every module in the package at import time. When your module is imported,
the `register_*_policy()` call at module level adds it to the registry.
The `--l2-store-policy` and `--l2-prefetch-policy` CLI arguments use the
registry to populate their `choices` list.
