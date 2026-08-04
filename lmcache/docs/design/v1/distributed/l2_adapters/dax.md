# DAX L2 Adapter Design

This document describes the built-in `dax` L2 adapter for LMCache
multiprocess mode and how it shares implementation with the non-MP DAX
storage backend.

## Goals

- Reuse one synchronous DAX core for MP and non-MP DAX storage.
- Keep the MP controller flow unchanged: adapters still use the normal
  submit, event-fd, and query-result contract.
- Keep the adapter facade stable while the runtime DAX device pool changes.
- Keep DAX volatile-only. Keys are indexed in process memory and are not
  recovered from device bytes after restart.

## Components

`lmcache/v1/storage_backend/dax/core.py` defines `DaxCore[KeyT]`. The core owns
the mapped DAX arena, fixed-size slot allocation, in-memory index, LRU order,
in-flight writes, external lock refcounts, active read borrow counts, close
coordination, and direct `ctypes.memmove` copies.

`lmcache/v1/storage_backend/plugins/dax_backend.py` is the non-MP wrapper. It
keeps existing non-MP behavior such as the local CPU backend requirement, TP=1
validation, optional async put, and the staging-slab batched restore path.

`lmcache/v1/distributed/l2_adapters/dax_l2_adapter.py` is the MP adapter. It
self-registers adapter type `dax`, owns separate event notifiers and worker
pools for store, lookup, and load operations, and uses one or more
`DaxCore[ObjectKey]` instances behind a stable facade.

`lmcache/v1/multiprocess/http_apis/reconfigure_api.py` exposes runtime
reconfiguration endpoints:

- `GET /reconfigure/dax/status`
- `POST /reconfigure/dax/add`
- `POST /reconfigure/dax/remove`
- `POST /reconfigure/dax/resize`

The HTTP layer routes `backend`, `operation`, and adapter-specific JSON payloads
into the generic L2 adapter reconfiguration API on `StorageManager`.
`StorageManager` only routes `operation` plus payload to a reconfigurable
adapter; DAX path, mode, and migration semantics stay inside `DaxL2Adapter`.
The same interface is intended for future adapters such as P2P, so the HTTP
layer does not inspect private adapter lists or DAX core state directly.

## Slot State

Each committed key points to one fixed-size slot. A slot is reusable only when:

- The key has been removed from the index.
- No external lock is held for the key.
- No store is in flight for the key.
- No active read has borrowed the slot.

Delete operations remove unlocked keys from the index immediately. If a read
has borrowed the slot, the slot is marked pending-free and recycled when the
borrow count reaches zero.

## MP Flow

Store:

1. `StoreController` calls `submit_store_task(keys, objects)`.
2. The adapter chooses an active DAX device. Existing keys prefer their current
   mapped device; new keys use the active device with the lowest slot usage.
3. A store worker copies each object into a DAX slot through `DaxCore.put_many`.
4. The adapter records task-level success as `all(per_key_results)`.
5. The store event fd is signaled and store listeners are notified for the keys
   that were actually accepted by the core.

Lookup and load:

1. `PrefetchController` calls `submit_lookup_and_lock_task(keys)`.
2. The adapter checks `key -> device` mappings first, then scans readable
   devices if needed.
3. The adapter calls `DaxCore.exists_many(keys, lock=True)` and returns a full
   bitmap, including holes.
4. Load workers call `DaxCore.load_many_into(keys, objects)` on the device that
   currently owns each key.
5. `submit_unlock(keys)` releases the external lock refcounts on every DAX
   core. This is deliberate because migration can update `key -> device`
   mappings between lookup and unlock.

## Runtime Hotplug

The DAX facade keeps the event fds and worker pools stable. Runtime hotplug only
mutates the device pool behind the facade, so `StoreController`,
`PrefetchController`, and the vLLM MP connector do not need ZMQ protocol changes
or poll-set re-registration.

Add:

1. Validate `hotplug_enabled`, path, and size.
2. Map a new `DaxCore[ObjectKey]`.
3. Append a `DaxDeviceEntry(state="active")`.
4. Return per-device status. Existing KV entries stay on their current devices.

Remove with migration:

1. Mark the source device `draining` so new stores do not choose it.
2. Reject the operation if externally locked or borrowed slots would be deleted.
3. Snapshot source keys and reserve source reads.
4. Copy each reserved payload from the source DAX pointer into another active
   DAX core with `put_reserved_from_ptr`.
5. Update `key -> device` mappings, delete the source entries, then close the
   source core.

Resize:

- Grow remaps the same core to a larger size after active reads and writes
  drain. No KV payload movement is needed.
- Shrink first proves that every live slot fits below the new slot count. If
  not, the out-of-range keys must migrate to another active device or the
  request fails. Shrink never silently evicts data.

## Restart Behavior

The adapter stores keys and metadata only in memory. Closing the adapter and
opening a new adapter against the same DAX device starts with an empty index.
Old bytes may remain on the device, but they are unreachable because PR1 does
not define any on-device metadata, scan, checkpoint, or recovery format.

## Capacity And Eviction

Usage is slot-based, not payload-byte-based. `get_usage()` reports occupied
slot capacity because the DAX arena is exhausted by slot count. The eviction
controller calls `delete(keys)`, which skips externally locked keys and
reclaims slots after active read borrows drain.

Runtime capacity is the sum of active, draining, migrating, resizing, and
removing device capacities. Closed, removed, and failed devices are excluded.

## Current Limits

- Runtime hotplug does not perform kernel-level CXL or DAX reconfiguration.
- No per-TP partitioning.
- No restart recovery.
- Only single-buffer objects are supported.
