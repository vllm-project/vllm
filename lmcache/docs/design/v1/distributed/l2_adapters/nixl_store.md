# Nixl Store L2 Adapter Design

## Overview

The Nixl L2 adapter family implements `L2AdapterInterface` using the
[Nixl](https://github.com/ai-infra-org/nixl) library to offload KV-cache
objects from L1 (DRAM/VRAM) to a secondary storage tier via DMA. There are
two variants:

| Adapter | Type name | Storage mode | Persist | Backends |
|---|---|---|---|---|
| `NixlStoreL2Adapter` | `nixl_store` | Static (pre-allocated files) | Not supported | GDS, GDS_MT, POSIX, HF3FS, OBJ, AZURE_BLOB |
| `DynamicNixlStoreL2Adapter` | `nixl_store_dynamic` | Dynamic (per-operation files) | Supported (default on) | GDS, GDS_MT, POSIX, HF3FS |

The **static** adapter pre-allocates all storage files at init and registers
them with Nixl as a single prepped descriptor list. The **dynamic** adapter
opens/registers files per operation, enabling persist/recover of cached KV
metadata across restarts and avoiding OS open-file-descriptor limits.

---

## Static Adapter: `NixlStoreL2Adapter`

### Key Components

#### `NixlStoreObj`
Metadata record for a single cached object in Nixl storage:
- `page_indices` — list of pre-allocated storage slot indices holding the data.
- `size` — byte size of the stored object.
- `layout` — optional `MemoryLayoutDesc` (shapes/dtypes) for reconstruction.
- `pin_count` — reference count preventing eviction while a load is in flight.

#### `NixlObjPool`
Thread-safe integer index pool representing the fixed set of pre-allocated
storage slots (`pool_size` entries). Slots are allocated before a store and
freed after a failed transfer or when the object is evicted.

#### `NixlStorageAgent`
Thin wrapper around the Nixl agent API. Responsibilities:
- Register the L1 memory buffer with Nixl (`init_mem_handlers`).
- Register storage slots (files or object keys) with Nixl
  (`init_storage_handlers_file` / `init_storage_handlers_object`).
- Produce pre-prepared transfer handles for batched DMA reads/writes
  (`get_mem_to_storage_handle`, `get_storage_to_mem_handle`).
- Drive transfers asynchronously (`post_non_blocking`).

#### `NixlStoreL2Adapter`
The public adapter implementing `L2AdapterInterface`. It owns:
- A background asyncio event loop (in a dedicated daemon thread) that
  executes all DMA coroutines.
- Three Linux event-fds (store / lookup / load) used to signal completion
  to the caller without polling.
- A shared `dict[ObjectKey, NixlStoreObj]` as the in-memory index.
- A single `threading.Lock` protecting all shared state.

---

## Operation Flow

### Store
```
submit_store_task(keys, objects)
  └─ schedules _execute_store_in_the_loop on the asyncio loop
       ├─ for each key/object: allocate storage slots, collect page indices
       ├─ issue single batched DMA write (mem → storage)
       ├─ on success: record key→NixlStoreObj in _memory_objects
       └─ on failure: free allocated slots; mark task failed
  └─ signals store event-fd
```

### Lookup & Lock
```
submit_lookup_and_lock_task(keys)
  └─ schedules _execute_lookup_in_the_loop (sync, via call_soon_threadsafe)
       ├─ for each key present: set bitmap bit, increment pin_count
       └─ records bitmap in _completed_lookup_tasks
  └─ signals lookup event-fd

submit_unlock(keys)
  └─ schedules pin_count decrement for each key (fire-and-forget)
```

### Load
```
submit_load_task(keys, objects)
  └─ schedules _execute_load_in_loop on the asyncio loop
       ├─ for each found key: collect mem/storage page indices, set bitmap bit
       ├─ issue single batched DMA read (storage → mem)
       └─ records bitmap in _completed_load_tasks
  └─ signals load event-fd
```

---

## Threading Model

| Thread | Role |
|---|---|
| Caller thread(s) | Call `submit_*` / `query_*`; never touch storage directly |
| Event-loop thread | Executes all Nixl DMA coroutines; owns `_memory_objects` mutations |
| Shared lock | Protects `_memory_objects`, task result dicts, and task-id counter |

Lookup is synchronous (scheduled via `call_soon_threadsafe`); store and load
are async coroutines (scheduled via `run_coroutine_threadsafe`).

---

## Memory Address → Page Index Mapping

L1 memory is registered with Nixl as a single contiguous buffer split into
fixed-size pages of `align_bytes`. A memory object at address `addr` of size
`sz` maps to page indices:

```
[addr // align_bytes, addr // align_bytes + 1, ..., addr // align_bytes + sz // align_bytes - 1]
```

Both `addr` and `sz` must be multiples of `align_bytes`.

---

## Dynamic Adapter: `DynamicNixlStoreL2Adapter`

**Source:** `l2_adapters/nixl_store_dynamic_l2_adapter.py`

### Motivation

The static adapter pre-allocates all storage files and registers them with
Nixl at init time. This has two limitations:

1. **OS file descriptor limits.** Each storage slot requires an open fd,
   limiting pool size in practice.
2. **No persist/recover.** Files are created with random UUIDs and the
   in-memory index (`_memory_objects`) is lost on shutdown.

The dynamic adapter solves both by opening/registering files per operation
and using deterministic file names derived from `ObjectKey`.

### Key Differences from Static

| Aspect | Static | Dynamic |
|---|---|---|
| File lifecycle | All opened at init, closed at shutdown | Opened per store/load, closed after each transfer |
| File naming | Random UUID (`obj_{i}_{uuid}.bin`) | Deterministic from ObjectKey (`{model}_{rank}_{hash}.bin`) |
| Nixl registration | Single prepped dlist for all storage | Per-operation register → transfer → deregister |
| Pool / page indices | `NixlObjPool` manages fixed slots | No pool; `NixlStoreObj.page_indices` unused (`[]`) |
| Capacity control | Pool size (slot count) | `max_capacity_gb` (byte-based) |
| Persist/recover | Not supported | Supported |
| Batching | One DMA transfer per batch of keys | One DMA transfer per key (each key = separate file) |

### Key Components

#### `DynamicNixlStorageAgent`

Similar to `NixlStorageAgent` but only registers L1 memory at init. Storage
file registration is per-operation:

- `dynamic_store_file(mem_indices, file_path, page_size)` — create file,
  register with Nixl, DMA write, deregister, close fd.
- `dynamic_load_file(mem_indices, file_path, page_size)` — open existing
  file, register, DMA read, deregister, close fd.
- `dynamic_delete_file(file_path)` — `os.unlink()`.

#### `DynamicNixlStoreL2Adapter`

Same `L2AdapterInterface` contract as the static adapter. Differences:

- **Store:** Iterates per key, calling `dynamic_store_file` for each.
  Checks `_total_bytes + obj_size > _max_capacity_bytes` before each write;
  stops the batch if capacity is exceeded.
- **Delete:** Removes the file from disk via `dynamic_delete_file` in
  addition to removing the key from `_memory_objects`.
- **Capacity:** Tracks `_total_bytes` (incremented on store and on
  secondary lookup, decremented on delete). `get_usage()` returns
  `_total_bytes / _max_capacity_bytes` for the eviction controller.
- **Close:** Stops the event loop first (waits for in-flight tasks);
  when `persist_enabled`, data files are kept on disk, otherwise all
  data files are deleted.
- **Lookup:** A lookup miss always falls through to a synchronous
  secondary lookup on disk; see the Persist / Secondary Lookup section
  below.

### Operation Flow

#### Store
```
submit_store_task(keys, objects)
  └─ schedules _execute_store_in_the_loop on the asyncio loop
       ├─ for each key/object:
       │    ├─ check capacity (skip remaining if exceeded)
       │    ├─ compute deterministic file path from ObjectKey
       │    ├─ open file, register with Nixl, DMA write, deregister, close
       │    └─ record key→NixlStoreObj in _memory_objects, update _total_bytes
       └─ signals store event-fd
```

#### Load
```
submit_load_task(keys, objects)
  └─ schedules _execute_load_in_loop on the asyncio loop
       ├─ for each found key:
       │    ├─ compute file path from ObjectKey
       │    └─ open file, register with Nixl, DMA read, deregister, close
       └─ signals load event-fd
```

Lookup and unlock are identical to the static adapter (in-memory index
lookup + pin count management).

---

## Persist / Secondary Lookup

### Config

`PersistConfig` (`l2_adapters/config.py`) has one boolean flag:

| Field | Default | Purpose |
|---|---|---|
| `persist_enabled` | `True` | If True, data files are kept on disk at shutdown. |

Parsed from the adapter JSON config key `"persist_enabled"` by
`L2AdapterConfigBase._parse_persist_config()`.

Lookup always checks secondary storage (disk) on miss — this is not
configurable.

Only the dynamic adapter (`nixl_store_dynamic`) uses persist; the
static adapter ignores it.

### How it works

There is no dedicated `persist()` or `recover()` method on the
`L2AdapterInterface`. Persist and recover are implemented implicitly
through two existing hooks:

#### Persist (file retention at shutdown)

In `close()`, after the event loop has stopped:

- If `persist_enabled`, data files are left on disk untouched.
- Otherwise, every file in `_memory_objects` is `os.unlink`'d to avoid
  orphaned storage.

No metadata JSON is written — the deterministic `ObjectKey → filename`
mapping is sufficient to rediscover each file on restart.

#### Secondary Lookup (lazy disk recovery)

`_execute_lookup_in_the_loop` always extends the in-memory index lookup
with a secondary lookup on miss:

1. Compute deterministic file path from the ObjectKey.
2. `os.stat(file_path)` — if the file exists, treat as a hit.
3. Populate `_memory_objects[key]` lazily with `size` from the stat
   result and `layout=None`.
4. Update `_total_bytes`; enforce capacity (skip if it would exceed).

The `NixlStoreObj.layout` field is left as `None` on secondary lookup. Layout
information is only needed at load time, where the caller supplies it
via the provided `MemoryObj`'s shape/dtype/phy_size.

---

## Configuration

### Static Adapter (`nixl_store`)

```json
{
  "type": "nixl_store",
  "backend": "POSIX",
  "backend_params": {
    "file_path": "/path/to/storage",
    "use_direct_io": "false"
  },
  "pool_size": 100
}
```

### Dynamic Adapter (`nixl_store_dynamic`)

```json
{
  "type": "nixl_store_dynamic",
  "backend": "POSIX",
  "backend_params": {
    "file_path": "/path/to/storage",
    "use_direct_io": "false",
    "max_capacity_gb": "10"
  },
  "persist_enabled": true
}
```

---