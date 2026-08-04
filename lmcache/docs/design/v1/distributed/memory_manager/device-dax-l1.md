# Device-DAX L1 Memory Manager Design

This document describes the Device-DAX-backed L1 tier and its runtime
reconfiguration: adding and removing Device-DAX devices by path while the
process keeps running.

## Goals

- Back L1 with one or more Device-DAX devices, optionally fronted by a DRAM
  tier (hybrid DRAM + Device-DAX).
- Add and remove Device-DAX capacity at runtime by device path, without a
  restart.
- Keep the L1 allocate / free / usage / descriptor interface unchanged for
  callers (`L1Manager`).
- Never move or invalidate a live allocation. L1 hands out raw device pointers
  that requests read and write directly, so an arena with live allocations must
  stay mapped.

## Components

`lmcache/v1/distributed/memory_manager/devdax_l1_memory_manager.py` defines
`DevDaxL1MemoryManager`, the L1 tier object. It is thin: it stores its config,
delegates `allocate` / `free` / `get_memory_usage` / `get_l1_memory_desc` to the
pooled allocator, and exposes the runtime-reconfigure surface
(`add_device`, `remove_device`, `get_arena_statuses`).

`lmcache/v1/memory_allocators/devdax_memory_allocator.py` defines
`DevDaxMemoryAllocator`, which owns the arena pool, the `host_mem_lock`, the
per-arena `TensorMemoryAllocator` instances, the mmap lifecycle, and
best-effort CUDA host-memory pinning. `_DevDaxArena` is the per-arena record.
`DevDaxArenaState`, `DevDaxRemoveMode`, and `DevDaxArenaStatus` are the public
reconfigure types.

`lmcache/v1/distributed/l1_manager.py` selects `DevDaxL1MemoryManager` when
`memory_config.devdax_path` is set, in preference to the CPU-only and GDS L1
managers.

The runtime-reconfigure HTTP surface is not part of this design; when added it
is intended to mirror the L2 `/reconfigure/dax/*` endpoints (see
[../l2_adapters/dax.md](../l2_adapters/dax.md)) and route only `operation` plus
payload down to the manager.

## Arena Pool

The pool is an optional DRAM local allocator plus an ordered list of Device-DAX
arenas. Each arena owns its own file descriptor, `mmap` (`MAP_SHARED`), flat
`torch.uint8` view of the mapped bytes, and a `TensorMemoryAllocator` with an
arena-local address space, plus a best-effort CUDA host-memory pin and a
lifecycle state.

Allocation order:

1. DRAM local allocator first, if present (hybrid mode).
2. Then each `active` arena in pool order as overflow.

Single-object allocation pre-checks each arena's free capacity and skips arenas
without room instead of probing them with a failed allocation attempt; a failed
probe logs a warning inside the arena allocator, which would emit one line per
full arena on every allocation once earlier arenas fill up.

Batched allocation fills greedily across active arenas and is all-or-nothing: if
the arenas cannot collectively satisfy the request, the partial allocation is
rolled back and the call fails, matching the single-allocator contract.

Free routing: every Device-DAX allocation carries the `DevDaxMemoryAllocator` as
its parent. The owning arena is located by pointer range
(`base_ptr <= data_ptr < base_ptr + size`), because each arena has an
arena-local address space and a freed object must return to the exact address
manager it came from. Batched frees are grouped by owning arena. `base_ptr` is
captured when the arena is mapped, so routing stays correct even for an arena
whose deferred unmap has already dropped its buffer references. After freeing,
the pool re-attempts the reap of every `draining` arena, so an unmap deferred
by lingering views completes on a later free.

## Primary Arena

The primary arena backs `get_l1_memory_desc()` and the `buffer` property and can
never be removed.

- Pure Device-DAX mode (no DRAM tier): the initial arena is primary.
- Hybrid mode (DRAM tier present): DRAM is the primary L1 region, so every
  Device-DAX arena, including the initial one, is removable overflow.

`is_primary` is therefore true only for the initial arena when there is no DRAM
local allocator.

## Runtime Reconfigure

The manager and allocator expose the same contract:

- `add_device(device_path, size_in_bytes) -> DevDaxArenaStatus`
- `remove_device(device_path, mode=DevDaxRemoveMode.DRAIN) -> DevDaxArenaStatus`
- `arena_statuses()` on the allocator / `get_arena_statuses()` on the manager,
  returning `list[DevDaxArenaStatus]` in pool order.

Add:

1. Validate the path is non-empty, the size is positive, the allocator is not
   closed, and the path is not already mapped.
2. Map the device: `open(O_RDWR)`, capacity check via `fstat.st_size`,
   `mmap(MAP_SHARED, RW)`; build a `TensorMemoryAllocator`; best-effort pin.
3. Append the arena as `active` and non-primary. It is immediately available as
   overflow. Existing allocations are untouched.

If any setup step fails (mapping, allocator construction, or pin registration),
the freshly opened fd and mmap are released before the error propagates.

Remove (drain):

1. Reject if no arena is mapped at the path, or the arena is primary.
2. Mark the arena `draining`; it is excluded from new allocations.
3. If it has no live allocations, unmap it immediately (`removed`). Otherwise it
   stays `draining`, and the `free` that releases its last allocation unmaps it
   automatically (auto-reap). If the unmap is blocked by lingering external
   views into the mapping (e.g. freed tensors awaiting garbage collection), the
   arena stays `draining` and later frees retry the reap.

State machine: `active -> draining -> removed`. `removed` is a report-only
terminal value; a removed arena has already left the pool, so it is never
observed by an in-pool lookup.

Modes that relocate live objects (migrate, evict) are intentionally not
supported here; see Current Limits.

## How to Reconfigure at Runtime

Configure the initial Device-DAX device when the server starts, either with the
MP server CLI flag `--l1-devdax-path /dev/dax0.0` (the mapped size follows the
L1 size settings) or programmatically:

```python
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.l1_manager import L1Manager

l1 = L1Manager(
    L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=32 << 30,
            use_lazy=False,
            shm_name="",
            devdax_path="/dev/dax0.0",
        )
    )
)
```

Until the HTTP control surface lands, reconfiguration is programmatic, on the
`DevDaxL1MemoryManager` owned by the `L1Manager`:

```python
manager = l1._memory_manager  # DevDaxL1MemoryManager

# Grow: map an already-provisioned device and add it to the pool. It serves
# overflow allocations immediately.
status = manager.add_device("/dev/dax1.0", 32 << 30)

# Inspect per-device usage: used/free bytes, live allocations, state.
for arena in manager.get_arena_statuses():
    print(arena.device_path, arena.state, arena.used_bytes, arena.free_bytes)

# Shrink: drain-remove a device. REMOVED means it was empty and is already
# unmapped; DRAINING means cached entries still live on it.
status = manager.remove_device("/dev/dax1.0")
```

A `DRAINING` device accepts no new allocations, keeps serving reads for the KV
entries already on it, and unmaps automatically once the last of them is freed
(deleted or evicted). Poll `get_arena_statuses()` until the path disappears
from the list; `active_allocations` on the draining entry shows how many
allocations still gate the unmap. Calling `remove_device` again on a draining
path is safe and returns the current status.

`add_device` rejects paths that are already mapped, and `remove_device`
rejects the primary arena (the initial device in pure Device-DAX mode); both
raise `ValueError`. The device must already exist and be readable and
writable; runtime reconfigure does not provision DAX namespaces (see Current
Limits).

## Thread Safety

`host_mem_lock` (non-reentrant) serializes every pool mutation and every
per-arena allocate, free, add, remove, and reap. The DRAM local allocator has
its own synchronization and is used outside this lock. Rollback of a failed
batched allocation and arena reaping both run while the lock is held; the
internal helpers assume the lock is held and never re-acquire it.

## mmap and CUDA Host Memory

Each arena maps its device with `mmap(MAP_SHARED, PROT_READ | PROT_WRITE)` and
exposes the bytes as a flat `torch.uint8` tensor via a ctypes array
(`from_buffer`). On unmap every reference into the mmap (the allocator buffer,
the arena buffer, and the ctypes array) is released before `mmap.close()`,
because CPython refuses to close a buffer that still has exported pointers.
`mmap` dups the underlying file descriptor, so unmap releases both the opened fd
and the mmap's dup.

CUDA host-memory registration (pinning) is per-arena and best-effort; a pin
failure is logged and the arena falls back to pageable host copies.

## Transfer-Channel Compatibility

Device-DAX L1 is not a single registerable memory region:
`l1_exposes_single_memory_region()` returns `False`, and P2P / NIXL reject
Device-DAX L1. Arenas can therefore be added and removed without invalidating a
whole-arena transfer registration.

## Capacity

`get_memory_usage()` returns used and total bytes computed under
`host_mem_lock`. Used bytes sum the live allocations of the DRAM local
allocator and every arena (active and draining). Total bytes sum the DRAM
allocator and only the *active* arenas: a draining arena accepts no new
allocations, so its free space is not usable headroom and its capacity is
excluded. A draining arena still holding live bytes therefore pushes used
above total (ratio > 1), which is intentional -- it keeps the eviction
watermark tracking real pressure on the active pool instead of being diluted
by capacity that is being removed.

## Verification

`tests/v1/distributed/test_devdax_l1_allocator.py` unit-tests the pool:
add/remove lifecycle, drain gating, per-arena usage, deferred unmap while
external views are alive, and mapping release on setup failure.
`tests/v1/distributed/test_devdax_l1_reconfigure_integration.py` (opt-in via
`RUN_DEVDAX_L1_INTEGRATION=1`) drives real mmap-backed devices end to end,
both at the memory-manager level and through the `L1Manager` KV-cache path:
KV entries land on a runtime-added device, stay readable while it drains, and
the device is unmapped only after the last cached entry is deleted. It accepts
real `/dev/dax` devices via `LMCACHE_TEST_DEVDAX_L1_PATHS`.

## Current Limits

- Only drain-based removal. Migrate and evict (relocating live objects to
  another arena or to DRAM) are deferred: L1 hands out raw device pointers that
  live requests read and write, so relocation requires hooking L1 eviction.
- The primary arena in pure Device-DAX mode cannot be removed at runtime.
- Existing arenas cannot be resized; the pool grows and shrinks by whole arenas
  (add / remove only).
- Runtime reconfigure maps and unmaps already-provisioned devices; it does not
  perform kernel-level CXL or DAX reconfiguration.
- No HTTP control surface yet; the reconfigure methods are the programmatic
  entry point.
