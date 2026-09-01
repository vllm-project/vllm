# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered-memory allocation and residency for Mooncake transfers.

The Producer pool stages source tensors in a registered slab.  The Consumer
pool owns destination allocations from reservation through publication,
resident reuse, CUDA-safe retirement, and pressure-driven reclamation.
"""

from __future__ import annotations

import bisect
import math
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    MooncakeTransfer,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_T = TypeVar("_T")


@dataclass
class MemoryAllocation:
    """Describe a tensor view carved from the Consumer receive slab.

    Attributes:
        offset: Byte offset of the allocation within the slab.
        size: Aligned number of slab bytes owned by the allocation.
        tensor: Typed tensor view exposed to transfer and cache code.
    """

    offset: int
    size: int
    tensor: torch.Tensor


@dataclass
class _ResidentEntry(Generic[_T]):
    """Store one resident value and its ownership accounting.

    Attributes:
        value: Resident value owned by the pool.
        nbytes: Capacity charged to the resident pool.
        pinned: Whether active cache state prevents LRU eviction.
        leases: Number of in-flight reservations borrowing this value.
    """

    value: _T
    nbytes: int
    pinned: bool = True
    leases: int = 0


@dataclass
class ResidentLease(Generic[_T]):
    """Represent a borrow of one resident entry.

    Attributes:
        key: Cache identifier used to find the canonical resident entry.
        _entry: Entry retained even if the canonical mapping is replaced.
        _active: Whether this lease still contributes to the reference count.
    """

    key: str
    _entry: _ResidentEntry[_T]
    _active: bool = True

    @property
    def value(self) -> _T:
        return self._entry.value


class ContiguousAllocator:
    """Allocate aligned regions from one contiguous byte range.

    Attributes:
        capacity: Total number of bytes managed by the allocator.
        alignment: Allocation granularity in bytes.
        _free: Sorted free ranges represented as ``(offset, size)`` pairs.
    """

    def __init__(self, capacity: int, alignment: int = 256):
        self.capacity = capacity
        self.alignment = alignment
        self._free = [(0, capacity)]

    def allocate(self, nbytes: int) -> tuple[int, int] | None:
        size = math.ceil(nbytes / self.alignment) * self.alignment
        for index, (offset, available) in enumerate(self._free):
            if size > available:
                continue
            if size == available:
                self._free.pop(index)
            else:
                self._free[index] = (offset + size, available - size)
            return offset, size
        return None

    def free(self, offset: int, size: int) -> None:
        index = bisect.bisect_left(self._free, (offset, size))
        self._free.insert(index, (offset, size))
        if index + 1 < len(self._free):
            next_offset, next_size = self._free[index + 1]
            if offset + size == next_offset:
                self._free[index] = (offset, size + next_size)
                self._free.pop(index + 1)
        if index > 0:
            previous_offset, previous_size = self._free[index - 1]
            current_offset, current_size = self._free[index]
            if previous_offset + previous_size == current_offset:
                self._free[index - 1] = (
                    previous_offset,
                    previous_size + current_size,
                )
                self._free.pop(index)


class ResidentPool(Generic[_T]):
    """Track resident values, active leases, and LRU eviction eligibility.

    Attributes:
        used: Total bytes charged by current and leased displaced entries.
        _entries: Canonical resident entries keyed by cache identifier.
        _evictable: Unpinned and unleased entries in LRU order.
    """

    def __init__(self):
        self.used = 0
        self._entries: dict[str, _ResidentEntry[_T]] = {}
        self._evictable: OrderedDict[str, None] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def num_evictable(self) -> int:
        return len(self._evictable)

    def referenced(self) -> list[str]:
        return [key for key, entry in self._entries.items() if entry.pinned]

    def get(self, key: str) -> _T | None:
        entry = self._entries.get(key)
        return entry.value if entry is not None else None

    def insert(
        self,
        key: str,
        value: _T,
        nbytes: int,
    ) -> _T | None:
        """Pin an entry and return a displaced value that has no owners."""
        previous = self._entries.get(key)
        entry = _ResidentEntry(value, nbytes)
        if previous is not None and previous.leases == 0:
            self.used -= previous.nbytes
        self._entries[key] = entry
        self.used += nbytes
        self._evictable.pop(key, None)
        if previous is not None and previous.leases == 0:
            return previous.value
        return None

    def pin(self, key: str) -> _T | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._evictable.pop(key, None)
        entry.pinned = True
        return entry.value

    def acquire(self, key: str) -> ResidentLease[_T] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        entry.leases += 1
        self._evictable.pop(key, None)
        return ResidentLease(key, entry)

    def release(self, lease: ResidentLease[_T]) -> _T | None:
        if not lease._active:
            return None
        lease._active = False
        entry = lease._entry
        entry.leases -= 1
        current = self._entries.get(lease.key)
        if current is not entry:
            if entry.leases == 0:
                self.used -= entry.nbytes
                return entry.value
            return None
        if not entry.pinned and entry.leases == 0:
            self._evictable[lease.key] = None
        return None

    def consume(self, lease: ResidentLease[_T]) -> tuple[_T, _T | None]:
        current = self._entries[lease.key]
        current.pinned = True
        self._evictable.pop(lease.key, None)
        released = self.release(lease)
        return current.value, released

    def retire(self, key: str) -> None:
        if key not in self._entries:
            return
        entry = self._entries[key]
        entry.pinned = False
        if entry.leases == 0:
            self._evictable[key] = None

    def evict_lru(self, evict: Callable[[str, _T], bool]) -> str | None:
        for key in list(self._evictable):
            entry = self._entries[key]
            if not evict(key, entry.value):
                continue
            self._evictable.pop(key, None)
            del self._entries[key]
            self.used -= entry.nbytes
            return key
        return None

    def clear(self) -> None:
        self._entries.clear()
        self._evictable.clear()
        self.used = 0


@dataclass
class StagedSources:
    """Own Producer tensor views and their staging-slab regions.

    Attributes:
        tensors: Registered tensor views used as Mooncake sources.
        regions: Allocator regions released after the write finishes.
    """

    tensors: list[torch.Tensor]
    regions: list[tuple[int, int]]


class ProducerMemoryPool:
    """Own the Producer staging slab and regions carved from it.

    Attributes:
        _capacity: Requested staging-slab size in bytes.
        _transfer: Data-plane owner used to register the slab.
        _pool: Lazily allocated registered byte tensor.
        _allocator: Region allocator for the staging slab.
        _disabled: Whether initialization failed and fallback is required.
        _lock: Lock protecting initialization and region allocation.
    """

    def __init__(self, capacity: int, transfer: MooncakeTransfer) -> None:
        self._capacity = capacity
        self._transfer = transfer
        self._pool: torch.Tensor | None = None
        self._allocator: ContiguousAllocator | None = None
        self._disabled = False
        self._lock = threading.Lock()

    @property
    def tensor(self) -> torch.Tensor | None:
        return self._pool

    def _ensure_pool(self, device: torch.device) -> None:
        if self._pool is not None or self._disabled:
            return
        with self._lock:
            if self._pool is not None or self._disabled:
                return
            try:
                pool = torch.empty(self._capacity, dtype=torch.uint8, device=device)
                ret = self._transfer.register_memory(pool)
                if ret != 0:
                    raise RuntimeError(f"Mooncake returned {ret}")
            except (RuntimeError, torch.OutOfMemoryError) as error:
                self._disabled = True
                logger.warning(
                    "Could not initialize the EC producer staging pool; falling "
                    "back to per-transfer registration: %s",
                    error,
                )
                return
            self._pool = pool
            self._allocator = ContiguousAllocator(pool.nbytes)
            logger.info(
                "Registered %d-byte staging pool for Mooncake EC pushes",
                pool.nbytes,
            )

    def _free_regions(self, regions: list[tuple[int, int]]) -> None:
        assert self._allocator is not None
        for offset, size in regions:
            self._allocator.free(offset, size)

    def stage(self, tensors: list[torch.Tensor]) -> StagedSources | None:
        """Copy tensors into one registered slab, or return None for fallback."""
        if not tensors:
            return StagedSources([], [])
        self._ensure_pool(tensors[0].device)
        pool = self._pool
        allocator = self._allocator
        if pool is None or allocator is None:
            return None
        staged: list[torch.Tensor] = []
        regions: list[tuple[int, int]] = []
        with self._lock:
            for tensor in tensors:
                region = allocator.allocate(tensor.nbytes)
                if region is None:
                    self._free_regions(regions)
                    return None
                regions.append(region)
                staged.append(
                    pool.narrow(0, region[0], tensor.nbytes)
                    .view(tensor.dtype)
                    .view(tensor.shape)
                )
        for destination, source in zip(staged, tensors):
            destination.copy_(source, non_blocking=True)
        return StagedSources(staged, regions)

    def release(self, staged: StagedSources) -> None:
        if not staged.regions:
            return
        with self._lock:
            self._free_regions(staged.regions)

    def close(self) -> None:
        """Retain the registered slab until the full close phase owns it."""


class ConsumerMemoryPool:
    """Own the registered receive slab and resident allocation lifecycle.

    Attributes:
        _capacity: Requested receive-slab size in bytes.
        _transfer: Data-plane owner used to register the slab.
        _metrics: Counters describing resident-cache behavior.
        _pool: Registered byte tensor that receives Mooncake writes.
        _allocator: Region allocator for the receive slab.
        _residents: Published allocations available for local reuse.
        _retire_events: CUDA events guarding retired resident entries.
        _pending_frees: Allocations waiting for CUDA consumers to finish.
        _reclaimed: Cache identifiers evicted under allocation pressure.
        _disabled: Whether receive-slab initialization has failed.
        lock: Reentrant lock shared with reservation state transitions.
    """

    def __init__(
        self,
        capacity: int,
        transfer: MooncakeTransfer,
    ) -> None:
        self._capacity = capacity
        self._transfer = transfer
        self._metrics: Counter[str] = Counter()
        self._pool: torch.Tensor | None = None
        self._allocator: ContiguousAllocator | None = None
        self._residents: ResidentPool[MemoryAllocation] = ResidentPool()
        self._retire_events: dict[str, torch.Event] = {}
        self._pending_frees: list[tuple[torch.Event, MemoryAllocation]] = []
        self._reclaimed: set[str] = set()
        self._disabled = False
        self.lock = threading.RLock()

    @property
    def tensor(self) -> torch.Tensor | None:
        return self._pool

    def prepare(
        self,
        device: torch.device,
        *,
        receiving_rank: bool,
        allow_host: bool = False,
    ) -> None:
        if not receiving_rank:
            return
        if (
            self._pool is not None
            or self._disabled
            or (device.type != "cuda" and not allow_host)
        ):
            return
        try:
            pool = torch.empty(self._capacity, dtype=torch.uint8, device=device)
            ret = self._transfer.register_memory(pool)
            if ret != 0:
                raise RuntimeError(f"Mooncake returned {ret}")
        except (RuntimeError, torch.OutOfMemoryError) as error:
            self._disabled = True
            logger.warning(
                "Could not initialize the EC consumer buffer pool; falling back "
                "to per-tensor registration: %s",
                error,
            )
            return
        self._pool = pool
        self._allocator = ContiguousAllocator(pool.nbytes)
        logger.info(
            "Prepared %d-byte CUDA receive pool for Mooncake EC (registered=%s)",
            pool.nbytes,
            receiving_rank,
        )

    def _free(self, allocation: MemoryAllocation) -> None:
        assert self._allocator is not None
        self._allocator.free(allocation.offset, allocation.size)

    def free(self, allocation: MemoryAllocation) -> None:
        with self.lock:
            self._free(allocation)

    def _defer_or_free(
        self, allocation: MemoryAllocation, event: torch.Event | None
    ) -> None:
        if event is None or event.query():
            self._free(allocation)
        else:
            self._pending_frees.append((event, allocation))

    def _poll_frees_locked(self) -> None:
        pending = []
        for event, allocation in self._pending_frees:
            if event.query():
                self._free(allocation)
            else:
                pending.append((event, allocation))
        self._pending_frees = pending

    def _reclaim_locked(self, nbytes: int) -> tuple[int, int] | None:
        assert self._allocator is not None

        def evict(mm_hash: str, allocation: MemoryAllocation) -> bool:
            event = self._retire_events.pop(mm_hash, None)
            self._defer_or_free(allocation, event)
            self._reclaimed.add(mm_hash)
            self._metrics["residents_reclaimed"] += 1
            return True

        while self._residents.evict_lru(evict) is not None:
            region = self._allocator.allocate(nbytes)
            if region is not None:
                return region
        return None

    def _make_allocation(
        self,
        region: tuple[int, int],
        nbytes: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> MemoryAllocation:
        assert self._pool is not None
        offset, size = region
        tensor = self._pool.narrow(0, offset, nbytes).view(dtype).view(shape)
        return MemoryAllocation(offset, size, tensor)

    def try_allocate(
        self, nbytes: int, shape: tuple[int, ...], dtype: torch.dtype
    ) -> MemoryAllocation | None:
        with self.lock:
            allocator = self._allocator
            assert self._pool is not None and allocator is not None
            self._poll_frees_locked()
            region = allocator.allocate(nbytes)
            if region is None:
                return None
            return self._make_allocation(region, nbytes, shape, dtype)

    def reclaim_and_allocate(
        self, nbytes: int, shape: tuple[int, ...], dtype: torch.dtype
    ) -> MemoryAllocation | None:
        with self.lock:
            region = self._reclaim_locked(nbytes)
            if region is None:
                return None
            return self._make_allocation(region, nbytes, shape, dtype)

    def acquire_cached(
        self, mm_hash: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> ResidentLease[MemoryAllocation] | None:
        with self.lock:
            allocation = self._residents.get(mm_hash)
            if allocation is None:
                return None
            if (
                tuple(allocation.tensor.shape) != shape
                or allocation.tensor.dtype != dtype
            ):
                raise ValueError("conflicting cached tensor for mm_hash")
            return self._residents.acquire(mm_hash)

    def take_resident(
        self, mm_hash: str, shape: tuple[int, ...], dtype_name: str
    ) -> torch.Tensor | None:
        with self.lock:
            allocation = self._residents.get(mm_hash)
            if allocation is None:
                self._metrics["residents_missed"] += 1
                return None
            tensor = allocation.tensor
            if (
                tuple(tensor.shape) != shape
                or str(tensor.dtype).split(".")[-1] != dtype_name
            ):
                self._metrics["residents_mismatched"] += 1
                return None
            self._residents.pin(mm_hash)
            self._retire_events.pop(mm_hash, None)
            self._metrics["residents_promoted"] += 1
            return tensor

    def _record_release_event(self) -> torch.Event | None:
        if self._pool is None or self._pool.device.type != "cuda":
            return None
        event = torch.Event()
        event.record(torch.accelerator.current_stream(self._pool.device))
        return event

    def release_cached(self, lease: ResidentLease[MemoryAllocation]) -> None:
        with self.lock:
            released = self._residents.release(lease)
            if released is not None:
                self._defer_or_free(released, self._record_release_event())

    def publish(
        self,
        mm_hash: str,
        allocation: MemoryAllocation,
        lease: ResidentLease[MemoryAllocation] | None = None,
    ) -> MemoryAllocation:
        with self.lock:
            if lease is not None:
                canonical, released = self._residents.consume(lease)
                self._retire_events.pop(mm_hash, None)
                if released is not None:
                    self._defer_or_free(released, self._record_release_event())
                return canonical
            previous = self._residents.get(mm_hash)
            displaced = self._residents.insert(mm_hash, allocation, allocation.size)
            event = None
            if previous is not None and previous is not allocation:
                event = self._retire_events.pop(mm_hash, None)
            if displaced is None or displaced is allocation:
                return allocation
            if event is None:
                event = self._record_release_event()
            self._defer_or_free(displaced, event)
            return allocation

    def retire_stale(
        self,
        encoder_cache: dict[str, torch.Tensor],
        reserved_hashes: set[str],
    ) -> None:
        if self._pool is None:
            return
        with self.lock:
            for mm_hash in self._residents.referenced():
                allocation = self._residents.get(mm_hash)
                if allocation is None:
                    continue
                if encoder_cache.get(mm_hash) is allocation.tensor:
                    continue
                if mm_hash in reserved_hashes:
                    continue
                event = torch.Event()
                event.record(torch.accelerator.current_stream(self._pool.device))
                self._retire_events[mm_hash] = event
                self._residents.retire(mm_hash)
                self._metrics["residents_retired"] += 1
            self._poll_frees_locked()

    def drain_reclaimed(self) -> set[str]:
        with self.lock:
            reclaimed = self._reclaimed
            self._reclaimed = set()
            return reclaimed

    def stats(self) -> tuple[int, int, int, int]:
        with self.lock:
            return (
                len(self._residents),
                len(self._residents.referenced()),
                self._residents.num_evictable,
                len(self._pending_frees),
            )

    def take_metrics(self) -> dict[str, int]:
        with self.lock:
            metrics = dict(self._metrics)
            self._metrics.clear()
            return metrics

    def close(self) -> None:
        with self.lock:
            pool = self._pool
            if pool is None or not self._transfer.unregister_memory(pool):
                return
            self._pool = None
            self._allocator = None
            self._residents.clear()
            self._retire_events.clear()
            self._pending_frees.clear()
