# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered-memory allocation and residency for Mooncake transfers.

The Producer pool stages source tensors in a registered slab.  The Consumer
pool owns destination allocations from reservation through publication,
resident reuse, CUDA-safe retirement, and pressure-driven reclamation.
"""

from __future__ import annotations

import bisect
import threading
from collections import OrderedDict
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
    """Describe a tensor view carved from the Consumer receive slab."""

    offset: int
    size: int
    tensor: torch.Tensor


@dataclass
class _ResidentEntry(Generic[_T]):
    """Store one resident value and its ownership accounting."""

    value: _T
    pinned: bool = True
    leases: int = 0


@dataclass
class ResidentLease(Generic[_T]):
    """Represent a borrow of one resident entry."""

    key: str
    _entry: _ResidentEntry[_T]
    _active: bool = True

    @property
    def value(self) -> _T:
        return self._entry.value


class ContiguousAllocator:
    """Own a registered slab and its aligned regions; callers serialize access."""

    def __init__(self, capacity: int, alignment: int = 256):
        self.alignment = alignment
        self._capacity = capacity
        self._free = [(0, capacity)]
        self.tensor: torch.Tensor | None = None
        self._disabled = False

    def prepare(self, device: torch.device, transfer: MooncakeTransfer) -> None:
        if self.tensor is not None or self._disabled:
            return
        try:
            tensor = torch.empty(self._capacity, dtype=torch.uint8, device=device)
            ret = transfer.register_memory(tensor)
            if ret != 0:
                raise RuntimeError(f"Mooncake returned {ret}")
        except (RuntimeError, torch.OutOfMemoryError) as error:
            self._disabled = True
            logger.warning("Could not initialize the EC registered buffer: %s", error)
            return
        self.tensor = tensor
        self._free = [(0, tensor.nbytes)]
        logger.info("Registered %d-byte buffer for Mooncake EC", tensor.nbytes)

    def view(
        self, offset: int, nbytes: int, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        assert self.tensor is not None
        return self.tensor.narrow(0, offset, nbytes).view(dtype).view(shape)

    def close(self, transfer: MooncakeTransfer) -> bool:
        if self.tensor is not None and not transfer.unregister_memory(self.tensor):
            return False
        self.tensor = None
        self._free.clear()
        return True

    def allocate(self, nbytes: int) -> tuple[int, int] | None:
        size = (nbytes + self.alignment - 1) // self.alignment * self.alignment
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
    """Track resident values, active leases, and LRU eviction eligibility."""

    def __init__(self):
        self._entries: dict[str, _ResidentEntry[_T]] = {}
        self._evictable: OrderedDict[str, None] = OrderedDict()

    def referenced(self) -> list[str]:
        return [key for key, entry in self._entries.items() if entry.pinned]

    def get(self, key: str) -> _T | None:
        entry = self._entries.get(key)
        return entry.value if entry is not None else None

    def insert(
        self,
        key: str,
        value: _T,
    ) -> _T | None:
        """Pin an entry and return a displaced value that has no owners."""
        previous = self._entries.get(key)
        entry = _ResidentEntry(value)
        self._entries[key] = entry
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
            return key
        return None

    def clear(self) -> None:
        self._entries.clear()
        self._evictable.clear()


@dataclass
class StagedSources:
    """Own Producer tensor views and their staging-slab regions."""

    tensors: list[torch.Tensor]
    regions: list[tuple[int, int]]


class ProducerMemoryPool:
    """Own the Producer staging slab and regions carved from it."""

    def __init__(self, capacity: int, transfer: MooncakeTransfer) -> None:
        self._transfer = transfer
        self._allocator = ContiguousAllocator(capacity)
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def tensor(self) -> torch.Tensor | None:
        return self._allocator.tensor

    def _free_regions(self, regions: list[tuple[int, int]]) -> None:
        for offset, size in regions:
            self._allocator.free(offset, size)

    def stage(self, tensors: list[torch.Tensor]) -> StagedSources | None:
        """Copy tensors into one registered slab, or return None for fallback."""
        if not tensors:
            return StagedSources([], [])
        allocator = self._allocator
        staged: list[torch.Tensor] = []
        regions: list[tuple[int, int]] = []
        with self._lock:
            allocator.prepare(tensors[0].device, self._transfer)
            pool = allocator.tensor
            if pool is None:
                return None
            for tensor in tensors:
                region = allocator.allocate(tensor.nbytes)
                if region is None:
                    self._free_regions(regions)
                    return None
                regions.append(region)
                staged.append(
                    allocator.view(
                        region[0], tensor.nbytes, tuple(tensor.shape), tensor.dtype
                    )
                )
        if pool.device.type == "cuda":
            stream = getattr(self._local, "stream", None)
            if stream is None:
                stream = torch.cuda.Stream(device=pool.device)
                self._local.stream = stream
            with torch.cuda.stream(stream):
                for destination, source in zip(staged, tensors):
                    destination.copy_(source, non_blocking=True)
            stream.synchronize()
        else:
            for destination, source in zip(staged, tensors):
                destination.copy_(source)
        return StagedSources(staged, regions)

    def release(self, staged: StagedSources) -> None:
        if not staged.regions:
            return
        with self._lock:
            self._free_regions(staged.regions)

    def close(self) -> None:
        with self._lock:
            self._allocator.close(self._transfer)


class ConsumerMemoryPool:
    """Own the registered receive slab and resident allocation lifecycle."""

    def __init__(
        self,
        capacity: int,
        transfer: MooncakeTransfer,
    ) -> None:
        self._transfer = transfer
        self._allocator = ContiguousAllocator(capacity)
        self._residents: ResidentPool[MemoryAllocation] = ResidentPool()
        self._retire_events: dict[str, torch.Event] = {}
        self._pending_frees: list[tuple[torch.Event, MemoryAllocation]] = []
        self._reclaimed: set[str] = set()
        self.lock = threading.RLock()

    @property
    def tensor(self) -> torch.Tensor | None:
        return self._allocator.tensor

    def prepare(
        self,
        device: torch.device,
    ) -> None:
        with self.lock:
            self._allocator.prepare(device, self._transfer)

    def _free(self, allocation: MemoryAllocation) -> None:
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
        def evict(mm_hash: str, allocation: MemoryAllocation) -> bool:
            event = self._retire_events.pop(mm_hash, None)
            self._defer_or_free(allocation, event)
            self._reclaimed.add(mm_hash)
            return True

        while self._residents.evict_lru(evict) is not None:
            # Eviction defers any free whose CUDA event is still pending, so
            # those bytes reach the allocator only once the event is polled.
            self._poll_frees_locked()
            region = self._allocator.allocate(nbytes)
            if region is not None:
                return region
        self._poll_frees_locked()
        return self._allocator.allocate(nbytes)

    def _make_allocation(
        self,
        region: tuple[int, int],
        nbytes: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> MemoryAllocation:
        offset, size = region
        tensor = self._allocator.view(offset, nbytes, shape, dtype)
        return MemoryAllocation(offset, size, tensor)

    def try_allocate(
        self, nbytes: int, shape: tuple[int, ...], dtype: torch.dtype
    ) -> MemoryAllocation | None:
        with self.lock:
            allocator = self._allocator
            assert allocator.tensor is not None
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
                return None
            tensor = allocation.tensor
            if (
                tuple(tensor.shape) != shape
                or str(tensor.dtype).split(".")[-1] != dtype_name
            ):
                return None
            self._residents.pin(mm_hash)
            self._retire_events.pop(mm_hash, None)
            return tensor

    def _record_release_event(self) -> torch.Event | None:
        pool = self.tensor
        if pool is None or pool.device.type != "cuda":
            return None
        event = torch.Event()
        event.record(torch.accelerator.current_stream(pool.device))
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
        *,
        pin: bool = True,
    ) -> MemoryAllocation:
        with self.lock:
            if lease is not None:
                canonical, released = self._residents.consume(lease)
                self._retire_events.pop(mm_hash, None)
                if released is not None:
                    self._defer_or_free(released, self._record_release_event())
                return canonical
            previous = self._residents.get(mm_hash)
            displaced = self._residents.insert(mm_hash, allocation)
            if not pin:
                self._residents.retire(mm_hash)
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
        reserved_hashes: set[str] | None = None,
        *,
        freed: list[str] | None = None,
    ) -> None:
        if self.tensor is None:
            return
        with self.lock:
            for mm_hash in self._residents.referenced() if freed is None else freed:
                allocation = self._residents.get(mm_hash)
                if allocation is None:
                    continue
                if encoder_cache.get(mm_hash) is allocation.tensor:
                    continue
                if reserved_hashes and mm_hash in reserved_hashes:
                    continue
                event = self._record_release_event()
                if event is not None:
                    self._retire_events[mm_hash] = event
                self._residents.retire(mm_hash)
            self._poll_frees_locked()

    def drain_reclaimed(self) -> set[str]:
        with self.lock:
            reclaimed = self._reclaimed
            self._reclaimed = set()
            return reclaimed

    def close(self) -> None:
        with self.lock:
            if not self._allocator.close(self._transfer):
                return
            self._residents.clear()
            self._retire_events.clear()
            self._pending_frees.clear()
