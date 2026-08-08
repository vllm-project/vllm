# SPDX-License-Identifier: Apache-2.0
"""Fixed-page allocator for spatially robust compact CPU KV storage."""

from __future__ import annotations

import dataclasses
import heapq
import itertools
from collections.abc import Sequence

_allocator_ids = itertools.count()


@dataclasses.dataclass(frozen=True)
class PageAllocation:
    allocator_id: int
    id: int
    page_ids: tuple[int, ...]
    logical_length: int
    allocated_length: int


class FixedPageAllocator:
    """Allocate logical payloads over interchangeable fixed-size pages."""

    def __init__(self, total_bytes: int, page_size: int) -> None:
        if total_bytes <= 0:
            raise ValueError("total_bytes must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if total_bytes % page_size:
            raise ValueError("total_bytes must be divisible by page_size")
        self._total_bytes = total_bytes
        self._page_size = page_size
        self._allocator_id = next(_allocator_ids)
        self._free_pages = list(range(total_bytes // page_size))
        heapq.heapify(self._free_pages)
        self._allocated: dict[int, PageAllocation] = {}
        self._next_handle_id = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def free_bytes(self) -> int:
        return len(self._free_pages) * self._page_size

    @property
    def used_bytes(self) -> int:
        return self._total_bytes - self.free_bytes

    @property
    def largest_free_block(self) -> int:
        # Every free page is independently usable; no contiguous-fit requirement.
        return self.free_bytes

    @property
    def fragmentation(self) -> float:
        return 0.0

    @property
    def num_active_handles(self) -> int:
        return len(self._allocated)

    def pages_required(self, size: int) -> int:
        if size <= 0:
            raise ValueError("allocation size must be positive")
        return (size + self._page_size - 1) // self._page_size

    def allocate(self, size: int) -> PageAllocation | None:
        num_pages = self.pages_required(size)
        if num_pages > len(self._free_pages):
            return None
        page_ids = tuple(heapq.heappop(self._free_pages) for _ in range(num_pages))
        allocation = PageAllocation(
            allocator_id=self._allocator_id,
            id=self._next_handle_id,
            page_ids=page_ids,
            logical_length=size,
            allocated_length=num_pages * self._page_size,
        )
        self._allocated[allocation.id] = allocation
        self._next_handle_id += 1
        return allocation

    def free(self, allocation: PageAllocation) -> None:
        if allocation.allocator_id != self._allocator_id:
            raise ValueError("allocation belongs to a different page allocator")
        existing = self._allocated.get(allocation.id)
        if existing is None:
            raise ValueError("unknown or already-freed page allocation")
        if existing != allocation:
            raise ValueError("page allocation field mismatch")
        self._allocated.pop(allocation.id)
        for page_id in allocation.page_ids:
            heapq.heappush(self._free_pages, page_id)

    def reset(self) -> None:
        self._free_pages = list(range(self._total_bytes // self._page_size))
        heapq.heapify(self._free_pages)
        self._allocated.clear()

    def simulate_batch_allocation(
        self,
        sizes: Sequence[int],
        frees: Sequence[PageAllocation] | None = None,
    ) -> bool:
        if any(size <= 0 for size in sizes):
            raise ValueError("allocation size must be positive")
        available_pages = len(self._free_pages)
        seen: set[int] = set()
        for allocation in frees or ():
            if allocation.allocator_id != self._allocator_id:
                raise ValueError("candidate belongs to a different page allocator")
            existing = self._allocated.get(allocation.id)
            if existing is None:
                raise ValueError("candidate free has unknown allocation")
            if existing != allocation:
                raise ValueError("candidate free field mismatch")
            if allocation.id in seen:
                raise ValueError("duplicate candidate page allocation")
            seen.add(allocation.id)
            available_pages += len(allocation.page_ids)
        required_pages = sum(self.pages_required(size) for size in sizes)
        return required_pages <= available_pages

    def page_spans(
        self, allocation: PageAllocation
    ) -> tuple[tuple[int, int, int], ...]:
        """Return coalesced ``(offset, logical, allocated)`` physical spans."""
        if allocation.allocator_id != self._allocator_id:
            raise ValueError("allocation belongs to a different page allocator")
        existing = self._allocated.get(allocation.id)
        if existing is None or existing != allocation:
            raise ValueError("unknown page allocation")
        runs: list[tuple[int, int]] = []
        for _, group in itertools.groupby(
            enumerate(allocation.page_ids), lambda item: item[1] - item[0]
        ):
            pages = [item[1] for item in group]
            runs.append((pages[0], len(pages)))

        remaining = allocation.logical_length
        spans: list[tuple[int, int, int]] = []
        for first_page, num_pages in runs:
            allocated = num_pages * self._page_size
            logical = min(remaining, allocated)
            spans.append((first_page * self._page_size, logical, allocated))
            remaining -= logical
        assert remaining == 0
        return tuple(spans)
