# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Paged shared memory manager with LRU eviction.

Write flow:
- open_write(items):
    allocates blocks atomically for multiple items, sets ref_count=-1.
- close_write(uuid):
    finalizes write; if not open_read, sets ref_count=0 and caches if eligible.

Read flow:
- open_read(uuid): increments ref_count, removes from cache if idle.
- close_read(uuid): decrements ref_count; if becomes 0, re-caches if cacheable.

Additional: pin/unpin prevent eviction; delete removes idle items.
"""

from collections import deque
from typing import Any

from vllm.utils.cache import LRUCache

from .types import ShmSlot, ShmWriteRequest

# Special reference count values
REF_WRITING = -1  # Item is being written (not yet readable)
REF_IDLE = 0  # Item is idle and may be cached or pinned


class PagedShmManager:
    """Manages a fixed-size, paged shm pool with LRU eviction."""

    def __init__(self, size: int, block_size: int):
        assert block_size > 0

        self.block_size = block_size
        self.n_block = size // block_size
        self.size = block_size * self.n_block

        assert self.size > 0
        assert self.n_block > 0

        # uuid -> ShmSlot
        self._all_items: dict[str, ShmSlot] = {}

        # Initially all blocks are free
        self._free_blocks = deque(range(self.n_block))
        self._total_available_blocks = self.n_block

        # LRU cache tracks idle cacheable items by their block count.
        self._lru_cache: LRUCache[str, ShmSlot] = LRUCache(
            capacity=self.n_block,
            getsizeof=lambda x: x.n_block(),
        )
        self._pinned_items: set[str] = set()

    def open_write(self, items: list[ShmWriteRequest]) -> list[ShmSlot]:
        """
        Allocate blocks for a batch of items. To avoid partial allocation,
        submit multiple items in one batch.
        Refer to the wiki: Dining philosophers problem.
        """
        # 0. Confirm there are no UUID conflicts with existing items.
        for item in items:
            if item.uuid in self._all_items:
                raise ValueError(f"UUID {item.uuid} already exists")

            if item.size <= 0:
                raise ValueError(f"item size {item.size} must be greater than zero.")

        # 1. Calculate required number of blocks for each item and total demand.
        needs = []
        total_need = 0
        for item in items:
            need = (item.size + self.block_size - 1) // self.block_size
            needs.append(need)
            total_need += need

        # 2. Confirm whether there is sufficient space to meet all requirements.
        if self._total_available_blocks < total_need:
            raise MemoryError(
                f"Not enough blocks: need {total_need}, "
                f"available {self._total_available_blocks}"
            )

        # 3. Evict cached items until enough free blocks are available.
        self._evict(total_need)

        # 4. Allocate blocks and record.
        allocated: list[ShmSlot] = []
        for idx, item in enumerate(items):
            need = needs[idx]
            blocks = [self._free_blocks.popleft() for _ in range(need)]
            new_item = ShmSlot(
                uuid=item.uuid,
                size=item.size,
                use_cache=item.use_cache,
                blocks=blocks,
                ref_count=REF_WRITING,
            )
            self._all_items[item.uuid] = new_item
            allocated.append(new_item)

        # Total available blocks decrease by what we just handed out
        self._total_available_blocks -= total_need
        return allocated

    def close_write(self, uuid: str, open_read: bool = False):
        """
        Finalize a write operation. If open_read is False, the item becomes idle
        and may be cached; if open_read is True, it gets one reader reference.
        """
        item = self._get_item(uuid)
        if item.ref_count != REF_WRITING:
            raise ValueError(f"UUID {uuid} not being written")

        if not open_read:
            item.ref_count = REF_IDLE
            # Insert into LRU cache if caching is enabled and item is not pinned
            if item.use_cache and uuid not in self._pinned_items:
                self._total_available_blocks += item.n_block()
                self._lru_cache.put(uuid, item)
        else:
            item.ref_count = 1  # start with one reader

    def open_read(self, uuid: str) -> ShmSlot:
        """
        Increment the read reference count. If the item is idle and cacheable,
        it is removed from the LRU cache (making its blocks unavailable for eviction).
        """
        item = self._get_item(uuid)
        if item.ref_count == REF_WRITING:
            raise ValueError(f"UUID {uuid} is being written")

        # If the item is idle and cacheable, take it out of the cache
        update_cache = (
            item.use_cache
            and item.ref_count == REF_IDLE
            and uuid not in self._pinned_items
        )
        if update_cache:
            self._lru_cache.pop(uuid)
            self._total_available_blocks -= len(item.blocks)

        item.ref_count += 1
        return item

    def close_read(self, uuid: str):
        """
        Decrement the read reference count. If the count drops to zero and the
        item is cacheable, it is put back into the LRU cache.
        """
        item = self._get_item(uuid)
        if item.ref_count == REF_WRITING:
            raise ValueError(f"UUID {uuid} being written")
        if item.ref_count == REF_IDLE:
            raise ValueError(f"UUID {uuid} not being read")

        if item.ref_count > 0:
            item.ref_count -= 1

        # If the item is now idle and cacheable, put it back into the cache
        update_cache = (
            item.use_cache
            and item.ref_count == REF_IDLE
            and uuid not in self._pinned_items
        )
        if update_cache:
            self._total_available_blocks += len(item.blocks)
            self._lru_cache.put(uuid, item)

    def pin(self, uuid: str):
        """
        Pin an item so it will not be evicted. Only applicable if use_cache is True.
        """
        item = self._get_item(uuid)

        if not item.use_cache:
            return

        if uuid in self._pinned_items:
            return

        self._pinned_items.add(uuid)

        # If the item is currently in the LRU cache, remove it
        if item.ref_count == REF_IDLE:
            self._lru_cache.pop(uuid)
            self._total_available_blocks -= len(item.blocks)

    def unpin(self, uuid: str):
        """
        Unpin an item. If the item becomes idle and cacheable, re‑insert it into LRU.
        If the item is not cacheable and idle, it is deleted immediately.
        """
        item = self._get_item(uuid)

        if not item.use_cache and item.ref_count == REF_IDLE:
            self.delete(uuid)
            return

        if uuid not in self._pinned_items:
            return

        self._pinned_items.discard(uuid)

        # If the item is idle, re‑insert it into the LRU cache
        if item.ref_count == REF_IDLE:
            self._total_available_blocks += len(item.blocks)
            self._lru_cache.put(uuid, item)

    def delete(self, uuid: str):
        """
        Permanently delete an item. Its blocks are returned to the free pool.
        The item must be idle (ref_count == REF_IDLE).
        """
        item = self._get_item(uuid)
        if item.ref_count != REF_IDLE:
            raise ValueError(f"UUID {uuid} is busy now")

        # If the item was not cached (or was pinned) its blocks were counted
        # as unavailable; now they become truly free.
        if not item.use_cache or uuid in self._pinned_items:
            self._total_available_blocks += item.n_block()

        # Remove from all tracking structures.
        self._lru_cache.pop(uuid, None)
        self._pinned_items.discard(uuid)
        self._all_items.pop(uuid)
        self._free_blocks.extend(item.blocks)

    def get_info(self, uuid: str) -> dict[str, Any]:
        item = self._all_items.get(uuid, None)
        if item is None:
            raise ValueError(f"UUID {uuid} not found")

        return {
            "uuid": item.uuid,
            "size": item.size,
            "use_cache": item.use_cache,
            "ref_count": item.ref_count,
        }

    def get_manager_state(self) -> dict[str, int]:
        """Return aggregated statistics about the manager."""
        idle_count = 0
        writing_count = 0
        reading_count = 0
        cached_blocks = self._total_available_blocks - len(self._free_blocks)

        for item in self._all_items.values():
            if item.ref_count == REF_WRITING:
                writing_count += 1
            elif item.ref_count == REF_IDLE:
                idle_count += 1
            else:
                reading_count += 1

        return {
            "size": self.size,
            "block_size": self.block_size,
            "n_block": self.n_block,
            "free_blocks_count": len(self._free_blocks),
            "total_available_blocks": self._total_available_blocks,
            "cached_items_count": len(self._lru_cache),
            "cached_blocks_count": cached_blocks,
            "pinned_items_count": len(self._pinned_items),
            "total_items_count": len(self._all_items),
            "idle_items_count": idle_count,
            "writing_items_count": writing_count,
            "reading_items_count": reading_count,
        }

    def _evict(self, needed: int) -> None:
        """
        Evict least-recently-used cacheable items until at least `needed`
        physical free blocks are available.
        """
        while len(self._free_blocks) < needed:
            uuid, victim = self._lru_cache.popitem()

            # Pinned items are never placed in the LRU cache, so they
            # cannot appear here. No change to _total_available_blocks;
            # we just convert evictable blocks into physically free ones.
            self._all_items.pop(uuid)
            self._free_blocks.extend(victim.blocks)

    def _get_item(self, uuid: str) -> ShmSlot:
        """Return the item, or raise ValueError if not found."""
        item = self._all_items.get(uuid)
        if item is None:
            raise ValueError(f"UUID {uuid} not found")
        return item
