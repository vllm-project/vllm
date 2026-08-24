# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Paged shared memory manager with LRU eviction.

Write flow:
- open_write(items):
    allocates blocks atomically for multiple items, sets ref_count=-1.
- close_write(uuid):
    finalizes write; if not open_read, sets ref_count=0 and caches if eligible.
    For non-cacheable items (use_cache=False) with no read references,
    the item is deleted immediately to free resources.

Read flow:
- open_read(uuid): increments ref_count, removes from cache if idle.
- close_read(uuid): decrements ref_count; if becomes 0, re-caches if cacheable,
                     else deletes immediately if non-cacheable.

Additional: delete removes idle items.
"""

from collections import deque
from typing import Any

from vllm.utils.cache import LRUCache

from .types import ShmSlot, ShmWriteRequest

# Special reference count values
_REF_WRITING = -1  # Item is being written (not yet readable)
_REF_IDLE = 0  # Item is idle and may be cached


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
                ref_count=_REF_WRITING,
            )
            self._all_items[item.uuid] = new_item
            allocated.append(new_item)

        # Total available blocks decrease by what we just handed out
        self._total_available_blocks -= total_need
        return allocated

    def close_write(self, uuid: str, open_n_reads: int = 0):
        """
        Finalize a write operation. If open_read is False, the item becomes idle
        and may be cached.
        For non-cacheable items (use_cache=False) with no open reads, delete
        immediately to free resources.
        """
        item = self._get_item(uuid)
        if item.ref_count != _REF_WRITING:
            raise ValueError(f"UUID {uuid} not being written")

        if open_n_reads <= 0:
            if item.use_cache:
                # Normal cacheable idle item: put into LRU
                item.ref_count = _REF_IDLE
                self._total_available_blocks += item.n_block()
                self._lru_cache.put(uuid, item)
            else:
                # Non-cacheable: release immediately
                # Set ref_count to idle to allow delete, then delete
                item.ref_count = _REF_IDLE
                self.delete(uuid, force=False)
        else:
            item.ref_count = open_n_reads

    def open_read(self, uuid: str) -> ShmSlot:
        """
        Increment the read reference count. If the item is idle and cacheable,
        it is removed from the LRU cache (making its blocks unavailable for eviction).
        """
        item = self._get_item(uuid)
        if item.ref_count == _REF_WRITING:
            raise ValueError(f"UUID {uuid} is being written")

        # If the item is idle and cacheable, take it out of the cache
        update_cache = item.use_cache and item.ref_count == _REF_IDLE
        if update_cache:
            self._lru_cache.pop(uuid)
            self._total_available_blocks -= len(item.blocks)

        item.ref_count += 1
        return item

    def close_read(self, uuid: str):
        """
        Decrement the read reference count. If the count drops to zero:
          - For cacheable items: put back into LRU.
          - For non-cacheable items: delete immediately to free blocks.
        """
        item = self._get_item(uuid)
        if item.ref_count == _REF_WRITING:
            raise ValueError(f"UUID {uuid} being written")
        if item.ref_count == _REF_IDLE:
            raise ValueError(f"UUID {uuid} not being read")

        if item.ref_count > 0:
            item.ref_count -= 1

        # Now handle the case when ref_count becomes idle
        if item.ref_count == _REF_IDLE:
            if not item.use_cache:
                # Non-cacheable item is no longer needed
                self.delete(uuid, force=False)
                return
            # Cacheable: re-insert
            self._total_available_blocks += len(item.blocks)
            self._lru_cache.put(uuid, item)

    def delete(self, uuid: str, force: bool = False):
        """
        Permanently delete an item. Its blocks are returned to the free pool.
        If force=True, the item is deleted regardless of ref_count.
        """
        item = self._get_item(uuid)
        if not force and item.ref_count != _REF_IDLE:
            raise ValueError(f"UUID {uuid} is busy now")

        # Check whether the item was in the LRU cache (its blocks are already
        # accounted for in _total_available_blocks).
        was_cached = self._lru_cache.pop(uuid, None) is not None

        # If it was not cached, its blocks are not yet counted as available,
        # so we must add them back now.
        if not was_cached:
            self._total_available_blocks += item.n_block()

        # Remove from all tracking structures.
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
            if item.ref_count == _REF_WRITING:
                writing_count += 1
            elif item.ref_count == _REF_IDLE:
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
            self._all_items.pop(uuid)
            self._free_blocks.extend(victim.blocks)

    def _get_item(self, uuid: str) -> ShmSlot:
        """Return the item, or raise ValueError if not found."""
        item = self._all_items.get(uuid)
        if item is None:
            raise ValueError(f"UUID {uuid} not found")
        return item

    def _get_item_blocks_copy(self, uuid: str) -> tuple[int, list[int]]:
        """
        This is used for PagedShmServer token-based open_read.
        """
        item = self._all_items.get(uuid)
        if item is None:
            raise ValueError(f"UUID {uuid} not found")
        if item.ref_count == _REF_WRITING:
            raise RuntimeError(f"Item {uuid} is still being written")
        return item.size, item.blocks.copy()
