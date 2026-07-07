# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Paged shared memory manager with LRU eviction.

Manages a fixed-size shared memory pool divided into equal-sized blocks.
Items can be allocated, written, read (with reference counting), pinned,
and deleted.  A block‑level LRU cache automatically evicts idle cacheable
items when free blocks run low.
"""

from collections import deque
from dataclasses import dataclass

from vllm.utils.cache import LRUCache


@dataclass
class Item:
    uuid: str
    size: int
    use_cache: bool  # whether the item should be cached after writing


@dataclass
class AllocatedItem(Item):
    """
    Represents an allocated item with block assignments and a reference count.

    ref_count == 0: item can be evicted from cache.
    ref_count > 0: item is currently being read by one or more users.
    ref_count < 0: item is currently being written (exclusive access).
    """

    blocks: list[int]
    ref_count: int = 0

    def n_block(self):
        return len(self.blocks)


class PagedShmManager:
    """Manages a fixed-size, paged shm pool with LRU eviction."""

    def __init__(self, size: int, block_size: int):
        assert block_size > 0

        self.block_size = block_size
        self.n_block = size // block_size
        self.size = block_size * self.n_block

        assert self.size > 0
        assert self.n_block > 0

        # uuid -> AllocatedItem
        self._all_items: dict[str, AllocatedItem] = {}

        # Initially all blocks are free
        self._free_blocks = deque(range(self.n_block))
        self._total_available_blocks = self.n_block

        # LRU cache tracks idle cacheable items by their block count.
        self._lru_cache: LRUCache[str, AllocatedItem] = LRUCache(
            capacity=self.n_block,
            getsizeof=lambda x: x.n_block(),
        )
        self._pinned_items: set[str] = set()

    def open_write(self, items: list[Item]) -> list[AllocatedItem]:
        """Allocate blocks for a batch of items for write.

        Note:
            If a request requires allocating multiple items, be sure to submit
            the allocation request as a single batch to avoid deadlock
            refer to the wiki:Dining philosophers problem.
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
            raise MemoryError("No sufficient space to meet all requirements")

        # 3. Evict cached items until enough free blocks are available.
        self._evict(total_need)

        # 4. Allocate blocks and record.
        allocated: list[AllocatedItem] = []
        for idx, item in enumerate(items):
            need = needs[idx]
            blocks = [self._free_blocks.popleft() for _ in range(need)]
            new_item = AllocatedItem(
                uuid=item.uuid,
                size=item.size,
                use_cache=item.use_cache,
                blocks=blocks,
                ref_count=-1,  # Item is being written
            )
            self._all_items[item.uuid] = new_item
            allocated.append(new_item)

        # Total available blocks decrease by what we just handed out
        self._total_available_blocks -= total_need
        return allocated

    def close_write(self, uuid: str):
        item = self._all_items.get(uuid, None)
        if item is None:
            raise ValueError(f"UUID {uuid} not found")
        if item.ref_count >= 0:
            raise ValueError(f"UUID {uuid} not being written")

        item.ref_count = 0

        # Insert into LRU cache if caching is enabled and item is not pinned
        if item.use_cache and uuid not in self._pinned_items:
            self._total_available_blocks += item.n_block()
            self._lru_cache.put(uuid, item)

    def open_read(self, uuid):
        item: AllocatedItem = self._all_items.get(uuid, None)
        if item is None:
            raise ValueError(f"UUID {uuid} not found")
        if item.ref_count < 0:
            raise ValueError(f"UUID {uuid} is being written")

        # If the item is idle and cacheable/pinned, take it out of the cache
        update_cache = (
            item.use_cache and item.ref_count == 0 and uuid not in self._pinned_items
        )
        if update_cache:
            self._lru_cache.pop(uuid)
            self._total_available_blocks -= len(item.blocks)

        item.ref_count += 1
        return item

    def close_read(self, uuid: str):
        item = self._all_items.get(uuid, None)
        if item is None:
            raise ValueError(f"UUID {uuid} not found")
        if item.ref_count < 0:
            raise ValueError(f"UUID {uuid} being written")
        if item.ref_count == 0:
            raise ValueError(f"UUID {uuid} not being read")

        if item.ref_count > 0:
            item.ref_count -= 1

        # If the item is now idle and cacheable/pinned, put it back into the cache
        update_cache = (
            item.use_cache and item.ref_count == 0 and uuid not in self._pinned_items
        )
        if update_cache:
            self._total_available_blocks += len(item.blocks)
            self._lru_cache.put(uuid, item)

    def pin(self, uuid: str):
        item = self._all_items.get(uuid, None)
        if item is None:
            raise ValueError(f"UUID {uuid} not found")

        if not item.use_cache:
            return

        if uuid in self._pinned_items:
            return

        self._pinned_items.add(uuid)

        # If the item is currently in the LRU cache, remove it
        if item.ref_count == 0:
            self._lru_cache.pop(uuid)
            self._total_available_blocks -= len(item.blocks)

    def unpin(self, uuid: str):
        item = self._all_items.get(uuid, None)
        if item is None:
            raise ValueError(f"UUID {uuid} not found")

        if not item.use_cache and item.ref_count == 0:
            self.delete(uuid)
            return

        if uuid not in self._pinned_items:
            return

        self._pinned_items.discard(uuid)

        # If the item is idle, re‑insert it into the LRU cache
        if item.ref_count == 0:
            self._total_available_blocks += len(item.blocks)
            self._lru_cache.put(uuid, item)

    def delete(self, uuid: str):
        item = self._all_items.get(uuid, None)
        if item is None:
            raise ValueError(f"UUID {uuid} not found")
        if item.ref_count != 0:
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

    def info(self, uuid: str) -> AllocatedItem:
        item = self._all_items.get(uuid, None)
        if item is None:
            raise ValueError(f"UUID {uuid} not found")

        return item

    def get_manager_state(self) -> dict[str, int]:
        cached_blocks = self._total_available_blocks - len(self._free_blocks)
        writing_count = 0
        idle_count = 0
        reading_count = 0
        for item in self._all_items.values():
            if item.ref_count < 0:
                writing_count += 1
            elif item.ref_count == 0:
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
            "writing_items_count": writing_count,
            "reading_items_count": reading_count,
            "idle_items_count": idle_count,
        }

    def _evict(self, needed: int) -> None:
        while len(self._free_blocks) < needed:
            uuid, victim = self._lru_cache.popitem()

            # Pinned items are never placed in the LRU cache, so they
            # cannot appear here.  No change to _total_available_blocks;
            # we just convert evictable blocks into physically free ones.
            self._all_items.pop(uuid)
            self._free_blocks.extend(victim.blocks)
