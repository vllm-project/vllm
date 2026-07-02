# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from dataclasses import dataclass

from vllm.utils.cache import LRUCache


@dataclass
class Item:
    uuid: str
    size: int
    cached: bool


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


class PagedSHMManager:
    """Manages a fixed-size, paged shm pool with LRU eviction."""

    def __init__(self, size: int, block_size: int):
        assert block_size > 0

        self.block_size = block_size
        self.n_block = size // block_size
        self.size = block_size * self.n_block

        assert self.size > 0
        assert self.n_block > 0

        # Initially all blocks are free
        self.free_blocks = deque(range(self.n_block))
        self.n_free_block = self.n_block

        # LRU cache capacity is measured in number of blocks, not items
        self.lru_cache: LRUCache[str, AllocatedItem] = LRUCache(
            capacity=self.n_block,
            getsizeof=lambda x: x.n_block(),
        )

        self.all_items: dict[str, AllocatedItem] = {}

    def allocate(self, items: list[Item]) -> list[AllocatedItem]:
        """Allocate blocks for a batch of items for write.

        Note:
            If a request requires allocating multiple items, be sure to submit
            the allocation request as a single batch to avoid deadlock
            refer to the wiki:Dining philosophers problem.
        """
        # 0. Confirm there are no UUID conflicts with existing items
        for item in items:
            if item.uuid in self.all_items:
                raise ValueError(f"UUID {item.uuid} already exists")

            if item.size <= 0:
                raise ValueError(f"item size {item.size} must be greater than zero.")

        # 1. Calculate required number of blocks for each item and total demand
        needs = []
        total_need = 0
        for item in items:
            need = (item.size + self.block_size - 1) // self.block_size
            needs.append(need)
            total_need += need

        # 2. Confirm whether there is sufficient space to meet all requirements.
        if self.n_free_block < total_need:
            raise MemoryError("No sufficient space to meet all requirements")

        # 3. Evict cached items until enough free blocks are available.
        self._evict(total_need)

        # 4. Allocate blocks and record.
        allocated: list[AllocatedItem] = []
        for idx, item in enumerate(items):
            need = needs[idx]
            blocks = [self.free_blocks.popleft() for _ in range(need)]
            new_item = AllocatedItem(
                uuid=item.uuid,
                size=item.size,
                cached=item.cached,
                blocks=blocks,
                ref_count=-1,  # Item is being written
            )
            self.all_items[item.uuid] = new_item
            allocated.append(new_item)

        self.n_free_block -= total_need
        return allocated

    def borrow(self, uuid):
        item: AllocatedItem = self.all_items.get(uuid, None)
        if item is None:
            raise ValueError(f"UUID {uuid} not found")
        if item.ref_count < 0:
            raise ValueError(f"UUID {uuid} is being written, cannot borrow")

        # Pin the item if it is currently evictable
        if item.ref_count == 0:
            self.lru_cache.pin(uuid)

        item.ref_count += 1
        self.lru_cache.touch(uuid)
        self.n_free_block -= len(item.blocks)

    def restore(self, uuid: str):
        item = self.all_items.get(uuid, None)
        if item is None:
            raise ValueError(f"UUID {uuid} not found")

        if not item.cached:
            # Uncached items can be released only when no one is reading them
            self.all_items.pop(uuid)
            self.free_blocks.extend(item.blocks)
            self.n_free_block += len(item.blocks)
        else:
            if item.ref_count < 0:
                # Writing completed
                item.ref_count = 0
                self.lru_cache.put(uuid, item)

            elif item.ref_count > 0:
                item.ref_count -= 1

                if item.ref_count == 0:
                    self.lru_cache._unpin(uuid)

            if item.ref_count == 0:
                self.n_free_block += len(item.blocks)

    def _evict(self, needed: int) -> None:
        while len(self.free_blocks) < needed:
            if not self.lru_cache.can_evict():
                raise MemoryError("No blocks can be freed (memory exhausted)")

            uuid, victim = self.lru_cache.popitem()

            self.all_items.pop(uuid)
            self.free_blocks.extend(victim.blocks)
