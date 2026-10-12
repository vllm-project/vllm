# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-owned eviction ordering for optional retention declarations."""

import heapq
from collections.abc import Callable

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


class RetentionFreeBlockQueue(FreeKVCacheBlockQueue):
    """Keep ordinary queue order as the tie-breaker for eviction priorities.

    The linked list remains authoritative for free membership. Lazy heap entries
    are bounded by compaction; retention never changes a block's reference count.
    """

    def __init__(
        self,
        free_blocks: list[KVCacheBlock],
        all_blocks: list[KVCacheBlock],
        priority: Callable[[KVCacheBlock], int],
        synchronize: Callable[[], None],
    ):
        super().__init__(free_blocks)
        self._all_blocks = all_blocks
        self._priority = priority
        self._synchronize = synchronize
        self._keys: dict[int, tuple[int, int]] = {
            block.block_id: (priority(block), order)
            for order, block in enumerate(free_blocks)
        }
        self._head_order = 0
        self._tail_order = len(free_blocks)
        self._heap = [
            (p, order, block_id) for block_id, (p, order) in self._keys.items()
        ]
        heapq.heapify(self._heap)
        self.last_displaced_cached_blocks = 0

    def priority(self, block: KVCacheBlock) -> int:
        self._synchronize()
        return self._priority(block)

    def popleft(self) -> KVCacheBlock:
        return self.popleft_n(1)[0]

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        if n < 0 or n > self.num_free_blocks:
            raise ValueError("Not enough free blocks")
        self.last_displaced_cached_blocks = 0
        if n == 0:
            return []
        self._synchronize()
        selected: list[KVCacheBlock] = []
        while len(selected) < n:
            priority, order, block_id = heapq.heappop(self._heap)
            if self._keys.get(block_id) != (priority, order):
                continue
            block = self._all_blocks[block_id]
            super().remove(block)
            del self._keys[block_id]
            self.last_displaced_cached_blocks += block.block_hash is not None
            selected.append(block)
        self._compact()
        return selected

    def remove(self, block: KVCacheBlock) -> None:
        super().remove(block)
        del self._keys[block.block_id]
        self._compact()

    def append(self, block: KVCacheBlock) -> None:
        super().append(block)
        self._add([block], prepend=False)

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        super().append_n(blocks)
        self._add(blocks, prepend=False)

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        super().prepend_n(blocks)
        self._add(blocks, prepend=True)

    def cache_changed(self, block: KVCacheBlock) -> None:
        previous = self._keys.get(block.block_id)
        if previous is None:
            return
        priority = self._priority(block)
        if previous[0] != priority:
            self._keys[block.block_id] = (priority, previous[1])
            heapq.heappush(self._heap, (priority, previous[1], block.block_id))
            self._compact()

    def rebuild(self) -> None:
        self._keys = {
            block_id: (self._priority(self._all_blocks[block_id]), order)
            for block_id, (_, order) in self._keys.items()
        }
        self._heap = [
            (p, order, block_id) for block_id, (p, order) in self._keys.items()
        ]
        heapq.heapify(self._heap)

    def _add(self, blocks: list[KVCacheBlock], *, prepend: bool) -> None:
        if prepend:
            self._head_order -= len(blocks)
            first = self._head_order
        else:
            first = self._tail_order
            self._tail_order += len(blocks)
        for offset, block in enumerate(blocks):
            order, priority = first + offset, self._priority(block)
            self._keys[block.block_id] = (priority, order)
            heapq.heappush(self._heap, (priority, order, block.block_id))
        self._compact()

    def _compact(self) -> None:
        if len(self._heap) > max(64, 2 * len(self._keys)):
            self._heap = [
                (p, order, block_id) for block_id, (p, order) in self._keys.items()
            ]
            heapq.heapify(self._heap)
