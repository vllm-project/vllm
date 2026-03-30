# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Priority-based eviction queue for KV-cache blocks.

This queue holds blocks that have been annotated with an explicit priority
by the orchestrator via retention directives. Blocks without priority
annotations remain in the standard FreeKVCacheBlockQueue (LRU).

The queue is a min-heap sorted by (effective_priority, last_freed_time),
so the lowest-priority, oldest block is evicted first. TTL expiry is
checked lazily on pop — no background timers.
"""

import heapq
import time

from vllm.v1.core.kv_cache_utils import KVCacheBlock


class PriorityEvictionQueue:
    """Min-heap of prioritized free blocks.

    Eviction order:
        1. Lowest effective_priority first.
        2. Ties broken by last_freed_time (oldest first, i.e. LRU).

    Blocks whose priority_expiry has passed are treated as priority=0
    (evict-first) when popped. This lazy check avoids background timers.
    """

    def __init__(self) -> None:
        # Heap entries: (effective_priority, last_freed_time, block_id, block)
        # block_id is included for stable ordering when priority and time
        # are equal.
        self._heap: list[tuple[int, float, int, KVCacheBlock]] = []
        # Track which block_ids are logically in the queue. Supports O(1)
        # membership check and lazy deletion.
        self._block_ids_in_queue: set[int] = set()
        self._num_blocks: int = 0

    def insert(self, block: KVCacheBlock) -> None:
        """Insert a prioritized block into the eviction queue.

        Args:
            block: A free block with priority != None.
        """
        assert block.priority is not None
        priority = self._effective_priority(block)
        entry = (priority, block.last_freed_time, block.block_id, block)
        heapq.heappush(self._heap, entry)
        self._block_ids_in_queue.add(block.block_id)
        self._num_blocks += 1

    def pop_lowest(self) -> KVCacheBlock | None:
        """Pop the block with the lowest effective priority.

        Lazily skips blocks that have been removed via remove().
        Lazily re-evaluates priority for blocks with expired TTL.

        Returns:
            The lowest-priority block, or None if empty.
        """
        while self._heap:
            priority, freed_time, block_id, block = heapq.heappop(self._heap)

            # Skip lazily deleted entries.
            if block_id not in self._block_ids_in_queue:
                continue

            # Check TTL expiry — if expired, treat as priority 0.
            current_priority = self._effective_priority(block)
            if current_priority != priority:
                # Priority changed (TTL expired since insertion).
                # Re-insert with updated priority and try again.
                entry = (current_priority, freed_time, block_id, block)
                heapq.heappush(self._heap, entry)
                continue

            self._block_ids_in_queue.discard(block_id)
            self._num_blocks -= 1
            return block

        return None

    def remove(self, block: KVCacheBlock) -> None:
        """Lazily remove a block from the queue.

        The block is marked as removed and will be skipped on future pops.
        This is O(1) — no heap restructuring.

        Args:
            block: The block to remove.
        """
        if block.block_id in self._block_ids_in_queue:
            self._block_ids_in_queue.discard(block.block_id)
            self._num_blocks -= 1

    def _effective_priority(self, block: KVCacheBlock) -> int:
        """Compute effective priority, accounting for TTL expiry.

        If priority_expiry is set and has passed, the block's priority
        is treated as 0 (evict first).
        """
        if block.priority is None:
            return 0
        if (
            block.priority_expiry is not None
            and time.monotonic() > block.priority_expiry
        ):
            # TTL expired — clear the priority on the block itself.
            block.priority = None
            block.priority_expiry = None
            return 0
        return block.priority

    @property
    def num_blocks(self) -> int:
        """Number of blocks logically in the queue."""
        return self._num_blocks

    def __len__(self) -> int:
        return self._num_blocks
