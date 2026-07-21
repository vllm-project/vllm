# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Callable, Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class LRUCachePolicy(CachePolicy):
    """
    LRU Caching policy that keeps a dedicated evictable list for fast eviction.
    A use is indicated by,
     - First time the key is added (store).
     - Load job completion
     - touch
    """

    def __init__(self, cache_capacity: int):
        # Blocks with ref_cnt 0 (not participating in any loads/stores) ordered in LRU
        self.evictable_blocks: OrderedDict[OffloadKey, None] = OrderedDict()
        self.blocks: dict[OffloadKey, BlockStatus] = {}

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        if block.ref_cnt == 0:
            self.evictable_blocks[key] = None

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        self.evictable_blocks.pop(key, None)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        for key in reversed(list(keys)):
            if key in self.evictable_blocks:
                self.evictable_blocks.move_to_end(key)
            # active blocks are untouched as they are non-evictable now. They
            # will eventually reach the end of evictable_blocks when they finish.

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []
        return self.evict_until(lambda c: len(c) >= n, protected)

    @override
    def clear(self) -> None:
        self.evictable_blocks.clear()
        self.blocks.clear()

    @override
    def evict_until(
        self,
        can_fit: Callable[[list[tuple[OffloadKey, BlockStatus]]], bool],
        protected: set[OffloadKey],
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        """
        Walk evictable blocks in LRU order, calling ``can_fit`` after each
        candidate.  On predicate success the collected prefix is removed
        from both data structures and returned.  On exhaustion returns None
        with zero mutation.
        """
        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        for key, _ in self.evictable_blocks.items():
            if key in protected:
                continue

            block = self.blocks[key]
            assert block.ref_cnt == 0
            candidates.append((key, block))

            if can_fit(candidates):
                for k, _ in candidates:
                    del self.evictable_blocks[k]
                    del self.blocks[k]
                return candidates

        return None

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        # blocks can become evictable when,
        # store completes - i.e. ref_cnt -1 -> 0 # not in evictable list
        # all loads complete - i.e ref_cnt 1 -> 0  # not in evictable list
        self.evictable_blocks[key] = None

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        # key must have been in the evictable list.
        del self.evictable_blocks[key]
