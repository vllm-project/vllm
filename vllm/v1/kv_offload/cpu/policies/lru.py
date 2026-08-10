# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, ChunkSlotStatus


class LRUCachePolicy(CachePolicy):
    """
    LRU Caching policy that keeps a dedicated evictable list for fast eviction.
    A use is indicated by,
     - First time the key is added (store).
     - Load job completion
     - touch
    """

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        # Chunks with ref_cnt 0 (not participating in any loads/stores) ordered in LRU
        self.evictable_chunks: OrderedDict[OffloadKey, None] = OrderedDict()
        self.chunks: dict[OffloadKey, ChunkSlotStatus] = {}

    @override
    def get(self, key: OffloadKey) -> ChunkSlotStatus | None:
        return self.chunks.get(key)

    @override
    def insert(self, key: OffloadKey, chunk: ChunkSlotStatus) -> None:
        self.chunks[key] = chunk
        if chunk.ref_cnt == 0:
            self.evictable_chunks[key] = None

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.chunks[key]
        self.evictable_chunks.pop(key, None)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        for key in reversed(list(keys)):
            if key in self.evictable_chunks:
                self.evictable_chunks.move_to_end(key)
            # active chunks are untouched as they are non-evictable now. They
            # will eventually reach the end of evictable_chunks when they finish.

    @override
    def clear(self) -> None:
        self.evictable_chunks.clear()
        self.chunks.clear()

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkSlotStatus]] | None:
        if n == 0:
            return []

        candidates: list[tuple[OffloadKey, ChunkSlotStatus]] = []
        for key, _ in self.evictable_chunks.items():
            if key in protected:
                continue

            chunk = self.chunks[key]
            assert chunk.ref_cnt == 0
            candidates.append((key, chunk))
            if len(candidates) == n:
                break

        if len(candidates) < n:
            return None
        for key, _ in candidates:
            del self.evictable_chunks[key]
            del self.chunks[key]
        return candidates

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        # chunks can become evictable when,
        # store completes - i.e. ref_cnt -1 -> 0 # not in evictable list
        # all loads complete - i.e ref_cnt 1 -> 0  # not in evictable list
        self.evictable_chunks[key] = None

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        # key must have been in the evictable list.
        del self.evictable_chunks[key]
