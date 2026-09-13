# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import heapq
from collections.abc import Iterable, Sequence

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import (
    CachePolicy,
    ChunkStatus,
    order_request_keys,
)


class LRUCachePolicy(CachePolicy):
    """
    LRU caching policy with logical recency independent of transfer pinning.

    Evictable chunks live in a lazy-invalidating min-heap. A chunk's recency
    can therefore be updated while it is pinned; when it later becomes
    evictable, it enters the heap with the order assigned by the request
    rather than its transfer-completion time.

    A use is indicated by,
     - First time the key is added (store).
     - A request-scoped access.
    """

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self.chunks: dict[OffloadKey, ChunkStatus] = {}
        self._ranks: dict[OffloadKey, int] = {}
        self._evictable: set[OffloadKey] = set()
        self._heap: list[tuple[int, OffloadKey]] = []
        self._next_rank = 0

    def _assign_new_rank(self, key: OffloadKey) -> bool:
        if key not in self.chunks:
            return False
        self._next_rank += 1
        self._ranks[key] = self._next_rank
        return key in self._evictable

    def _push_evictable(self, key: OffloadKey) -> None:
        heapq.heappush(
            self._heap,
            (self._ranks[key], key),
        )

    def _is_current(self, entry: tuple[int, OffloadKey]) -> bool:
        rank, key = entry
        return key in self._evictable and self._ranks.get(key) == rank

    def _maybe_compact_heap(self) -> None:
        if len(self._heap) < max(64, 2 * len(self._evictable)):
            return
        self._heap = [(self._ranks[key], key) for key in self._evictable]
        heapq.heapify(self._heap)

    def _update_recency(self, keys: Iterable[OffloadKey]) -> None:
        updated_evictable = [key for key in keys if self._assign_new_rank(key)]
        # Rebuilding is linear and substantially cheaper than k heap pushes
        # for a long prefix. Small updates retain the incremental path.
        if len(updated_evictable) >= max(64, len(self._evictable) // 4):
            self._heap = [(self._ranks[key], key) for key in self._evictable]
            heapq.heapify(self._heap)
        else:
            for key in updated_evictable:
                self._push_evictable(key)
            self._maybe_compact_heap()

    @override
    def get(self, key: OffloadKey) -> ChunkStatus | None:
        return self.chunks.get(key)

    @override
    def insert(self, key: OffloadKey, chunk: ChunkStatus) -> None:
        self.chunks[key] = chunk
        self._next_rank += 1
        self._ranks[key] = self._next_rank
        if chunk.ref_cnt == 0:
            self._evictable.add(key)
            self._push_evictable(key)

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.chunks[key]
        self._ranks.pop(key, None)
        self._evictable.discard(key)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        self._update_recency(reversed(list(keys)))

    @override
    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[OffloadKey]],
        insertion_only_keys: set[OffloadKey],
        reused_keys: set[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        del insertion_only_keys, reused_keys
        self._update_recency(reversed(order_request_keys(key_groups, req_context)))

    @override
    def clear(self) -> None:
        self.chunks.clear()
        self._ranks.clear()
        self._evictable.clear()
        self._heap.clear()
        self._next_rank = 0

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []

        selected: list[tuple[tuple[int, OffloadKey], ChunkStatus]] = []
        selected_keys: set[OffloadKey] = set()
        deferred: list[tuple[int, OffloadKey]] = []
        while self._heap and len(selected) < n:
            entry = heapq.heappop(self._heap)
            if not self._is_current(entry):
                continue
            key = entry[1]
            # Re-pinning without a recency change can leave an equivalent
            # lazy entry behind. Select each cache key at most once.
            if key in selected_keys:
                continue
            if key in protected:
                deferred.append(entry)
                continue
            chunk = self.chunks[key]
            assert chunk.ref_cnt == 0
            selected.append((entry, chunk))
            selected_keys.add(key)

        if len(selected) < n:
            for entry, _ in selected:
                heapq.heappush(self._heap, entry)
            for entry in deferred:
                heapq.heappush(self._heap, entry)
            return None

        for entry in deferred:
            heapq.heappush(self._heap, entry)

        candidates: list[tuple[OffloadKey, ChunkStatus]] = []
        for entry, chunk in selected:
            key = entry[1]
            self._evictable.remove(key)
            del self.chunks[key]
            del self._ranks[key]
            candidates.append((key, chunk))
        self._maybe_compact_heap()
        return candidates

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        # chunks can become evictable when,
        # store completes - i.e. ref_cnt -1 -> 0 # not in evictable list
        # all loads complete - i.e ref_cnt 1 -> 0  # not in evictable list
        self._evictable.add(key)
        self._push_evictable(key)
        self._maybe_compact_heap()

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        # key must have been evictable before it was pinned.
        self._evictable.remove(key)
