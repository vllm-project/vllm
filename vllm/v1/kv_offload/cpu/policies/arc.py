# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import (
    CachePolicy,
    ChunkStatus,
    order_request_keys,
)


class ARCCachePolicy(CachePolicy):
    """
    ARC (Adaptive Replacement Cache) cache policy.

    Data Structures:
        T1: Recent cache containing chunks accessed once.
        T2: Frequent cache containing chunks accessed multiple times.
        B1/B2: Ghost lists tracking recently evicted chunks from T1/T2.
        target_t1_size: Adaptive target size for the T1 partition.

    Algorithm Flow:
        1. Cache lookup (lookup):
           Searches T1 and T2 for chunk hashes and counts consecutive hits
           until a miss or non-ready chunk is encountered.

        2. Request access - Adaptive Learning:
           - Ready chunks reused by a request move from T1 to T2 once, or
             move to the MRU end of T2.
           - B1/B2 misses adjust target_t1_size once per request before
             insertion removes their ghost entries.

        3. Chunk eviction (evict) - Adaptive Replacement:
           Determines eviction source based on adaptive target:
           - If T1 size >= target_t1_size: Evict from T1, add to B1.
           - Otherwise: Evict from T2, add to B2.
           Finally, bound each ghost list size.

        4. Chunk insertion (insert):
           New chunks are always inserted into T1 and removed from B1/B2 if
           present. A later request reuse may promote them to T2.

    Adaptive Behavior:
        The algorithm self-tunes the recency vs. frequency trade-off:
        - B1 hit: Recent access patterns matter more → increase T1.
        - B2 hit: Frequent access patterns matter more → decrease T1.
    """

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self.target_t1_size: float = 0.0
        self.t1: OrderedDict[OffloadKey, ChunkStatus] = OrderedDict()
        self.t2: OrderedDict[OffloadKey, ChunkStatus] = OrderedDict()
        # key -> None (only care about presence)
        self.b1: OrderedDict[OffloadKey, None] = OrderedDict()
        self.b2: OrderedDict[OffloadKey, None] = OrderedDict()

    @override
    def get(self, key: OffloadKey) -> ChunkStatus | None:
        return self.t1.get(key) or self.t2.get(key)

    @override
    def insert(self, key: OffloadKey, chunk: ChunkStatus) -> None:
        self.t1[key] = chunk
        self.b1.pop(key, None)
        self.b2.pop(key, None)

    @override
    def remove(self, key: OffloadKey) -> None:
        if self.t1.pop(key, None) is None:
            self.t2.pop(key, None)

    def _adapt_to_ghost_hit(self, key: OffloadKey) -> bool:
        if key in self.b1:
            delta = max(1, len(self.b2) / len(self.b1))
            self.target_t1_size = min(self.target_t1_size + delta, self.cache_capacity)
            self.b1.move_to_end(key)
            return True
        if key in self.b2:
            delta = max(1, len(self.b1) / len(self.b2))
            self.target_t1_size = max(self.target_t1_size - delta, 0)
            self.b2.move_to_end(key)
            return True
        return False

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        for key in reversed(list(keys)):
            if key in self.t1:
                chunk = self.t1.pop(key)
                if not chunk.is_ready:
                    # chunk was just prepared to be stored, not really touched
                    # twice — keep it in T1 and mark as most recently used
                    self.t1[key] = chunk
                else:
                    self.t2[key] = chunk

            elif key in self.t2:
                self.t2.move_to_end(key)

            else:
                self._adapt_to_ghost_hit(key)

    @override
    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[OffloadKey]],
        insertion_only_keys: set[OffloadKey],
        reused_keys: set[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        for key in reversed(order_request_keys(key_groups, req_context)):
            if key in insertion_only_keys:
                # A store is not a frequency hit. Preserve T1 membership
                # while still restoring tail-to-head recency.
                if key in self.t1:
                    self.t1.move_to_end(key)
                elif key in self.t2:
                    self.t2.move_to_end(key)
                continue

            # Ready chunks reused by this request count as one access.
            if key in reused_keys and key in self.t1:
                chunk = self.t1.pop(key)
                self.t2[key] = chunk
            elif key in reused_keys and key in self.t2:
                self.t2.move_to_end(key)

    @override
    def clear(self) -> None:
        self.t1.clear()
        self.t2.clear()
        self.b1.clear()
        self.b2.clear()
        self.target_t1_size = 0.0

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []

        # Collect candidates atomically: simulate T1 size changes as we select,
        # but do not modify actual data structures until all n are found.
        candidates: list[
            tuple[OffloadKey, ChunkStatus, bool]
        ] = []  # (key, chunk, from_t1)
        virtual_t1_size = len(self.t1)
        # Keep the scans monotonic: restarting from the LRU end after every
        # selection makes a batch eviction quadratic in the number of chunks.
        t1_iter = iter(self.t1.items())
        t2_iter = iter(self.t2.items())

        def next_candidate(
            entries: Iterator[tuple[OffloadKey, ChunkStatus]],
        ) -> tuple[OffloadKey, ChunkStatus] | None:
            for key, chunk in entries:
                if chunk.ref_cnt == 0 and key not in protected:
                    return key, chunk
            return None

        for _ in range(n):
            candidate: tuple[OffloadKey, ChunkStatus, bool] | None = None

            if virtual_t1_size >= int(self.target_t1_size):
                entry = next_candidate(t1_iter)
                if entry is not None:
                    candidate = (*entry, True)
                    virtual_t1_size -= 1

            if candidate is None:
                entry = next_candidate(t2_iter)
                if entry is not None:
                    candidate = (*entry, False)

            if candidate is None:
                entry = next_candidate(t1_iter)
                if entry is None:
                    return None
                candidate = (*entry, True)
                virtual_t1_size -= 1

            candidates.append(candidate)

        # Apply all evictions now that we know n candidates exist.
        result: list[tuple[OffloadKey, ChunkStatus]] = []
        for key, chunk, from_t1 in candidates:
            if from_t1:
                del self.t1[key]
                self.b1[key] = None
            else:
                del self.t2[key]
                self.b2[key] = None
            result.append((key, chunk))

        # Trim ghost lists to cache_capacity.
        for ghost in (self.b1, self.b2):
            for _ in range(len(ghost) - self.cache_capacity):
                ghost.popitem(last=False)

        return result
