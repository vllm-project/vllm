# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Segmented LRU (SLRU) cache policy with ghost set.

Splits the cache into two segments:
  - **Probation**: new blocks land here. Evicted first.
  - **Protected**: blocks promoted here after a second access. Evicted last.

A bounded *ghost set* remembers block hashes that were previously in the
protected segment. When a block with a ghost-hit hash is re-inserted,
it enters protected directly, skipping probation. This accelerates
cache recovery after transient pressure spikes (e.g., scale-in events
that temporarily flush the cache with one-time prefixes).

Compared to ARC (also two-segment + ghost):
  - ARC ghost hits adjust the T1/T2 *ratio* (advisory).
  - SLRU ghost hits directly *place* the block in protected (prescriptive).
  This means SLRU recovers faster after cache pressure because returning
  hot blocks skip the probation phase entirely.
"""

from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.cpu.policies.abstract import BlockStatus, CachePolicy

_PROTECTED_RATIO = 0.8


class SLRUCachePolicy(CachePolicy):
    """Segmented LRU with ghost set."""

    def __init__(self, cache_capacity: int):
        self.cache_capacity = cache_capacity
        self.max_protected = int(cache_capacity * _PROTECTED_RATIO)

        self.probation: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.protected: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        # Ghost set bounded to cache_capacity (same sizing as ARC ghost lists).
        self.ghost: OrderedDict[BlockHash, None] = OrderedDict()

    def get(self, block_hash: BlockHash) -> BlockStatus | None:
        return (self.probation.get(block_hash)
                or self.protected.get(block_hash))

    def insert(self, block_hash: BlockHash, block: BlockStatus) -> None:
        if block_hash in self.ghost:
            del self.ghost[block_hash]
            self.protected[block_hash] = block
            self._maybe_demote()
        else:
            self.probation[block_hash] = block

    def remove(self, block_hash: BlockHash) -> None:
        if block_hash in self.probation:
            del self.probation[block_hash]
        elif block_hash in self.protected:
            del self.protected[block_hash]

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        for block_hash in reversed(list(block_hashes)):
            if block_hash in self.probation:
                block = self.probation[block_hash]
                if not block.is_ready:
                    self.probation.move_to_end(block_hash)
                else:
                    del self.probation[block_hash]
                    self.protected[block_hash] = block
                    self._maybe_demote()

            elif block_hash in self.protected:
                self.protected.move_to_end(block_hash)

    def evict(
        self, n: int, protected: set[BlockHash],
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        if n == 0:
            return []

        candidates: list[tuple[BlockHash, BlockStatus, bool]] = []

        for block_hash, block in self.probation.items():
            if block.ref_cnt == 0 and block_hash not in protected:
                candidates.append((block_hash, block, True))
                if len(candidates) == n:
                    break

        if len(candidates) < n:
            for block_hash, block in self.protected.items():
                if block.ref_cnt == 0 and block_hash not in protected:
                    candidates.append((block_hash, block, False))
                    if len(candidates) == n:
                        break

        if len(candidates) < n:
            return None

        result: list[tuple[BlockHash, BlockStatus]] = []
        for block_hash, block, from_probation in candidates:
            if from_probation:
                del self.probation[block_hash]
            else:
                del self.protected[block_hash]
                self.ghost[block_hash] = None
            result.append((block_hash, block))

        # Trim ghost set once after all insertions.
        while len(self.ghost) > self.cache_capacity:
            self.ghost.popitem(last=False)

        return result

    def _maybe_demote(self) -> None:
        """If protected exceeds capacity, demote LRU entries to probation."""
        while len(self.protected) > self.max_protected:
            block_hash, block = self.protected.popitem(last=False)
            self.probation[block_hash] = block
