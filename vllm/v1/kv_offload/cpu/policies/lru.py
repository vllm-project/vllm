# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.cpu.policies.abstract import BlockStatus, CachePolicy


class LRUCachePolicy(CachePolicy):
    """LRU cache policy backed by a single OrderedDict."""

    def __init__(self, cache_capacity: int):
        # cache_capacity unused by LRU but accepted for a uniform constructor
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()

    def get(self, block_hash: BlockHash) -> BlockStatus | None:
        return self.blocks.get(block_hash)

    def insert(self, block_hash: BlockHash, block: BlockStatus) -> None:
        self.blocks[block_hash] = block

    def remove(self, block_hash: BlockHash) -> None:
        del self.blocks[block_hash]

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        for block_hash in reversed(list(block_hashes)):
            if block_hash in self.blocks:
                self.blocks.move_to_end(block_hash)

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        if n == 0:
            return []
        candidates: list[tuple[BlockHash, BlockStatus]] = []
        # First pass: prefer evicting not-ready blocks (ref_cnt == -1).
        # These are speculative prefetch blocks with no valid data —
        # they're expendable and should be evicted before confirmed data.
        for block_hash, block in self.blocks.items():
            if block.ref_cnt == -1 and block_hash not in protected:
                candidates.append((block_hash, block))
                if len(candidates) == n:
                    break
        # Second pass: evict ready blocks (ref_cnt == 0) if still needed.
        if len(candidates) < n:
            for block_hash, block in self.blocks.items():
                if block.ref_cnt == 0 and block_hash not in protected:
                    if (block_hash, block) not in candidates:
                        candidates.append((block_hash, block))
                        if len(candidates) == n:
                            break
        if len(candidates) < n:
            return None
        for block_hash, _ in candidates:
            del self.blocks[block_hash]
        return candidates
