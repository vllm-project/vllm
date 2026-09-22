# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-free CPU -> GPU prefetch of offloaded KV.

A paused agentic session's KV is stored on finish and loaded again only when a
later request's prefix lookup hits the offloaded tier, which puts the load on
that request's critical path. This module holds the state for bringing those
blocks back *before* a request exists: it is an execution primitive, not a
policy. It takes no lead time, predicts nothing, ranks nothing and never
evicts to make room - a prefetch either fits in the free pool behind a reserve
or it is deferred.
"""

from dataclasses import dataclass
from enum import Enum, auto

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.kv_offload.base import OffloadKey


class PrefetchOutcome(Enum):
    """Result of a request-free prefetch attempt."""

    # Destination blocks were reserved and a load was submitted. The blocks
    # enter the prefix cache when the transfer completes.
    ACCEPTED = auto()
    # Not enough free GPU blocks behind the reserve. Nothing was reserved and
    # nothing was submitted; the caller may try again later.
    DEFERRED = auto()
    # Nothing to do: none of the requested blocks are resident in an offloaded
    # tier, or they are already in the GPU prefix cache.
    COMPLETED = auto()
    # The deployment cannot serve request-free prefetches (feature disabled,
    # prefix caching off, or no GPU block pool bound).
    UNSUPPORTED = auto()


@dataclass
class PrefetchJob:
    """A submitted request-free prefetch, tracked until the load finishes."""

    # Offload keys being loaded, for OffloadingManager.complete_load().
    keys: set[OffloadKey]
    # Destination blocks, held with a reference until the load completes.
    blocks: list[KVCacheBlock]
    # Per-block prefix-cache hash, parallel to `blocks`.
    block_hashes: list[BlockHash]
    group_idx: int
    tokens_per_block: int
    # Set when a destination block is handed to a request while the load is in
    # flight: the data landing in it can no longer be published.
    invalidated: bool = False


class GPUPrefetchReservation:
    """Destination-block bookkeeping for request-free prefetches.

    Reserved blocks are ordinary allocated blocks (`ref_cnt == 1`, no hash), so
    nothing can read them while the transfer is in flight. On completion they
    are published into the prefix cache and released as last-resort eviction
    candidates, which is what makes a prefetch cheap to be wrong about: an
    unused prefetched block is evicted before any block a request still wants.
    """

    def __init__(self, block_pool: BlockPool, reserve_blocks: int):
        self.block_pool = block_pool
        # Free blocks kept out of reach of prefetches, so a prefetch can never
        # consume the last blocks a running request needs to grow.
        self.reserve_blocks = reserve_blocks

    def can_reserve(self, num_blocks: int) -> bool:
        free = self.block_pool.get_num_free_blocks()
        return free - self.reserve_blocks >= num_blocks

    def reserve(self, num_blocks: int) -> list[KVCacheBlock] | None:
        if not self.can_reserve(num_blocks):
            return None
        return self.block_pool.get_new_blocks(num_blocks)

    def publish(self, job: PrefetchJob) -> int:
        """Insert the loaded blocks into the prefix cache and release them.

        Returns the number of blocks published.
        """
        if job.invalidated:
            self.release(job)
            return 0
        self.block_pool.cache_prefetched_blocks(
            job.blocks,
            job.block_hashes,
            job.group_idx,
            job.tokens_per_block,
            on_reuse=lambda block: None,
        )
        return len(job.blocks)

    def release(self, job: PrefetchJob) -> None:
        """Give the destination blocks back without publishing them."""
        self.block_pool.free_blocks(job.blocks)
