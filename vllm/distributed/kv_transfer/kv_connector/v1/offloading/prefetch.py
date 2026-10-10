# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-free CPU -> GPU prefetch of offloaded KV.

A paused agentic session's KV is stored on finish and loaded again only when a
later request's prefix lookup hits the offloaded tier, which puts the load on
that request's critical path. This module holds the state for bringing those
blocks back *before* a request exists.

It is an execution mechanism, not a policy: it does not decide whether, when or
for whom to prefetch, predicts nothing and ranks nothing. It checks whether a
directive is feasible right now (the target resolves to offloaded content and
fits in the free GPU blocks behind a configured reserve) and executes it; an
infeasible prefetch is deferred. It never preempts a request and never takes a
block that a request references. Its destination blocks come from the free pool
exactly as a request's allocation would, so, like any allocation, they may
reclaim unreferenced prefix-cache blocks in LRU order. Whether a prefetch should
be allowed to displace idle cached content at all is a policy question left to
the caller; enforcing "no displacement" would need allocator support for
drawing only uncached free blocks, which this primitive does not add.

Submission and completion are separate. `request_free_prefetch` returns a
`PrefetchSubmitResult` synchronously; for an accepted prefetch, a
`PrefetchCompletion` follows once the transfer has finished and the blocks are
registered as reusable GPU prefix-cache residency (or once it is cancelled).
"""

from dataclasses import dataclass
from enum import Enum, auto

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.kv_offload.base import OffloadKey


class PrefetchSubmitOutcome(Enum):
    """Synchronous result of submitting a request-free prefetch.

    None of these means the KV is usable except ALREADY_AVAILABLE, which is
    only returned after checking every requested block in the GPU prefix
    cache. Usable residency for an accepted prefetch is reported later by a
    `PrefetchCompletion`.
    """

    # Destination blocks were reserved and the load was submitted. The blocks
    # are not readable until the matching PrefetchCompletion is COMPLETED.
    ACCEPTED = auto()
    # Not feasible now: too few free GPU blocks behind the reserve, or the
    # offloaded tier cannot serve the target yet (a store still in flight, or
    # the backend asked to retry). Nothing was reserved, submitted or
    # reclaimed; the caller may submit again later.
    DEFERRED = auto()
    # The target does not resolve to loadable offloaded content: the first
    # block that is not already on the GPU is not in the offloaded tier, or the
    # target is shorter than one offloaded chunk.
    MISSING = auto()
    # Verified: every requested block is already in the GPU prefix cache
    # (vacuously true for an empty target).
    ALREADY_AVAILABLE = auto()
    # The deployment cannot execute request-free prefetches (feature disabled,
    # prefix caching off, no GPU block pool bound, more than one KV cache
    # group, or a block size that differs from the hash block size).
    UNSUPPORTED = auto()


@dataclass(frozen=True)
class PrefetchSubmitResult:
    outcome: PrefetchSubmitOutcome
    # Set only for ACCEPTED; the matching PrefetchCompletion carries it.
    prefetch_id: str | None = None
    # Number of GPU blocks in the target (one per block hash).
    num_blocks_requested: int = 0
    # Target blocks verified to be in the GPU prefix cache at submission.
    num_blocks_available: int = 0
    # Destination blocks reserved for the submitted load (ACCEPTED only).
    num_blocks_to_load: int = 0

    @property
    def num_blocks_unresolved(self) -> int:
        """Target blocks neither on the GPU nor covered by the load."""
        return (
            self.num_blocks_requested
            - self.num_blocks_available
            - self.num_blocks_to_load
        )


class PrefetchCompletionStatus(Enum):
    # The load finished and every destination block is registered in the GPU
    # prefix cache (or its hash already was), i.e. usable residency.
    COMPLETED = auto()
    # The load was abandoned before publication (the offloading cache was
    # reset while it was in flight); the destination blocks were released.
    CANCELLED = auto()


@dataclass(frozen=True)
class PrefetchCompletion:
    prefetch_id: str
    status: PrefetchCompletionStatus
    # Blocks newly registered in the prefix cache by this prefetch.
    num_blocks_published: int = 0
    # Loaded blocks whose hash was already cached when the load finished (for
    # example recomputed by a request meanwhile); released, not duplicated.
    num_blocks_already_cached: int = 0


@dataclass
class PrefetchJob:
    """A submitted request-free prefetch, tracked until the load finishes."""

    prefetch_id: str
    # Offload keys being loaded, for OffloadingManager.complete_load().
    keys: set[OffloadKey]
    # Destination blocks, held with a reference until the load completes.
    blocks: list[KVCacheBlock]
    # Per-block prefix-cache hash, parallel to `blocks`.
    block_hashes: list[BlockHash]
    group_idx: int
    tokens_per_block: int


class GPUPrefetchReservation:
    """Destination-block bookkeeping for request-free prefetches.

    Reserved blocks are ordinary allocated blocks (`ref_cnt == 1`, no hash), so
    nothing can read or reallocate them while the transfer is in flight. On
    completion they are published into the prefix cache and released as
    last-resort eviction candidates, which is what makes a prefetch cheap to be
    wrong about: an unused prefetched block is evicted before any block a
    request still wants.
    """

    def __init__(self, block_pool: BlockPool, reserve_blocks: int):
        self.block_pool = block_pool
        # A resource-safety constraint for execution-time feasibility: free
        # blocks a prefetch may not take, so it cannot consume the last blocks
        # a running request needs to grow. Not a policy on what to prefetch.
        self.reserve_blocks = reserve_blocks

    def can_reserve(self, num_blocks: int) -> bool:
        free = self.block_pool.get_num_free_blocks()
        return free - self.reserve_blocks >= num_blocks

    def reserve(self, num_blocks: int) -> list[KVCacheBlock] | None:
        if not self.can_reserve(num_blocks):
            return None
        return self.block_pool.get_new_blocks(num_blocks)

    def is_cached(self, block_hash: BlockHash, group_idx: int) -> bool:
        return (
            self.block_pool.get_cached_block(block_hash, kv_cache_group_ids=[group_idx])
            is not None
        )

    def publish(self, job: PrefetchJob) -> tuple[int, int]:
        """Register the loaded blocks in the prefix cache and release them.

        A block whose hash is already cached (a request computed the same
        content while the load was in flight) is released instead of being
        registered a second time. Returns (published, already_cached).
        """
        fresh_blocks: list[KVCacheBlock] = []
        fresh_hashes: list[BlockHash] = []
        duplicates: list[KVCacheBlock] = []
        for block, block_hash in zip(job.blocks, job.block_hashes):
            if self.is_cached(block_hash, job.group_idx):
                duplicates.append(block)
            else:
                fresh_blocks.append(block)
                fresh_hashes.append(block_hash)
        if duplicates:
            self.block_pool.free_blocks(duplicates)
        if fresh_blocks:
            self.block_pool.cache_prefetched_blocks(
                fresh_blocks,
                fresh_hashes,
                job.group_idx,
                job.tokens_per_block,
                on_reuse=lambda block: None,
            )
        return len(fresh_blocks), len(duplicates)

    def release(self, job: PrefetchJob) -> None:
        """Give the destination blocks back without publishing them."""
        self.block_pool.free_blocks(job.blocks)
