# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ExampleSecondaryTier: A simple in-memory secondary tier for testing.

This implementation provides a minimal secondary tier that stores blocks
in memory (using a dictionary) and simulates async transfers with immediate
completion. It's useful for testing the TieringOffloadingManager without
requiring actual storage or network backends.
"""

from collections import OrderedDict
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class _JobMetadata:
    """Internal metadata for tracking job details."""

    job_id: JobId
    keys: Collection[OffloadKey]
    is_store: bool  # True for store jobs, False for load jobs


class ExampleSecondaryTier(SecondaryTierManager):
    """
    A simple in-memory secondary tier for testing.

    This implementation:
    - Stores blocks in a dictionary (key -> True)
    - Simulates async transfers with immediate completion
    - Uses LRU eviction policy
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        primary_kv_view: memoryview,
        max_blocks: int = 1000,
        simulate_async: bool = False,
    ):
        """
        Initialize the example secondary tier.

        Args:
            vllm_config: Global vLLM configuration.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            max_blocks: Maximum number of blocks this tier can store
            simulate_async: If True, jobs complete on next get_finished() call.
                          If False, jobs complete immediately.
        """
        super().__init__(vllm_config, primary_kv_view)
        self.max_blocks = max_blocks
        self.simulate_async = simulate_async

        # key -> True (only care about presence)
        self.blocks: OrderedDict[OffloadKey, bool] = OrderedDict()

        # Completed jobs waiting to be retrieved by get_finished()
        self.completed_jobs: list[JobResult] = []

        # Pending jobs (for simulated async mode)
        self.pending_jobs: list[_JobMetadata] = []

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Check whether a block exists in this secondary tier.

        Args:
            key: Offload key to look up.
            req_context: Per-request context.

        Returns:
            True if the block is present, False if not found.
        """
        return key in self.blocks

    def submit_store(self, job_metadata: JobMetadata) -> None:
        """
        Submit an async job to store blocks from primary tier to this tier.

        Args:
            job_metadata: Job metadata including job_id, keys, and
                          spec for reading blocks from the primary tier.
        """
        job_id = job_metadata.job_id
        keys = job_metadata.keys
        block_ids = job_metadata.block_ids

        assert len(keys) == len(block_ids), (
            f"Length mismatch: {len(keys)} keys but {len(block_ids)} block_ids"
        )

        # Filter out blocks already present
        blocks_to_store = [bh for bh in keys if bh not in self.blocks]

        if not blocks_to_store:
            # All blocks already present
            return

        # Evict blocks if needed (LRU policy)
        num_blocks_to_evict = len(blocks_to_store) - (
            self.max_blocks - len(self.blocks)
        )

        evicted = []
        if num_blocks_to_evict > 0:
            # Collect eviction candidates first (LRU order), then delete atomically
            protected = set(keys)
            for key in self.blocks:
                if key not in protected:
                    evicted.append(key)
                    if len(evicted) == num_blocks_to_evict:
                        break
            else:
                # Could not collect enough eviction candidates
                return
            for key in evicted:
                del self.blocks[key]

        # Create internal job metadata
        internal_job_metadata = _JobMetadata(
            job_id=job_id, keys=blocks_to_store, is_store=True
        )

        if self.simulate_async:
            # Job will complete on next get_finished() call
            self.pending_jobs.append(internal_job_metadata)
        else:
            # Job completes immediately
            self._complete_store_job(internal_job_metadata)

    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit an async job to load blocks from this tier to primary tier.

        Args:
            job_metadata: Job metadata including job_id, keys, and
                          spec for writing blocks into the primary tier.
        """
        job_id = job_metadata.job_id
        keys = job_metadata.keys
        block_ids = job_metadata.block_ids

        assert len(keys) == len(block_ids), (
            f"Length mismatch: {len(keys)} keys but {len(block_ids)} block_ids"
        )

        # Verify all blocks exist
        for key in keys:
            if key not in self.blocks:
                return

        # Create internal job metadata
        internal_job_metadata = _JobMetadata(job_id=job_id, keys=keys, is_store=False)

        if self.simulate_async:
            # Job will complete on next get_finished() call
            self.pending_jobs.append(internal_job_metadata)
        else:
            # Job completes immediately
            self._complete_load_job(internal_job_metadata)

    def get_finished(self) -> Iterable[JobResult]:
        """
        Poll for finished async jobs.

        Returns:
            Iterable of JobResult objects for all jobs that have
            finished since the last call.
        """
        # Move pending jobs to completed
        if self.simulate_async and self.pending_jobs:
            for job_metadata in self.pending_jobs:
                if job_metadata.is_store:
                    self._complete_store_job(job_metadata)
                else:
                    self._complete_load_job(job_metadata)
            self.pending_jobs.clear()

        # Return completed jobs
        result = self.completed_jobs
        self.completed_jobs = []
        return result

    def _complete_store_job(self, job_metadata: _JobMetadata):
        """Complete a store job by adding blocks to storage."""
        for key in job_metadata.keys:
            self.blocks[key] = True
        # Return simplified JobResult (only job_id and success)
        self.completed_jobs.append(JobResult(job_id=job_metadata.job_id, success=True))

    def _complete_load_job(self, job_metadata: _JobMetadata):
        """Complete a load job."""
        # Return simplified JobResult (only job_id and success)
        self.completed_jobs.append(JobResult(job_id=job_metadata.job_id, success=True))

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as recently used (move to end of LRU list).

        Args:
            keys: Blocks to mark as recently used.
            req_context: Per-request context.
        """
        for key in reversed(list(keys)):
            if key in self.blocks:
                self.blocks.move_to_end(key)

    @staticmethod
    def get_tier_type() -> str:
        return "example"

    def get_num_blocks(self) -> int:
        """Get the number of blocks currently stored in this tier."""
        return len(self.blocks)

    def clear(self):
        """Clear all blocks and pending jobs (for testing)."""
        self.blocks.clear()
        self.completed_jobs.clear()
        self.pending_jobs.clear()
