# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ExampleSecondaryTier: A simple in-memory secondary tier.

This implementation provides a minimal secondary tier that stores blocks
in memory (using a dictionary) with immediate completion. It serves as a
reference for writing new tiers and is useful for testing the
TieringOffloadingManager without requiring actual storage or network backends.
"""

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
    A simple in-memory secondary tier.

    This implementation:
    - Stores blocks in a dictionary (key -> True)
    - Completes transfers immediately (synchronous)
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        primary_kv_view: memoryview,
    ):
        """
        Initialize the example secondary tier.

        Args:
            vllm_config: Global vLLM configuration.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
        """
        super().__init__(vllm_config, primary_kv_view)

        # key -> True (only care about presence)
        self.blocks: dict[OffloadKey, bool] = {}

        # Completed jobs waiting to be retrieved by get_finished()
        self.completed_jobs: list[JobResult] = []

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
        Submit a job to store blocks from primary tier to this tier.

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

        internal_job_metadata = _JobMetadata(job_id=job_id, keys=keys, is_store=True)

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

        self._complete_load_job(internal_job_metadata)

    def get_finished(self) -> Iterable[JobResult]:
        """
        Poll for finished async jobs.

        Returns:
            Iterable of JobResult objects for all jobs that have
            finished since the last call.
        """
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

    @staticmethod
    def get_tier_type() -> str:
        return "example"

    def get_num_blocks(self) -> int:
        """Get the number of blocks currently stored in this tier."""
        return len(self.blocks)

    def clear(self):
        """Clear all blocks and jobs (for testing)."""
        self.blocks.clear()
        self.completed_jobs.clear()
