# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ExampleSecondaryTier: A simple in-memory secondary tier.

This implementation provides a minimal secondary tier that stores blocks
in memory (using a dictionary) with immediate completion. It serves as a
reference for writing new tiers and is useful for testing the
TieringOffloadingManager without requiring actual storage or network backends.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


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
        capacity: int | None = None,
    ):
        """
        Initialize the example secondary tier.

        Args:
            vllm_config: Global vLLM configuration.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            capacity: Maximum number of blocks to store. None means unlimited.
        """
        super().__init__(vllm_config, primary_kv_view)

        self.capacity = capacity

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
        keys = job_metadata.keys
        block_ids = job_metadata.block_ids

        assert len(keys) == len(block_ids), (
            f"Length mismatch: {len(keys)} keys but {len(block_ids)} block_ids"
        )

        if self.capacity is not None and len(self.blocks) + len(keys) > self.capacity:
            self.completed_jobs.append(
                JobResult(job_id=job_metadata.job_id, success=False)
            )
            return

        for key in keys:
            self.blocks[key] = True
        self.completed_jobs.append(JobResult(job_id=job_metadata.job_id, success=True))

    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit a job to load blocks from this tier to primary tier.

        Args:
            job_metadata: Job metadata including job_id, keys, and
                          spec for writing blocks into the primary tier.
        """
        keys = job_metadata.keys
        block_ids = job_metadata.block_ids

        assert len(keys) == len(block_ids), (
            f"Length mismatch: {len(keys)} keys but {len(block_ids)} block_ids"
        )

        for key in keys:
            if key not in self.blocks:
                return

        self.completed_jobs.append(JobResult(job_id=job_metadata.job_id, success=True))

    def get_finished(self) -> Iterable[JobResult]:
        """
        Poll for finished jobs.

        Returns:
            Iterable of JobResult objects for all jobs that have
            finished since the last call.
        """
        result = self.completed_jobs
        self.completed_jobs = []
        return result

    @staticmethod
    def get_tier_type() -> str:
        return "example"

    def get_num_blocks(self) -> int:
        """Get the number of blocks currently stored in this tier."""
        return len(self.blocks)
