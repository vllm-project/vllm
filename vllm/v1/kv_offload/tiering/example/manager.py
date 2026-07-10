# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ExampleSecondaryTierManager: A simple in-memory secondary tier.

This implementation provides a minimal secondary tier that stores blocks
in memory (using a dictionary) with immediate completion. It serves as a
reference for writing new tiers and is useful for testing the
TieringOffloadingManager without requiring actual storage or network backends.
"""

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

from typing_extensions import override

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec


class ExampleSecondaryTierManager(SecondaryTierManager):
    """
    A simple in-memory secondary tier.

    This implementation:
    - Stores blocks in a dictionary (key -> True)
    - Completes transfers immediately (synchronous)
    """

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        custom_param: int = 0,
    ):
        """
        Initialize the example secondary tier.

        Args:
            custom_param: Dummy parameter demonstrating custom args.
        """
        super().__init__(
            offloading_spec=offloading_spec,
            primary_kv_view=primary_kv_view,
            tier_type=tier_type,
        )

        logger.info(
            "ExampleSecondaryTierManager initialized with custom_param=%d", custom_param
        )

        # key -> True (only care about presence)
        self.blocks: dict[OffloadKey, bool] = {}

        # Completed jobs waiting to be retrieved by get_finished_jobs()
        self.completed_jobs: list[JobResult] = []

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        """
        Check whether a block exists in this secondary tier.

        Args:
            key: Offload key to look up.
            req_context: Per-request context.

        Returns:
            HIT if the block is present, MISS if not found.
        """
        return LookupResult.HIT if key in self.blocks else LookupResult.MISS

    @override
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

        for key in keys:
            self.blocks[key] = True
        self.completed_jobs.append(JobResult(job_id=job_metadata.job_id, success=True))

    @override
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
                self.completed_jobs.append(
                    JobResult(job_id=job_metadata.job_id, success=False)
                )
                return

        self.completed_jobs.append(JobResult(job_id=job_metadata.job_id, success=True))

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Poll for finished jobs.

        Returns:
            Iterable of JobResult objects for all jobs that have
            finished since the last call.
        """
        result = self.completed_jobs
        self.completed_jobs = []
        return result

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def drain_jobs(self) -> None:
        """Synchronous tier — submit_*() returns only after the operation
        completes, so there is nothing to wait for."""
        return

    def get_num_blocks(self) -> int:
        """Get the number of blocks currently stored in this tier."""
        return len(self.blocks)
