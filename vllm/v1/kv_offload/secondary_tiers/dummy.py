# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DummySecondaryTier: A simple in-memory secondary tier for testing.

This implementation provides a minimal secondary tier that stores blocks
in memory (using a dictionary) and simulates async transfers with immediate
completion. It's useful for testing the TieringOffloadingManager without
requiring actual storage or network backends.
"""

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.v1.kv_offload.abstract import (
    JobId,
    JobMetadata,
    JobResult,
    LoadStoreSpec,
    OffloadKey,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec


@dataclass
class _JobMetadata:
    """Internal metadata for tracking job details."""

    job_id: JobId
    keys: list[OffloadKey]
    is_store: bool  # True for store jobs, False for load jobs


class DummyLoadStoreSpec(LoadStoreSpec):
    """
    Minimal LoadStoreSpec for DummySecondaryTier testing.

    This spec is never actually used for I/O since DummySecondaryTier
    stores blocks in memory. It exists to provide proper type semantics
    and serve as a template for real secondary tier implementations.
    """

    @staticmethod
    def medium() -> str:
        return "Dummy"


class DummySecondaryTier(SecondaryTierManager):
    """
    A simple in-memory secondary tier for testing.

    This implementation:
    - Stores blocks in a dictionary (key -> True)
    - Simulates async transfers with immediate completion
    - Uses LRU eviction policy
    - Tracks in-flight transfers to return None from lookup()
    """

    def __init__(
        self,
        tier_name: str = "DummyStorage",
        max_blocks: int = 1000,
        simulate_async: bool = False,
    ):
        """
        Initialize the dummy secondary tier.

        Args:
            tier_name: Name of this tier (for identification)
            max_blocks: Maximum number of blocks this tier can store
            simulate_async: If True, jobs complete on next get_finished() call.
                          If False, jobs complete immediately.
        """
        self.tier_name = tier_name
        self.max_blocks = max_blocks
        self.simulate_async = simulate_async

        self._primary_view: memoryview | None = None

        # key -> True (only care about presence)
        self.blocks: OrderedDict[OffloadKey, bool] = OrderedDict()

        # Tracks in-flight transfers: key -> job_id
        self.in_flight: dict[OffloadKey, JobId] = {}

        # Completed jobs waiting to be retrieved by get_finished()
        self.completed_jobs: list[JobResult] = []

        # Pending jobs (for simulated async mode)
        self.pending_jobs: list[_JobMetadata] = []

    def set_primary_view(self, view: memoryview) -> None:
        self._primary_view = view

    def lookup(self, keys: Iterable[OffloadKey]) -> int | None:
        """
        Check which blocks exist in this secondary tier.

        Args:
            keys: Block hashes to look up.

        Returns:
            Number of consecutive blocks (from start) that are present and ready,
            or None if blocks are being transferred (retry later).
        """
        hit_count = 0
        for key in keys:
            # Check if block is in-flight
            if key in self.in_flight:
                # Block is being transferred, return None (retry later)
                return None

            # Check if block exists in this tier
            if key not in self.blocks:
                break

            hit_count += 1

        return hit_count

    def submit_store(self, job_metadata: JobMetadata) -> None:
        """
        Submit an async job to store blocks from primary tier to this tier.

        Args:
            job_metadata: Job metadata including job_id, keys, and
                          spec for reading blocks from the primary tier.
        """
        job_id = job_metadata.job_id
        keys_list = list(job_metadata.keys)
        primary_read_spec = job_metadata.spec

        # Validate spec type and consistency
        assert isinstance(primary_read_spec, CPULoadStoreSpec), (
            f"Expected CPULoadStoreSpec, got {type(primary_read_spec)}"
        )
        assert len(keys_list) == len(primary_read_spec.block_ids), (
            f"Length mismatch: {len(keys_list)} keys but "
            f"{len(primary_read_spec.block_ids)} block_ids in spec"
        )

        # Filter out blocks already present
        blocks_to_store = [bh for bh in keys_list if bh not in self.blocks]

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
            protected = set(keys_list)
            for key in self.blocks:
                if key not in protected and key not in self.in_flight:
                    evicted.append(key)
                    if len(evicted) == num_blocks_to_evict:
                        break
            else:
                # Could not collect enough eviction candidates
                return
            for key in evicted:
                del self.blocks[key]

        # Mark blocks as in-flight
        for key in blocks_to_store:
            self.in_flight[key] = job_id

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
        keys_list = list(job_metadata.keys)
        primary_write_spec = job_metadata.spec

        # Validate spec type and consistency
        assert isinstance(primary_write_spec, CPULoadStoreSpec), (
            f"Expected CPULoadStoreSpec, got {type(primary_write_spec)}"
        )
        assert len(keys_list) == len(primary_write_spec.block_ids), (
            f"Length mismatch: {len(keys_list)} keys but "
            f"{len(primary_write_spec.block_ids)} block_ids in spec"
        )

        # Verify all blocks exist
        for key in keys_list:
            if key not in self.blocks:
                return

        # Mark blocks as in-flight
        for key in keys_list:
            self.in_flight[key] = job_id

        # Create internal job metadata
        internal_job_metadata = _JobMetadata(
            job_id=job_id, keys=keys_list, is_store=False
        )

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
            del self.in_flight[key]
        # Return simplified JobResult (only job_id and success)
        self.completed_jobs.append(JobResult(job_id=job_metadata.job_id, success=True))

    def _complete_load_job(self, job_metadata: _JobMetadata):
        """Complete a load job by removing in-flight markers."""
        for key in job_metadata.keys:
            del self.in_flight[key]
        # Return simplified JobResult (only job_id and success)
        self.completed_jobs.append(JobResult(job_id=job_metadata.job_id, success=True))

    def touch(self, keys: Iterable[OffloadKey]):
        """
        Mark blocks as recently used (move to end of LRU list).

        Args:
            keys: Blocks to mark as recently used.
        """
        for key in reversed(list(keys)):
            if key in self.blocks:
                self.blocks.move_to_end(key)

    def get_tier_name(self) -> str:
        """
        Get the name of this tier.

        Returns:
            Tier name string.
        """
        return self.tier_name

    def get_num_blocks(self) -> int:
        """Get the number of blocks currently stored in this tier."""
        return len(self.blocks)

    def get_num_in_flight(self) -> int:
        """Get the number of blocks currently in-flight."""
        return len(self.in_flight)

    def clear(self):
        """Clear all blocks and in-flight transfers (for testing)."""
        self.blocks.clear()
        self.in_flight.clear()
        self.completed_jobs.clear()
        self.pending_jobs.clear()
