# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Abstract interfaces and data types for the secondary tiering layer.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field

from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadKey, ReqContext

# Type alias for job IDs used in async transfer tracking
JobId = int


@dataclass
class JobMetadata:
    """Metadata for an in-flight async transfer job."""

    job_id: JobId
    keys: list[OffloadKey]
    spec: LoadStoreSpec
    req_context: ReqContext = field(default_factory=ReqContext)


@dataclass
class JobResult:
    """Result of an async transfer job (successful or failed)."""

    job_id: JobId
    success: bool


class SecondaryTierManager(ABC):
    """
    Abstract interface for managing a single non-primary offloading tier.

    Secondary tiers cannot directly access GPU memory. All data transfers
    must go through the primary tier (implemented as CPU in current version):
      - Store: GPU → primary → secondary  (cascade)
      - Load:  secondary → primary → GPU  (promotion)

    IMPORTANT: All methods run in the Scheduler process and must be
    lightweight and non-blocking. submit_load() and submit_store() submit
    async jobs; get_finished() polls for completion.
    """

    @abstractmethod
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Check whether a block exists in this secondary tier.

        Args:
            key: Offload key to look up.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            True if the block is present and ready,
            False if not found,
            or None if the block is being transferred (retry later).
        """
        pass

    @abstractmethod
    def submit_store(self, job_metadata: JobMetadata) -> None:
        """
        Submit an async job to store blocks from the primary tier to this
        secondary tier.

        This method is lightweight: it allocates metadata and submits the
        transfer job, but does NOT perform the actual data transfer on the
        calling thread.

        The caller (TieringOffloadingManager) must have already called
        primary.prepare_read(keys) to obtain job_metadata.spec and
        to increment ref_cnt on those blocks. ref_cnt will be decremented
        when get_finished() reports this job_id as complete and
        primary.unprepare_read() is called.

        This method is responsible for:
          1. Filtering out blocks already present in this secondary tier
          2. Evicting blocks from this secondary tier if needed (secondary
             tiers are responsible for their own evictions)
          3. Allocating space in this secondary tier
          4. Submitting the async transfer: primary → secondary

        Args:
            job_metadata: Job metadata including job_id, keys, and
                          spec for reading blocks from the primary tier
                          (obtained via primary.prepare_read()).
                          spec is a CPULoadStoreSpec with block_ids.
        """
        pass

    @abstractmethod
    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit an async job to load blocks from this secondary tier to the
        primary tier.

        This method is lightweight: it marks blocks as in-flight and submits
        the transfer job, but does NOT perform the actual data transfer on
        the calling thread.

        The caller (TieringOffloadingManager) must have already called
        primary.prepare_write(keys) to obtain job_metadata.spec and
        to allocate space in the primary tier. When get_finished() reports
        this job_id as complete, primary.complete_write() is called to make
        the blocks available for GPU loads.

        Args:
            job_metadata: Job metadata including job_id, keys, and
                          spec for writing blocks into the primary tier
                          (obtained via primary.prepare_write()).
                          spec is a CPULoadStoreSpec with block_ids.
        """
        pass

    @abstractmethod
    def get_finished(self) -> Iterable[JobResult]:
        """
        Poll for finished async jobs (both loads and stores).

        This is the mechanism by which the TieringOffloadingManager learns
        that a transfer has finished and can:
          - Call primary.unprepare_read() to decrement ref_cnt (for stores)
          - Call primary.complete_write() to make blocks loadable (for loads)

        Returns:
            Iterable of JobResult objects for all jobs that have
            finished since the last call.
        """
        pass

    def set_primary_view(self, view: memoryview) -> None:
        """
        Provide a long-lived memoryview of the primary-tier CPU tensor.

        Called once by TieringOffloadingManager during initialisation.
        Override to store the view for use in `submit_store` and `submit_load`.
        Use `view.strides[0]` to obtain the byte stride between block slots.

        Args:
            view: Memoryview of the primary tier's CPU KV cache tensor.
        """
        return

    def touch(self, keys: Iterable[OffloadKey]):
        """
        Mark blocks as recently used for eviction policy.

        Args:
            keys: Offload keys to mark as recently used.
        """
        return

    @abstractmethod
    def get_tier_name(self) -> str:
        """
        Get the name of this tier (e.g., "Storage", "Network").

        Returns:
            Tier name string.
        """
        pass
