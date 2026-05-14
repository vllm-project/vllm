# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Abstract interfaces and data types for the secondary tiering layer.
"""

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vllm.v1.kv_offload.base import OffloadKey, ReqContext

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Type alias for job IDs used in async transfer tracking
JobId = int


@dataclass
class JobMetadata:
    """Metadata for an in-flight async transfer job."""

    job_id: JobId
    keys: Collection[OffloadKey]
    block_ids: np.ndarray
    is_promotion: bool
    req_context: ReqContext


@dataclass
class JobResult:
    """Result of an async transfer job (successful or failed)."""

    job_id: JobId
    success: bool


class SecondaryTierManager(ABC):
    """
    Abstract interface for managing a single non-primary offloading tier.

    Secondary tiers cannot directly access GPU memory. All data transfers
    must go through the CPU (primary) tier:
      - Store: GPU → CPU (primary) → secondary  (cascade)
      - Load:  secondary → CPU (primary) → GPU  (promotion)

    IMPORTANT: All methods run in the Scheduler process and must be
    lightweight and non-blocking. submit_load() and submit_store() submit
    async jobs; get_finished() polls for completion.
    """

    def __init__(self, vllm_config: "VllmConfig", primary_kv_view: memoryview) -> None:
        self._vllm_config = vllm_config
        self._primary_kv_view: memoryview = primary_kv_view

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

        This method must be lightweight and non-blocking: allocate metadata
        and submit the transfer, but do NOT perform the data copy on the
        calling thread.

        Preconditions (guaranteed by the framework):
          - ``job_metadata.block_ids`` are valid primary-tier slots, pinned
            (ref-counted) for the duration of the transfer.

        The implementation is responsible for:
          1. Filtering out blocks already present in this tier
          2. Evicting blocks if capacity is needed
          3. Allocating space in this tier
          4. Submitting the async transfer (read from primary via block_ids)

        Report completion via ``get_finished()``.

        Args:
            job_metadata: Job metadata including job_id, keys, and block_ids
                          identifying the primary-tier slots to read from.
        """
        pass

    @abstractmethod
    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit an async job to load blocks from this secondary tier to the
        primary tier.

        This method must be lightweight and non-blocking: mark blocks as
        in-flight and submit the transfer, but do NOT perform the data copy
        on the calling thread.

        Preconditions (guaranteed by the framework):
          - ``job_metadata.block_ids`` are allocated primary-tier slots
            ready to receive data.

        The implementation must copy data from this tier into the
        primary-tier slots identified by ``block_ids``.

        Report completion via ``get_finished()``.

        Args:
            job_metadata: Job metadata including job_id, keys, and block_ids
                          identifying the primary-tier slots to write into.
        """
        pass

    @abstractmethod
    def get_finished(self) -> Iterable[JobResult]:
        """
        Return all jobs (loads and stores) that completed since the last call.

        The framework uses these results to release resources and finalize
        transfers.

        Returns:
            Iterable of JobResult objects for jobs finished since the
            last call.
        """
        pass

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as recently used for eviction policy.

        Args:
            keys: Offload keys to mark as recently used.
            req_context: Per-request context.
        """
        return

    def shutdown(self) -> None:
        """Release resources held by this tier (threads, connections, etc.)."""
        return

    @staticmethod
    @abstractmethod
    def get_tier_type() -> str:
        """
        Get the type identifier of this tier (e.g., "example", "storage").

        Must match the "type" field in the tier config dict.

        Returns:
            Tier type string.
        """
        pass
