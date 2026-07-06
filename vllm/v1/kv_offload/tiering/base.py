# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Abstract interfaces and data types for the secondary tiering layer.
"""

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingEvent,
    OffloadingMetricMetadata,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
        OffloadingConnectorStats,
    )
    from vllm.v1.kv_offload.base import OffloadingSpec

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
    async jobs; get_finished_jobs() polls for completion.
    """

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
    ) -> None:
        """
        Args:
            offloading_spec: Offloading configuration.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory
                from the registered tier type.
        """
        self._offloading_spec = offloading_spec
        self._primary_kv_view: memoryview = primary_kv_view
        self.tier_type = tier_type

    @abstractmethod
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        """
        Check whether a block exists in this secondary tier.

        Args:
            key: Offload key to look up.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            HIT if the block is present and ready,
            MISS if not found,
            or RETRY if the block is being transferred (retry later).
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

        Report completion via ``get_finished_jobs()``.

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

        Report completion via ``get_finished_jobs()``.

        Args:
            job_metadata: Job metadata including job_id, keys, and block_ids
                          identifying the primary-tier slots to write into.
        """
        pass

    @abstractmethod
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Return all jobs (loads and stores) that completed since the last call.

        The framework uses these results to release resources and finalize
        transfers.

        Returns:
            Iterable of JobResult objects for jobs finished since the
            last call.
        """
        pass

    def has_pending_work(self) -> bool:
        """Whether this tier needs the engine to keep stepping.

        While True, on_schedule_end() and get_finished_jobs() continue
        to be called even when no requests are scheduled.
        """
        return False

    def take_events(self) -> Iterable[OffloadingEvent]:
        """Take KV events for storage state owned by this tier."""
        return ()

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as recently used for eviction policy.

        Args:
            keys: Offload keys to mark as recently used.
            req_context: Per-request context.
        """
        return

    @abstractmethod
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """
        Called when a new request is first seen by the scheduler.

        Returns a RequestOffloadingContext expressing this tier's preference
        for how blocks should be offloaded for this request.

        Args:
            req_context: Per-request context.
        """
        pass

    def on_request_finished(self, req_context: ReqContext) -> None:
        """
        Called when a request has finished.

        By the time this is called, all per-request calls for this request
        (submit_store, submit_load, touch) have already been issued, and none
        will follow. Note this does NOT imply the tier's transfers have
        completed: jobs already submitted may still be in flight and will
        report via get_finished_jobs(). This is the right place to release
        per-request bookkeeping.

        Args:
            req_context: per-request context.
        """
        return

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        """Called once at the end of each scheduler step.

        Secondary tiers may override this for per-step cleanup or
        deferred work submission.
        """
        return

    @abstractmethod
    def drain_jobs(self) -> None:
        """Block until every submitted load/store job has completed or failed.

        After this returns, no tier I/O is touching the primary memoryview,
        and every submitted job's result is available from `get_finished_jobs()`
        (yielded by a prior call or queued for the next one). Used by
        `TieringOffloadingManager.reset_cache` to release primary slots
        without racing with in-flight transfers.

        Implementations must not abort a mid-flight transfer: a partial copy
        would corrupt either the primary memoryview or the secondary backing
        store. Queued (not-yet-started) transfers may be cancelled, but their
        failure result must still appear in `get_finished_jobs()`.
        """
        pass

    def shutdown(self) -> None:
        """Release resources held by this tier (threads, connections, etc.)."""
        return

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        """Return Prometheus metric definitions emitted by this tier."""
        return {}

    def get_stats(self) -> "OffloadingConnectorStats | None":
        """Return and reset metric observations collected by this tier."""
        return None
