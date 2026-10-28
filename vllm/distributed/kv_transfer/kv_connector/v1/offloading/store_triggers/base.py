# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import ClassVar

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    ReqId,
    TransferJob,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventsTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.state import (
    RequestOffloadState,
    SchedulerOffloadConfig,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    StoreTrigger,
)


@dataclass
class StoreJobsOutput:
    """The store jobs of one handler in one scheduler step."""

    # The scheduler gives each job its ID, in this order.
    jobs: list[tuple[TransferJob, TransferJobStatus]] = field(default_factory=list)
    # Number of prepare_store calls that returned None.
    num_allocation_failures: int = 0


class StoreTriggerHandler(ABC):
    """Makes the store jobs of one StoreTrigger.

    The handler offers chunks to the manager and describes the store jobs. The
    scheduler gives the jobs their IDs, tracks them, and completes them.

    Handlers are isolated. A handler does not know of other handlers, and its
    result does not depend on which handlers run, or on their order. A handler
    keeps its own state. The scheduler holds only global state, and gives the
    same hooks to each handler.
    """

    # The trigger that this handler serves.
    trigger: ClassVar[StoreTrigger]

    def __init__(
        self,
        config: SchedulerOffloadConfig,
        manager: OffloadingManager,
        events_tracker: OffloadingEventsTracker,
    ):
        self._config = config
        # A handler offers chunks only through _prepare_store.
        self._manager = manager
        self._events_tracker = events_tracker

    def _prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """Offer keys to the manager, with the trigger of this handler."""
        return self._manager.prepare_store(keys, req_context, self.trigger)

    def _make_store_job(
        self,
        req_id: ReqId,
        keys: Collection[OffloadKey],
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
        fenced_block_ids: list[int] | None = None,
        deferred_fence_block_ids: list[int] | None = None,
    ) -> tuple[TransferJob, TransferJobStatus]:
        """Make a store job, and the status that tracks it.

        Args:
            req_id: the request that owns the blocks.
            keys: the keys that the job stores.
            src_spec: the GPU blocks to read.
            dst_spec: where the manager stores the blocks.
            fenced_block_ids: the source blocks to fence from job creation.
            deferred_fence_block_ids: the source blocks to fence after the
                request finishes.

        """
        job = TransferJob(req_id=req_id, src_spec=src_spec, dst_spec=dst_spec)
        job_status = TransferJobStatus(
            req_id=req_id,
            pending_count=self._config.num_workers,
            keys=set(keys),
            is_store=True,
            deferred_fence_block_ids=deferred_fence_block_ids,
            fenced_block_ids=fenced_block_ids,
        )
        return job, job_status

    def on_load_scheduled(
        self,
        req_status: RequestOffloadState,
        num_cached_tokens: int,
    ) -> None:
        """Called when the scheduler makes a load job for a request.

        Args:
            req_status: the request of the load.
            num_cached_tokens: the number of tokens on the GPU after the load.

        """
        return

    def on_request_removed(self, req_id: ReqId) -> None:
        """Called after the scheduler removes the state of a request.

        Args:
            req_id: the removed request.

        """
        return

    def on_request_restarted(self, req_id: ReqId) -> None:
        """Called when a request starts again from token 0, for example after
        a preemption. Its GPU blocks are gone.

        Args:
            req_id: the restarted request.

        """
        return

    def reset(self) -> None:
        """Called when the scheduler resets the cache, after it clears the jobs."""
        return

    @abstractmethod
    def first_pending_chunk_idx(
        self, req_status: RequestOffloadState, config_idx: int
    ) -> int:
        """The first chunk of a group that this handler can still store.

        The scheduler zeroes stale sliding-window block ids only from the
        lowest of these chunks on.

        Args:
            req_status: the request that holds the group.
            config_idx: the index of the group in the offloading config.

        """

    @abstractmethod
    def build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
        req_states: Mapping[ReqId, RequestOffloadState],
    ) -> StoreJobsOutput:
        """Offer the chunks of this trigger to the manager, and make the jobs.

        Args:
            scheduler_output: the output of the current scheduler step.
            req_states: the state of each tracked request.

        Returns:
            The new store jobs, in the order of their job IDs.

        """
