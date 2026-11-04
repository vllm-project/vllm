# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    ReqId,
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
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    StoreTrigger,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


class StoreJobSink(Protocol):
    """The scheduler's store-job intake for one handler, in one step.

    The sink carries the trigger of its handler, so a handler cannot stamp a
    job with the wrong cause. The scheduler gives each job its ID and tracks
    it.
    """

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext | None
    ) -> PrepareStoreOutput | None:
        """Offer keys to the manager, with the trigger of this handler."""
        ...

    def add_job(
        self,
        req_id: ReqId | None,
        keys: Collection[OffloadKey],
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
        *,
        fenced_block_ids: list[int] | None = None,
        deferred_fence_block_ids: list[int] | None = None,
    ) -> None:
        """Make a store job, and let the scheduler track it.

        The fences apply only to a job that belongs to a request: a
        request-free job is waited for before the next forward pass, so no
        block can be reallocated under it.
        """
        ...


class StoreTriggerHandler(ABC):
    """Makes the store jobs of one StoreTrigger.

    The handler offers chunks to the manager through the sink, and describes
    the store jobs. The scheduler gives the jobs their IDs, tracks them, and
    completes them.

    Handlers are isolated. A handler does not know of other handlers, and its
    result does not depend on which handlers run, or on their order. A handler
    keeps its own state. The scheduler holds only global state, and gives the
    same hooks to each handler.
    """

    # The trigger that this handler serves.
    trigger: ClassVar[StoreTrigger]

    @classmethod
    def check_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        extra_config: Mapping[str, Any],
        events_tracker: OffloadingEventsTracker,
    ) -> None:
        """Raise ValueError if this handler cannot run with the given config.

        StoreHandlerFactory calls it before it makes the handler.

        Args:
            vllm_config: the engine config.
            kv_cache_config: the KV cache layout of this engine.
            extra_config: kv_connector_extra_config of this connector.
            events_tracker: the events tracker that the handler will get.

        """
        return

    def __init__(
        self,
        config: SchedulerOffloadConfig,
        events_tracker: OffloadingEventsTracker,
    ):
        self._config = config
        self._events_tracker = events_tracker

    def on_load_scheduled(
        self,
        req_status: RequestOffloadState,
        num_cached_tokens: int,
        load_chunk_ranges: Sequence[tuple[int, int]],
        load_block_ids: Sequence[Sequence[int]],
    ) -> None:
        """Called when the scheduler makes a load job for a request.

        Args:
            req_status: the request of the load.
            num_cached_tokens: the number of tokens on the GPU after the load.
            load_chunk_ranges: per group, the [start, end) range of the
                loaded chunks. An empty range if the group loads nothing.
            load_block_ids: per group, the GPU block ids of the request after
                the allocation. The request state gets them in a later step.

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

    def reset(self, req_states: Mapping[ReqId, "RequestOffloadState"]) -> None:
        """Called when the scheduler resets the cache, after it clears the jobs.

        Args:
            req_states: the requests that stay active after the reset.

        """
        return

    def on_load_complete(self, job_status: TransferJobStatus) -> None:
        """Called when a load job of a tracked request completes.

        Args:
            job_status: the status of the completed load job.

        """
        return

    def on_request_blocks_allocated(
        self,
        req_status: "RequestOffloadState",
        block_ids: Sequence[Sequence[int]],
        start_block_idxs: Sequence[int],
    ) -> None:
        """Called when the block pool gives blocks to a request.

        Called before on_blocks_allocated() of the same step, also for a
        request that is not in the scheduler output of the step.

        Args:
            req_status: the request that gets the blocks.
            block_ids: per offloaded group, the allocated block ids, starting
                at start_block_idxs[g]. 0 is the null block.
            start_block_idxs: per offloaded group, the position in the request
                of the first id of block_ids.

        """
        return

    def on_blocks_allocated(self, block_ids: Collection[int]) -> None:
        """Receive the blocks that the pool gave out in the current step.

        Called once per step, after the step's allocations and before
        build_store_jobs(). Includes the blocks of all KV cache groups and the
        destinations of block copies.

        Args:
            block_ids: the allocated block ids, without the null block. A
                read-only snapshot: the scheduler keeps its own set.

        """
        return

    @abstractmethod
    def first_live_chunk_idx(
        self, req_status: RequestOffloadState, config_idx: int
    ) -> int | None:
        """The first chunk of a group whose GPU block ids this handler may
        still read.

        The scheduler zeroes stale sliding-window block ids only from the
        lowest of these chunks on.

        Args:
            req_status: the request that holds the group.
            config_idx: the index of the group in the offloading config.

        Returns:
            The first live chunk, or None if this handler reads no chunks of
            the request.

        """

    @abstractmethod
    def build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
        req_states: Mapping[ReqId, RequestOffloadState],
        sink: StoreJobSink,
    ) -> None:
        """Offer the chunks of this trigger to the manager, and make the jobs.

        Args:
            scheduler_output: the output of the current scheduler step.
            req_states: the state of each tracked request.
            sink: the intake for the store jobs of this step.

        """
