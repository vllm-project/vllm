# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from typing import ClassVar, Protocol

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    ReqId,
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


class StoreJobSink(Protocol):
    """The scheduler's store-job intake for one handler, in one step.

    The sink carries the trigger of its handler, so a handler cannot stamp a
    job with the wrong cause. The scheduler gives each job its ID and tracks
    it.
    """

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """Offer keys to the manager, with the trigger of this handler."""
        ...

    def add_job(
        self,
        req_id: ReqId,
        keys: Collection[OffloadKey],
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
        *,
        fenced_block_ids: list[int] | None = None,
        deferred_fence_block_ids: list[int] | None = None,
    ) -> None:
        """Make a store job, and let the scheduler track it."""
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
        sink: StoreJobSink,
    ) -> None:
        """Offer the chunks of this trigger to the manager, and make the jobs.

        Args:
            scheduler_output: the output of the current scheduler step.
            req_states: the state of each tracked request.
            sink: the intake for the store jobs of this step.

        """
