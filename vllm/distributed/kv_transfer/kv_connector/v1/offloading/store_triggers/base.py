# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    ReqId,
    TransferJob,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventsTracker,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.base import OffloadingManager

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        RequestOffloadState,
        SchedulerOffloadConfig,
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
    scheduler gives the jobs their IDs and tracks them.
    """

    def __init__(
        self,
        config: "SchedulerOffloadConfig",
        manager: OffloadingManager,
        events_tracker: OffloadingEventsTracker,
    ):
        self._config = config
        self._manager = manager
        self._events_tracker = events_tracker

    def on_load_scheduled(
        self, req_status: "RequestOffloadState", num_cached_tokens: int
    ) -> None:
        """Called when the scheduler makes a load job for a request.

        Args:
            req_status: the request of the load.
            num_cached_tokens: the number of tokens on the GPU after the load.

        """
        return

    def reset(self, req_states: Mapping[ReqId, "RequestOffloadState"]) -> None:
        """Called when the scheduler resets the cache, after it clears the jobs.

        Args:
            req_states: the requests that stay active after the reset.

        """
        return

    @abstractmethod
    def build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
        req_states: Mapping[ReqId, "RequestOffloadState"],
    ) -> StoreJobsOutput:
        """Offer the chunks of this trigger to the manager, and make the jobs.

        Args:
            scheduler_output: the output of the current scheduler step.
            req_states: the state of each tracked request.

        Returns:
            The new store jobs, in the order of their job IDs.

        """
