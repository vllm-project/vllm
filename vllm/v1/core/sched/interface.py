# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional

from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.metrics.stats import SchedulerStats
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager


class SchedulerInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def schedule(self) -> "SchedulerOutput":
        """Schedule the requests to process in this scheduling step.

        The scheduling decision is made at the iteration level. Each scheduling
        step corresponds to a single forward pass of the model. Therefore, this
        method is called repeatedly by a busy loop in the engine.

        Essentially, the scheduler produces a dictionary of {req_id: num_tokens}
        that specifies how many tokens to process for each request in this
        scheduling step. For example, num_tokens can be as large as the number
        of prompt tokens for new requests, or it can be 1 for the requests that
        are auto-regressively generating new tokens one by one. Otherwise, it
        can be somewhere in between in case of chunked prefills, prefix caching,
        speculative decoding, etc.

        Additionally, the scheduler also returns useful data about each request
        or the batch as a whole. The model runner will use this information in
        preparing inputs to the model.

        Returns:
            A SchedulerOutput object containing information about the scheduled
            requests.
        """
        raise NotImplementedError

    @abstractmethod
    def get_grammar_bitmask(
        self, scheduler_output: "SchedulerOutput"
    ) -> "GrammarOutput | None":
        raise NotImplementedError

    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        """Update the scheduler state based on the model runner output.

        This method is called after the model runner has processed the scheduled
        requests. The model runner output includes generated token ids, draft
        token ids for next step, etc. The scheduler uses this information to
        update its states, checks the finished requests, and returns the output
        for each request.

        Returns:
            A dict of client index to EngineCoreOutputs object containing the
            outputs for each request originating from that client.
        """
        raise NotImplementedError

    @abstractmethod
    def update_draft_token_ids(self, draft_token_ids: "DraftTokenIds") -> None:
        """Update requests with newly generated draft token ids, applying
        structured output grammar validation if needed.

        Args:
            draft_token_ids: The input draft token ids for each request.
        """
        raise NotImplementedError

    @abstractmethod
    def update_draft_token_ids_in_output(
        self, draft_token_ids: "DraftTokenIds", scheduler_output: "SchedulerOutput"
    ) -> None:
        """Update scheduler output with newly generated draft token ids, applying
        structured output grammar validation if needed.

        Args:
            draft_token_ids: The input draft token ids for each request.
            scheduler_output: Update the given scheduler_output
                with the corresponding draft token ids.
        """
        raise NotImplementedError

    @abstractmethod
    def add_request(self, request: "Request") -> None:
        """Add a new request to the scheduler's internal queue.

        Args:
            request: The new request being added.
        """
        raise NotImplementedError

    @abstractmethod
    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: "RequestStatus",
    ) -> None:
        """Finish the requests in the scheduler's internal queue. If the request
        is not in the queue, this method will do nothing.

        This method is called in two cases:
        1. When the request is aborted by the client.
        2. When the frontend process detects a stop string of the request after
           de-tokenizing its generated tokens.

        Args:
            request_ids: A single or a list of request IDs.
            finished_status: The finished status of the given requests.
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """Number of unfinished requests in the scheduler's internal queue."""
        raise NotImplementedError

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests in the scheduler's
        internal queue."""
        return self.get_num_unfinished_requests() > 0

    @abstractmethod
    def has_finished_requests(self) -> bool:
        """Returns True if there are finished requests that need to be cleared.
        NOTE: This is different from `not self.has_unfinished_requests()`.

        The scheduler maintains an internal list of the requests finished in the
        previous step. This list is returned from the next call to schedule(),
        to be sent to the model runner in the next step to clear cached states
        for these finished requests.

        This method checks if this internal list of finished requests is
        non-empty. This information is useful for DP attention.
        """
        raise NotImplementedError

    def has_work(self) -> bool:
        """Returns True if one of the below exist:
        1. Unfinished requests
        2. Finished requests not yet returned in SchedulerOutputs
        3. KV Connector work (e.g. KV blocks being transferred)
        """
        return (
            self.has_unfinished_requests()
            or self.has_finished_requests()
            or self.has_kv_connector_work()
        )

    @abstractmethod
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the prefix cache for KV cache.

        This is particularly required when the model weights are live-updated.

        Args:
            reset_running_requests: If True, all the running requests will be
                preempted and moved to the waiting queue. Otherwise, this method
                will only reset the KV prefix cache when there is no running request
                taking KV cache.
        """
        raise NotImplementedError

    @abstractmethod
    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        raise NotImplementedError

    @abstractmethod
    def make_stats(self) -> Optional["SchedulerStats"]:
        """Make a SchedulerStats object for logging.

        The SchedulerStats object is created for every scheduling step.
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        raise NotImplementedError

    def get_kv_connector(self) -> Optional["KVConnectorBase_V1"]:
        return None

    def has_kv_connector_work(self) -> bool:
        """Returns True if there's KV Connector work
        (e.g. KV blocks being transferred)
        """
        return False
