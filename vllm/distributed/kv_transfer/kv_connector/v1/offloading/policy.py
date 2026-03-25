# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import ReqId
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


class OffloadingPolicy(ABC):
    """
    Abstraction for offloading policy decisions.
    """

    @abstractmethod
    def should_lookup(self, request: "Request") -> bool:
        """Whether to look up offloaded tokens for this request"""
        pass

    @abstractmethod
    def allow_redundant_load_prevention(self) -> bool:
        """
        Whether to enable the redundant load prevention mechanism.

        Policies that never produce overlapping load requests (e.g.
        preemptions-only) can return False to skip this bookkeeping.
        """
        pass

    @abstractmethod
    def get_store_candidate_req_ids(
        self,
        scheduler_output: "SchedulerOutput",
        all_req_ids: list["ReqId"],
    ) -> Iterable["ReqId"]:
        """Return the request IDs that should be considered for storing."""
        pass

    @abstractmethod
    def should_store_in_wait_for_save(self) -> bool:
        """Whether prepare_store_kv should be called during wait_for_save"""
        pass


class OffloadAllPolicy(OffloadingPolicy):
    """
    Default policy: offload all running requests continuously.
    """

    def should_lookup(self, request: "Request") -> bool:
        return True

    def allow_redundant_load_prevention(self) -> bool:
        return True

    def get_store_candidate_req_ids(
        self,
        scheduler_output: "SchedulerOutput",
        all_req_ids: list["ReqId"],
    ) -> Iterable["ReqId"]:
        return all_req_ids

    def should_store_in_wait_for_save(self) -> bool:
        return True
