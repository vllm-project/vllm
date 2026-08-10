# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.v1.core.sched.plugins.interface import (
    CandidateSelection,
    PreemptionPlugin,
    QueueSortPlugin,
    WaitingQueue,
)
from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    PriorityRequestQueue,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.v1.core.sched.request_queue import RequestQueue
    from vllm.v1.request import Request


class FCFSSchedulerPlugin(QueueSortPlugin, PreemptionPlugin):
    name = "fcfs"

    def create_request_queue(self) -> "RequestQueue":
        return FCFSRequestQueue()

    def select_queue(
        self,
        waiting_head: "Request | None",
        skipped_head: "Request | None",
    ) -> WaitingQueue | None:
        if skipped_head is not None:
            return WaitingQueue.SKIPPED
        if waiting_head is not None:
            return WaitingQueue.WAITING
        return None

    def preemption_key(
        self,
        request: "Request",
        running_position: int,
    ) -> tuple[int | float, ...]:
        return (running_position,)

    def order_candidates(
        self,
        waiting: "Sequence[Request]",
        skipped: "Sequence[Request]",
    ) -> list[CandidateSelection]:
        return [
            *(CandidateSelection(request, WaitingQueue.SKIPPED) for request in skipped),
            *(CandidateSelection(request, WaitingQueue.WAITING) for request in waiting),
        ]


class PrioritySchedulerPlugin(QueueSortPlugin, PreemptionPlugin):
    name = "priority"

    def create_request_queue(self) -> "RequestQueue":
        return PriorityRequestQueue()

    def select_queue(
        self,
        waiting_head: "Request | None",
        skipped_head: "Request | None",
    ) -> WaitingQueue | None:
        if waiting_head is None:
            return WaitingQueue.SKIPPED if skipped_head is not None else None
        if skipped_head is None:
            return WaitingQueue.WAITING
        if waiting_head < skipped_head:
            return WaitingQueue.WAITING
        return WaitingQueue.SKIPPED

    def preemption_key(
        self,
        request: "Request",
        running_position: int,
    ) -> tuple[int | float, ...]:
        return (request.priority, request.arrival_time)

    def order_candidates(
        self,
        waiting: "Sequence[Request]",
        skipped: "Sequence[Request]",
    ) -> list[CandidateSelection]:
        candidates = [
            *(CandidateSelection(request, WaitingQueue.WAITING) for request in waiting),
            *(CandidateSelection(request, WaitingQueue.SKIPPED) for request in skipped),
        ]
        return sorted(candidates, key=lambda candidate: candidate.request)
