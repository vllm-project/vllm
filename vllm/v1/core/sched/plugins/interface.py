# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.request_queue import RequestQueue
    from vllm.v1.request import Request


class WaitingQueue(Enum):
    WAITING = "waiting"
    SKIPPED = "skipped"


class SchedulerPlugin:
    """Base interface for scheduler framework plugins."""

    name: str


class QueueSortPlugin(SchedulerPlugin, ABC):
    """Plugin interface for the QueueSort extension point."""

    @abstractmethod
    def create_request_queue(self) -> "RequestQueue":
        """Create a queue using this plugin's base ordering."""
        raise NotImplementedError

    @abstractmethod
    def select_queue(
        self,
        waiting_head: "Request | None",
        skipped_head: "Request | None",
    ) -> WaitingQueue | None:
        """Select the queue from which the core should schedule next."""
        raise NotImplementedError


class PreemptionPlugin(SchedulerPlugin, ABC):
    """Plugin interface for the Preemption extension point."""

    @abstractmethod
    def preemption_key(
        self,
        request: "Request",
        running_position: int,
    ) -> tuple[int | float, ...]:
        """Return a key for ranking core-approved preemption candidates."""
        raise NotImplementedError
