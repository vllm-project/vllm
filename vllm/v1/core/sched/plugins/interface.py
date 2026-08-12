# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.request_queue import RequestQueue
    from vllm.v1.request import Request


class WaitingQueue(Enum):
    WAITING = "waiting"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class CandidateSelection:
    request: "Request"
    queue: WaitingQueue


@dataclass(frozen=True)
class CandidateInfo:
    queue: WaitingQueue
    queue_position: int
    waiting_time: float


@dataclass(frozen=True)
class SchedulingCycleState:
    now: float
    block_size: int
    token_budget: int
    encoder_budget: int
    num_running_requests: int
    candidates: Mapping[str, CandidateInfo]


@dataclass(frozen=True)
class FilterResult:
    allowed: bool
    reason: str | None = None

    @classmethod
    def allow(cls) -> "FilterResult":
        return cls(allowed=True)

    @classmethod
    def reject(cls, reason: str | None = None) -> "FilterResult":
        return cls(allowed=False, reason=reason)


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

    @abstractmethod
    def order_candidates(
        self,
        waiting: "RequestQueue",
        skipped: "RequestQueue",
        limit: int,
    ) -> list[CandidateSelection]:
        """Return up to ``limit`` candidates in base QueueSort order."""
        raise NotImplementedError


class FilterPlugin(SchedulerPlugin, ABC):
    """Plugin interface for the Filter extension point."""

    @abstractmethod
    def filter(
        self,
        request: "Request",
        state: SchedulingCycleState,
    ) -> FilterResult:
        """Return whether a request may be selected in this cycle."""
        raise NotImplementedError


class ScorePlugin(SchedulerPlugin, ABC):
    """Plugin interface for the Score extension point."""

    @abstractmethod
    def score(
        self,
        request: "Request",
        state: SchedulingCycleState,
    ) -> float:
        """Return a finite score. Higher values are preferred."""
        raise NotImplementedError


class PreemptionPlugin(SchedulerPlugin, ABC):
    """Plugin interface for the Preemption extension point."""

    def select_victim_index(self, running: Sequence["Request"]) -> int:
        """Return the index of the request to preempt."""
        return max(
            range(len(running)),
            key=lambda index: self.preemption_key(running[index], index),
        )

    @abstractmethod
    def preemption_key(
        self,
        request: "Request",
        running_position: int,
    ) -> tuple[int | float, ...]:
        """Return a key for ranking core-approved preemption candidates."""
        raise NotImplementedError
