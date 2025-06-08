# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque

from vllm.v1.request import Request


class WaitingQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def push_request(self,
                     request: Request,
                     priority: int = 0,
                     arrival_time: float = 0.0) -> None:
        """Push a request back to the queue (used for skipped requests)."""
        pass

    @abstractmethod
    def extend_left_requests(self, requests: WaitingQueue) -> None:
        """Extend left with requests from another WaitingQueue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self):
        """Iterate over the queue according to the policy."""
        pass


class FCFSWaitingQueue(WaitingQueue, deque[Request]):
    """A first-come-first-served queue that supports deque operations."""

    def __init__(self) -> None:
        super().__init__()
        deque.__init__(self)

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def push_request(self,
                     request: Request,
                     priority: int = 0,
                     arrival_time: float = 0.0) -> None:
        """Push a request back to the queue (used for skipped requests)."""
        self.appendleft(request)

    def extend_left_requests(self, requests: WaitingQueue) -> None:
        """Extend left with requests from another WaitingQueue."""
        if isinstance(requests, FCFSWaitingQueue):
            self.extendleft(requests)
        else:
            # Handle priority queue case
            for item in requests:
                if isinstance(item, tuple):
                    _, _, request = item
                    self.appendleft(request)
                else:
                    self.appendleft(item)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return deque.__len__(self)

    def __iter__(self):
        """Iterate over the queue according to FCFS policy."""
        # Cast to deque to access its iterator
        return iter(deque[Request](self))


class PriorityWaitingQueue(WaitingQueue):
    """A priority queue that supports heap operations."""

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, Request]] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap,
                       (request.priority, request.arrival_time, request))

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, request = heapq.heappop(self._heap)
        return request

    def push_request(self,
                     request: Request,
                     priority: int = 0,
                     arrival_time: float = 0.0) -> None:
        """Push a request back to the queue (used for skipped requests)."""
        heapq.heappush(self._heap, (priority, arrival_time, request))

    def extend_left_requests(self, requests: WaitingQueue) -> None:
        """Extend left with requests from another WaitingQueue."""
        if isinstance(requests, PriorityWaitingQueue):
            for item in requests._heap:
                heapq.heappush(self._heap, item)
        else:
            # Handle FCFS queue case
            for request in requests:
                heapq.heappush(
                    self._heap,
                    (request.priority, request.arrival_time, request))

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap = [(p, t, r) for p, t, r in self._heap if r != request]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self):
        """Iterate over the queue according to priority policy."""
        return iter(self._heap)


def create_waiting_queue(policy: str, ) -> WaitingQueue:
    """Create waiting queue based on scheduling policy."""
    if policy == "priority":
        return PriorityWaitingQueue()
    else:
        return FCFSWaitingQueue()
