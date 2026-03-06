# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum
from typing import Any

from vllm.v1.core.sched.policy.shortest_job_first import (
    TimeAndLengthScorer,
    WeightedScoreSorter,
)
from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    SJF = "sjf"


class RequestQueue(ABC):
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
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
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
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class RequestHeap(RequestQueue):
    """A queue that supports heap operations."""

    def __init__(self) -> None:
        self._heap: list = []

    def _request_to_heap(self, request: Request) -> Any:
        """Convert a request to the appropriate heap element."""
        raise NotImplementedError

    def _heap_to_request(self, element: Any) -> Request:
        """Extract the request from a heap element."""
        raise NotImplementedError

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to heap priority."""
        heapq.heappush(self._heap, self._request_to_heap(request))

    def pop_request(self) -> Request:
        """Pop the highest priority request from the heap."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return self._heap_to_request(heapq.heappop(self._heap))

    def peek_request(self) -> Request:
        """Peek at the highest priority request in the heap without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap_to_request(self._heap[0])

    def prepend_request(self, request: Request) -> None:
        """Add a request to the heap. In heap-based queues there is no beginning as
        elements are ordered by priority/score. This behaves like add_request."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue to the heap. In heap-based queues there
        is no beginning as elements are ordered by priority/score. This behaves like
        add_request."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the heap."""
        self._heap.remove(self._request_to_heap(request))
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the heap."""
        remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [h for h in self._heap if self._heap_to_request(h) not in remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue to heap order."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield self._heap_to_request(heapq.heappop(heap_copy))


class PriorityRequestQueue(RequestHeap):
    """A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where requests with a smaller
    value of `priority` are processed first. If multiple requests have the same
    priority, the one with the earlier `arrival_time` is processed first."""

    def _request_to_heap(self, request: Request) -> Request:
        """For priority queue, the heap element is the request itself."""
        return request

    def _heap_to_request(self, element: Request) -> Request:
        """Extract request from heap element with type checking."""
        return element


class SJFRequestQueue(RequestHeap):
    """A Shortest Job First (SJF) queue where requests are ordered by weighted score.
    Requests with higher weighted scores (shorter jobs) are processed first."""

    def __init__(self):
        super().__init__()
        self.scorer = TimeAndLengthScorer()

    def _request_to_heap(self, request: Request) -> WeightedScoreSorter:
        """Convert request to `WeightedScoreSorter` for heap."""
        return WeightedScoreSorter(request, self.scorer)

    def _heap_to_request(self, element: WeightedScoreSorter) -> Request:
        """Extract request from the `WeightedScoreSorter`."""
        return element.request


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.SJF:
        return SJFRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
