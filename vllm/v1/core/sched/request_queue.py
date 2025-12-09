# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.core.sched.policy.weighted_score_sorter import WeightedScoreSorter
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


class HeapBasedRequestQueue(RequestQueue, ABC):
    """Base class for heap-based request queues (priority and SJF)."""

    def __init__(self) -> None:
        self._heap: list = []

    @abstractmethod
    def _to_heap_element(self, request: Request) -> object:
        """Convert a request to the appropriate heap element."""
        pass

    @abstractmethod
    def _from_heap_element(self, heap_element: object) -> Request:
        """Extract the request from a heap element."""
        pass

    def add_request(self, request: Request) -> None:
        """Add a request to the heap queue."""
        heap_element = self._to_heap_element(request)
        heapq.heappush(self._heap, heap_element)

    def pop_request(self) -> Request:
        """Pop the highest priority request from the heap."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        heap_element = heapq.heappop(self._heap)
        return self._from_heap_element(heap_element)

    def peek_request(self) -> Request:
        """Peek at the highest priority request without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._from_heap_element(self._heap[0])

    def prepend_request(self, request: Request) -> None:
        """
        Add request to the queue. In heap-based queues, "prepend" has no
        special meaning as elements are ordered by priority/score. This
        behaves like add_request.
        """
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """
        Add all requests from another queue. In heap-based queues,
        "prepend" has no special meaning as elements are ordered by
        priority/score. This behaves like adding all requests.
        """
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the heap."""
        try:
            self._heap.remove(request)
            heapq.heapify(self._heap)
        except ValueError as err:
            raise ValueError(
                f"Request {request.request_id} not found in queue"
            ) from err

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the heap."""
        requests_to_remove = (
            set(requests) if not isinstance(requests, set) else requests
        )
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over requests in priority/score order."""
        heap_copy = self._heap.copy()
        while heap_copy:
            heap_element = heapq.heappop(heap_copy)
            yield self._from_heap_element(heap_element)


class PriorityRequestQueue(HeapBasedRequestQueue):
    """
    A priority queue where requests are ordered by (priority, arrival_time).
    Lower priority values and earlier arrival times are processed first.
    """

    def _to_heap_element(self, request: Request) -> Request:
        """For priority queue, the heap element is the request itself."""
        return request

    def _from_heap_element(self, heap_element: object) -> Request:
        """Extract request from heap element with type checking."""
        assert isinstance(heap_element, Request)
        return heap_element


class SJFRequestQueue(HeapBasedRequestQueue):
    """
    A Shortest Job First (SJF) queue where requests are ordered by weighted score.
    Requests with higher weighted scores (shorter jobs) are processed first.
    """

    def _to_heap_element(self, request: Request) -> tuple[WeightedScoreSorter, Request]:
        """Convert request to (weighted_score, request) tuple for heap."""
        assert request.prompt_token_ids is not None
        return (
            WeightedScoreSorter(len(request.prompt_token_ids), request.arrival_time),
            request,
        )

    def _from_heap_element(self, heap_element: object) -> Request:
        """Extract request from the (score, request) tuple with type checking."""
        assert isinstance(heap_element, tuple) and len(heap_element) == 2
        _, request = heap_element
        return request

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the SJF heap."""
        original_length = len(self._heap)
        self._heap = [(ws, r) for (ws, r) in self._heap if r != request]
        if len(self._heap) == original_length:
            raise ValueError(f"Request {request.request_id} not found in SJF queue")
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the SJF heap."""
        requests_to_remove = set(requests)
        self._heap = [(ws, r) for (ws, r) in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)


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
