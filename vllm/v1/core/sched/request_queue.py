# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.metrics.iteration_workload import QueuedWorkloadTally
from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"


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
    def prepend_requests(self, requests: Iterable[Request]) -> None:
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

    def prepend_requests(self, requests: Iterable[Request]) -> None:
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


class PriorityRequestQueue(RequestQueue):
    """A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


class TalliedRequestQueue(RequestQueue):
    """A queue that reports the requests entering and leaving it to a tally."""

    def __init__(self, queue: RequestQueue, tally: QueuedWorkloadTally) -> None:
        self._queue = queue
        self._tally = tally

    def add_request(self, request: Request) -> None:
        self._queue.add_request(request)
        self._tally.add(self, request)

    def pop_request(self) -> Request:
        request = self._queue.pop_request()
        self._tally.discard(self, request)
        return request

    def peek_request(self) -> Request:
        return self._queue.peek_request()

    def prepend_request(self, request: Request) -> None:
        self._queue.prepend_request(request)
        self._tally.add(self, request)

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        requests = list(requests)
        self._queue.prepend_requests(requests)
        for request in requests:
            self._tally.add(self, request)

    def remove_request(self, request: Request) -> None:
        self._queue.remove_request(request)
        self._tally.discard(self, request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        requests = set(requests)
        self._queue.remove_requests(requests)
        for request in requests:
            self._tally.discard(self, request)

    def __bool__(self) -> bool:
        return bool(self._queue)

    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self) -> Iterator[Request]:
        return iter(self._queue)


def create_request_queue(
    policy: SchedulingPolicy, tally: QueuedWorkloadTally | None = None
) -> RequestQueue:
    """Create request queue based on scheduling policy.

    With a tally, the queue reports the requests entering and leaving it.
    """
    queue: RequestQueue
    if policy == SchedulingPolicy.PRIORITY:
        queue = PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        queue = FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
    return queue if tally is None else TalliedRequestQueue(queue, tally)
