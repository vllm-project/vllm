# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

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


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.

    Uses lazy deletion: ``remove_request`` and ``remove_requests`` mark
    entries as removed instead of rebuilding the heap, reducing the cost
    of removal from O(n + heapify) to O(1) per request. Stale entries
    are transparently skipped during ``pop_request`` / ``peek_request``
    and periodically compacted when they accumulate.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []
        self._active_ids: set[int] = set()

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)
        self._active_ids.add(id(request))

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        while self._heap:
            request = heapq.heappop(self._heap)
            rid = id(request)
            if rid in self._active_ids:
                self._active_ids.remove(rid)
                return request
        raise IndexError("pop from empty heap")

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        while self._heap:
            if id(self._heap[0]) in self._active_ids:
                return self._heap[0]
            heapq.heappop(self._heap)
        raise IndexError("peek from empty heap")

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue.

        Uses lazy deletion: the entry remains in the heap but is skipped
        during pop/peek, making this O(1) instead of O(n + heapify).
        """
        self._active_ids.discard(id(request))
        self._maybe_cleanup()

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue.

        Uses lazy deletion to avoid the O(n) heapify cost of the eager
        approach.  Stale entries are compacted lazily when they
        accumulate beyond a threshold.
        """
        for request in requests:
            self._active_ids.discard(id(request))
        self._maybe_cleanup()

    def _maybe_cleanup(self) -> None:
        """Compact the heap when removed entries accumulate."""
        # Rebuild the heap when stale entries dominate.
        if len(self._heap) > 2 * len(self._active_ids):
            self._heap = [r for r in self._heap if id(r) in self._active_ids]
            heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._active_ids)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._active_ids)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        for request in sorted(self._heap):
            if id(request) in self._active_ids:
                yield request


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
