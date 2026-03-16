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

    def prepend_requests(self, requests: RequestQueue) -> None:
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


class LongShortRequestQueue(RequestQueue):
    """
    A FCFS queue that supports skipping long requests when the limit is reached.
    
    This queue maintains FCFS order but allows skipping long requests when
    the global limit of running long requests is reached. Short requests can
    still be scheduled even if long requests are blocked.
    
    Args:
        long_request_threshold: Token threshold to classify a request as long.
            Requests with prefill_tokens >= threshold are considered long.
        max_long_requests: Maximum number of long requests that can run
            simultaneously.
        get_running_long_count: Callable that returns the current count of
            running long requests. This is called during pop_request() to
            check if more long requests can be scheduled.
    """

    def __init__(
        self,
        long_request_threshold: int,
        max_long_requests: int,
    ) -> None:
        if long_request_threshold <= 0:
            raise ValueError("long_request_threshold must be positive")
        if max_long_requests < 0:
            raise ValueError("max_long_requests must be non-negative")

        self._queue: deque[Request] = deque()
        self.long_request_threshold = long_request_threshold
        self.max_long_requests = max_long_requests
        self.running_long_count = 0
        self.has_slot_for_long_request = True

    def is_long_request(self, request: Request) -> bool:
        """Check if a request is a long request based on token threshold."""
        num_prefill_tokens = request.num_tokens - request.num_output_tokens
        return num_prefill_tokens >= self.long_request_threshold

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self._queue.append(request)

    def pop_request(self) -> Request:
        """
        Pop a request from the queue, skipping long requests if limit is reached.
        
        Searches from _search_index to find the first schedulable request.
        Long requests are skipped if max_long_requests limit is reached.
        """
        if not self._queue:
            raise IndexError("pop from empty queue")

        queue_len = len(self._queue)
        # Search from _search_index, wrapping around if needed
        for i in range(queue_len):
            idx = i % queue_len
            request = self._queue[idx]

            if self.is_long_request(request):
                # Check if we can schedule more long requests
                print(f"running_long_count: {self.running_long_count}, max_long_requests: {self.max_long_requests}, has_slot_for_long_request: {self.has_slot_for_long_request}", flush=True)
                if self.running_long_count >= self.max_long_requests or not self.has_slot_for_long_request:
                    # Skip this long request, continue searching
                    continue
                # Can schedule this long request
                return self._pop_at_index(idx)
            else:
                # Short request, can always schedule
                return self._pop_at_index(idx)

        # No schedulable request found (all are long requests and limit reached)
        raise IndexError("no schedulable request (all long requests blocked), and it is not reachable")

    def _pop_at_index(self, index: int) -> Request:
        """
        Pop request at the given index using rotate for efficiency.
        
        This is more efficient than remove() which is O(n).
        """
        if index == 0:
            # Already at front, just pop
            request = self._queue.popleft()
            return request

        # Rotate to bring target to front
        self._queue.rotate(-index)
        request = self._queue.popleft()
        # Rotate back to restore relative order
        self._queue.rotate(index - 1)
        return request

    def peek_request(self) -> Request:
        """
        Peek at the next schedulable request without removing it.
        
        Returns the first request that can be scheduled (respecting long
        request limits), without modifying the queue.
        """
        if not self._queue:
            raise IndexError("peek from empty queue")

        queue_len = len(self._queue)
        # Search from _search_index
        for i in range(queue_len):
            idx = i % queue_len
            request = self._queue[idx]

            if self.is_long_request(request):
                # Check if we can schedule more long requests
                if self.running_long_count >= self.max_long_requests or not self.has_slot_for_long_request:
                    # Skip this long request, continue searching
                    continue
                # Can schedule this long request
                return request
            else:
                # Short request, can always schedule
                return request

        # No schedulable request found
        return None

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self._queue.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this queue."""
        # Convert to list in reverse order for prepending
        requests_list = list(requests)
        for request in reversed(requests_list):
            self._queue.appendleft(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        try:
            self._queue.remove(request)
        except ValueError:
            raise ValueError(f"Request {request.request_id} not found in queue")

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self._queue if req not in requests_to_remove]
        self._queue.clear()
        self._queue.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self._queue) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._queue)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue in FCFS order."""
        return iter(self._queue)

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse FCFS order."""
        return reversed(self._queue)

    def __repr__(self) -> str:
        return (f"LongShortRequestQueue(long_request_threshold={self.long_request_threshold}, "
                + f"max_long_requests={self.max_long_requests}, "
                + f"running_long_count={self.running_long_count}, "
                + f"has_slot_for_long_request={self.has_slot_for_long_request})")


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
