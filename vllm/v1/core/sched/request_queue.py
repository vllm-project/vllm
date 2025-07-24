# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""
    FCFS = "fcfs"
    PRIORITY = "priority"
    SHORTEST_PREFILL_FIRST = "shortest_prefill_first"


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
    def prepend_requests(self, requests: RequestQueue) -> None:
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

    @abstractmethod
    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
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
        queue."""
        self.extendleft(reversed(requests))

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [
            req for req in self if req not in requests_to_remove
        ]
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

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        return super().__reversed__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

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

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, _, request = self._heap[0]
        return request

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
        self._heap = [(p, t, r) for p, t, r in self._heap if r != request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        self._heap = [(p, t, r) for p, t, r in self._heap
                      if r not in requests_to_remove]
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
            _, _, request = heapq.heappop(heap_copy)
            yield request

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse priority order."""
        return reversed(list(self))


class ShortestPrefillFirstRequestQueue(RequestQueue):
    """
    A queue that pops the request with least number of cache miss token first.

    TODO: add a fairness parameter to avoid request starvation.
    """

    def __init__(self, kv_cache_manager: KVCacheManager) -> None:
        """
        Initialize shortest prefill first queue.

        It requires the KV cache manager to estimate the number of cache miss
        tokens.

        Args:
            kv_cache_manager: The KV cache manager to use for estimating the
                number of cache miss tokens.
        """
        self._requests: list[Request] = []
        self._kv_cache_manager = kv_cache_manager
        assert self._kv_cache_manager is not None

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        self._requests.append(request)

    def get_num_cache_miss_tokens(self, request: Request) -> int:
        """Get the number of cache miss tokens of one request.
        We use number of cache miss tokens as the proxy for the prefill time.
        Empirically it has >98% pearson correlation with the prefill time
        and is enough for scheduling.
        """
        num_computed_tokens = \
            self._kv_cache_manager.get_num_computed_tokens(request)
        num_cache_miss_tokens = request.num_tokens - num_computed_tokens
        return num_cache_miss_tokens

    def peek_request(self) -> Request:
        """Peek the request with least number of cache miss tokens."""
        if not self._requests:
            raise IndexError("peek / pop from empty queue")

        min_num_cache_miss_tokens = float('inf')
        min_request: Request | None = None

        for request in self._requests:
            num_cache_miss_tokens = self.get_num_cache_miss_tokens(request)
            if num_cache_miss_tokens < min_num_cache_miss_tokens:
                min_num_cache_miss_tokens = num_cache_miss_tokens
                min_request = request

        if min_request is None:
            raise IndexError("peek / pop from empty queue")

        return min_request

    def pop_request(self) -> Request:
        """Pop the request with least number of cache miss tokens."""
        request = self.peek_request()
        self._requests.remove(request)
        return request

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the queue."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to this queue."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._requests.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        self._requests = [
            req for req in self._requests if req not in requests_to_remove
        ]

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._requests)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._requests)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to shortest prefill first policy."""
        copy_requests = [(self.get_num_cache_miss_tokens(req), req)
                         for req in self._requests]
        heapq.heapify(copy_requests)
        while copy_requests:
            _, request = heapq.heappop(copy_requests)
            yield request

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        return reversed(self._requests)


def create_request_queue(policy: SchedulingPolicy,
                         kv_cache_manager: KVCacheManager) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.SHORTEST_PREFILL_FIRST:
        return ShortestPrefillFirstRequestQueue(kv_cache_manager)
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
