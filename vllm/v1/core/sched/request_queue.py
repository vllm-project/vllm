# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from enum import Enum

from vllm.logger import init_logger

from vllm.v1.request import Request


logger = init_logger(__name__)

class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    RESIDUAL_SJF = "residual_sjf"


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


class ResidualSJFRequestQueue(RequestQueue):
    """A dynamic queue ordered by local prefill work and bounded aging."""

    def __init__(
        self,
        residual_cost_fn: Callable[[Request], int],
        max_wait_ms: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._requests: list[Request] = []
        self._residual_cost_fn = residual_cost_fn
        self._max_wait_seconds = max_wait_ms / 1000
        self._clock = clock
        self._selected_request: Request | None = None

    @staticmethod
    def _is_recovery_request(request: Request) -> bool:
        return request.num_preemptions > 0 or request.num_computed_tokens > 0

    def _selection_key(
        self, request: Request, now: float
    ) -> tuple[int, int, float, str]:
        if self._is_recovery_request(request):
            return 0, 0, request.arrival_time, request.request_id
        if now - request.arrival_time >= self._max_wait_seconds:
            return 1, 0, request.arrival_time, request.request_id
        return (
            2,
            self._residual_cost_fn(request),
            request.arrival_time,
            request.request_id,
        )

    def get_best_request(
        self, now: float | None = None
    ) -> tuple[Request, tuple[int, int, float, str]]:
        """Return the best current request without changing queue state."""
        if not self._requests:
            raise IndexError("peek from an empty queue")
        now = self._clock() if now is None else now
        return min(
            (
                (request, self._selection_key(request, now))
                for request in self._requests
            ),
            key=lambda item: item[1],
        )

    def pin_request(self, request: Request) -> None:
        """Pin a previously selected request until it is popped or removed."""
        if request not in self._requests:
            raise ValueError("Cannot pin a request that is not in the queue")
        self._selected_request = request

    def _clear_selection(self) -> None:
        self._selected_request = None

    def add_request(self, request: Request) -> None:
        self._clear_selection()
        self._requests.append(request)

    def pop_request(self) -> Request:
        if not self._requests:
            raise IndexError("pop from an empty queue")
        request = self._selected_request
        if request is None:
            request, _ = self.get_best_request()
        self._requests.remove(request)
        self._clear_selection()
        return request

    def peek_request(self) -> Request:
        if self._selected_request is None:
            request, _ = self.get_best_request()
            self._selected_request = request
        return self._selected_request

    def prepend_request(self, request: Request) -> None:
        self._clear_selection()
        self._requests.insert(0, request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        self._clear_selection()
        for request in requests:
            self._requests.insert(0, request)

    def remove_request(self, request: Request) -> None:
        self._requests.remove(request)
        if self._selected_request is request:
            self._clear_selection()

    def remove_requests(self, requests: Iterable[Request]) -> None:
        requests_to_remove = set(requests)
        self._requests = [
            request for request in self._requests if request not in requests_to_remove
        ]
        if self._selected_request in requests_to_remove:
            self._clear_selection()

    def __bool__(self) -> bool:
        return bool(self._requests)

    def __len__(self) -> int:
        return len(self._requests)

    def __iter__(self) -> Iterator[Request]:
        now = self._clock()
        return iter(
            sorted(
                self._requests, key=lambda request: self._selection_key(request, now)
            )
        )


def create_request_queue(
    policy: SchedulingPolicy,
    residual_cost_fn: Callable[[Request], int] | None = None,
    residual_sjf_max_wait_ms: int = 10_000,
) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.RESIDUAL_SJF:
        if residual_cost_fn is None:
            raise ValueError("residual_sjf requires a residual cost function")
        logger.warning_once(
            "residual_sjf is enabled with residual_sjf_max_wait_ms=%d. "
            "This value controls the mean/tail tradeoff: smaller values "
            "behave closer to FCFS (flatter tail), larger values improve "
            "mean latency at the cost of P95/P99. Start at 1/2 to 1/3 of "
            "the target P95 wait and tune after load testing.",
            residual_sjf_max_wait_ms,
        )
        return ResidualSJFRequestQueue(
            residual_cost_fn, residual_sjf_max_wait_ms
        )
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
