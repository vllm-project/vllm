# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm._C import ChunkedHashTree
from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    FEATHER = "feather"


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
        """Prepend all requests from another queue to the front of this queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    def free_request(self, request: Request) -> None:  # noqa: B027
        """Free a request that has been executed.

        This can be used by some policies (e.g., chunked hash tree) to
        update the internal state of the queue after a request has been
        executed and is no longer pending.
        """

    def should_add_more_to_batch(self, **kwargs) -> bool:
        """Determine whether more requests should be added to the current batch.

        Based on the policy. This is used by some policies (e.g., chunked hash
        tree) to decide whether to keep adding requests to the current batch or
        to start executing the batch.
        """
        return True

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
        """Prepend all requests from another queue to the front of this queue.

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
        front. Requests are ordered by (priority, arrival_time).
        """
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time).
        """
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


class FeatherRequestQueue(RequestQueue):
    """A request queue that uses a chunked hash tree to prioritize requests.

    Requests are prioritized based on the prefix they share with other
    pending requests. Those sharing more chunks are scheduled first, as
    they can be executed together to save computation and increase prefix
    homogeneity.

    Arxiv link: https://arxiv.org/abs/2605.06046
    """

    CHUNK_SIZE = 50  # Number of tokens per chunk.
    # Maximum chunks lost before a request is considered "homogeneous enough".
    MAXIMUM_CHUNK_LOSS = 5
    # Relaxed ceiling used under high load
    # (requests_waiting >= HIGH_LOAD_THRESHOLD).
    MAXIMUM_CHUNK_LOSS_LENIENT = 7
    # Queue depth at which we switch from MAXIMUM_CHUNK_LOSS
    # to MAXIMUM_CHUNK_LOSS_LENIENT.
    HIGH_LOAD_THRESHOLD = 10
    MIN_BATCH_FLOOR = (
        1  # Starting value of the dynamic minimum batch size each scheduling round.
    )
    # Set to 1 (not 0) so even the very first candidate is evaluated for chunk loss
    # before the floor is raised — giving homogeneity a chance before relaxing.
    # Capped here to guarantee forward progress regardless of prefix structure.
    MAX_BATCH_FLOOR = 10

    def __init__(self):
        self._radix = ChunkedHashTree(self.CHUNK_SIZE)
        self._pending_requests: dict[str, Request] = {}
        self._request_id_to_int: dict[str, int] = {}
        self._int_to_request_id: dict[int, str] = {}
        self._next_int_id = 1
        self._current_batch_floor = (
            self.MIN_BATCH_FLOOR
        )  # Dynamic floor, reset each scheduling round.
        # Grows by 1 each time a request is admitted
        # within MAXIMUM_CHUNK_LOSS, rewarding prefix-
        # homogeneous admissions before relaxing the gate.

    def _intern_id(self, request_id: str) -> int:
        """Convert a request_id to an internal integer ID for the radix tree."""
        int_id = self._request_id_to_int.get(request_id)
        if int_id is None:
            int_id = self._next_int_id
            self._next_int_id += 1
            self._request_id_to_int[request_id] = int_id
            self._int_to_request_id[int_id] = request_id
        return int_id

    def _release_id(self, request_id: str) -> None:
        """Release the internal integer ID for a request_id.

        Called when the request is completed or removed.
        """
        int_id = self._request_id_to_int.pop(request_id, None)
        if int_id is not None:
            self._int_to_request_id.pop(int_id, None)

    def add_request(self, request: Request) -> None:
        int_id = self._intern_id(request.request_id)
        # Insert the request into the radix tree based on its token IDs.
        self._radix.insert(int_id, list(request.all_token_ids))
        self._pending_requests[request.request_id] = request

    def pop_request(self) -> Request:
        int_id, _, _, _ = self._radix.find_best_request()
        if int_id == 0:
            raise IndexError("pop from empty radix queue")
        request_id = self._int_to_request_id[int_id]
        request = self._pending_requests.pop(request_id)
        # Mark as active; affects chunk sharing of subsequent requests.
        self._radix.activate_request(int_id)
        return request

    def peek_request(self) -> Request:
        int_id, _, _, _ = self._radix.find_best_request()
        if int_id == 0:
            raise IndexError("peek from empty radix queue")
        request_id = self._int_to_request_id[int_id]
        return self._pending_requests[request_id]

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        request_id = request.request_id
        if request_id not in self._pending_requests:
            return
        int_id = self._request_id_to_int.get(request_id)
        if int_id is not None:
            # Remove from radix tree to update chunk sharing of subsequent requests.
            self._radix.remove(int_id)
        del self._pending_requests[request_id]
        self._release_id(request_id)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            self.remove_request(request)

    def free_request(self, request: Request) -> None:
        request_id = request.request_id
        int_id = self._request_id_to_int.get(request_id)
        if int_id is not None:
            # Mark as finished to update chunk sharing of subsequent requests.
            self._radix.finish_request(int_id)
        self._release_id(request_id)

    def should_add_more_to_batch(self, current_batch_size: int = 0, **kwargs) -> bool:
        int_id, chunks_before, chunks_after, requests_waiting = (
            self._radix.find_best_request()
        )

        # No candidates at all — nothing to add.
        if int_id == 0:
            return False

        # Always admit the very first request unconditionally.
        if current_batch_size == 0:
            return True

        chunk_loss = chunks_before - chunks_after

        # Determine the active chunk-loss ceiling based on queue pressure.
        # Under high load we allow slightly worse prefix matches to avoid starvation.
        active_ceiling = (
            self.MAXIMUM_CHUNK_LOSS_LENIENT
            if requests_waiting >= self.HIGH_LOAD_THRESHOLD
            else self.MAXIMUM_CHUNK_LOSS
        )

        # A request is "homogeneous enough" if its chunk loss is within the ceiling.
        is_homogeneous = chunk_loss <= active_ceiling

        if is_homogeneous:
            # Reward a homogeneous admission by raising the floor by 1, up to
            # MAX_BATCH_FLOOR. This incrementally makes it harder for a future
            # non-homogeneous request to slip in under the "below the floor"
            # escape hatch — the more homogeneous requests we accumulate, the
            # less we need to relax.
            self._current_batch_floor = min(
                self._current_batch_floor + 1, self.MAX_BATCH_FLOOR
            )
            return True

        # The candidate exceeds the chunk-loss ceiling.
        # Only admit it if we haven't yet reached the dynamic floor —
        # i.e., we still need more requests and can't afford to be selective yet.
        # Reject once at or above the floor to preserve prefix homogeneity.
        return current_batch_size < self._current_batch_floor

    def __bool__(self) -> bool:
        int_id, _, _, _ = self._radix.find_best_request()
        return int_id != 0

    def __len__(self) -> int:
        return len(self._pending_requests)

    def __iter__(self) -> Iterator[Request]:
        return iter(self._pending_requests.values())


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.FEATHER:
        return FeatherRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
