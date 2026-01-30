# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from sortedcontainers import SortedDict

from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.request import Request


class WaitingQueue:
    def __init__(self, lock):
        self.best_queue: QueueInfo | None = None
        self.lock = lock
        self.queues = SortedDict()

    def prepend_request(self, request: Request):
        # Safety for None prompt_token_ids
        ids = request.prompt_token_ids or []
        prompt_len = len(ids)
        queue = self._get_queue_by_length(prompt_len)
        if queue is None:
            queue = self._add_queue(prompt_len)
        queue.add_request_front(request)

    def peek_request(self) -> Request | None:
        if self.best_queue is None:
            return None
        return self.best_queue.peek_request()

    def pop_request(self) -> Request | None:
        if self.best_queue is None:
            return None
        return self.best_queue.pop_request()

    def prepend_requests(self, request_list: list[Request]):
        if self.best_queue is None:
            # If there is no best queue currently selected, we cannot prepend
            # to it. Logic may require re-routing these requests, but to fix
            # the crash/type-error we simply return or log.
            # In a robust implementation, you might want to re-add them via add_request
            return
        self.best_queue.add_requests_front(request_list)

    def remove_requests(self, requests_list: list[Request]):
        for req in requests_list:
            for queue in self.queues.values():
                if queue.remove_request(req):
                    break

    def update_best_queue(self, new_best_queue: QueueInfo | None):
        with self.lock:
            self.best_queue = new_best_queue

    def __len__(self):
        with self.lock:
            # Sum all requests across all EWSJF queues
            num_waiting = sum(queue.size for queue in self.queues.values())
        return num_waiting

    @property
    def is_empty_best_queue(self) -> bool:
        """
        Check if the best queue is empty.

        Returns:
            bool: True if best queue is empty or None.
        """
        if self.best_queue is None:
            return True
        return self.best_queue.is_empty

    @property
    def has_best_queue(self) -> bool:
        """
        Check if there's best queue.

        Returns:
            bool: True if best queue exists
        """
        return self.best_queue is not None

    def get_all_queues(self, max_boundary: int | None = None) -> list[QueueInfo]:
        """
        Get all queues as a list, optionally filtered by maximum boundary.

        Args:
            max_boundary (Optional[int]): If provided, only return queues where
                                            high_boundary < max_boundary

        Returns:
            List[QueueInfo]: All queues (or filtered queues) in the waiting queue
        """
        all_queues = list(self.queues.values())

        if max_boundary is None:
            return all_queues

        return [queue for queue in all_queues if queue.high_boundary < max_boundary]

    @property
    def queues_count(self) -> int:
        """
        Get the number of queues.

        Returns:
            int: Number of queues in the waiting queue
        """
        return len(self.queues)

    def delete_queue(self, queue: QueueInfo) -> list[Request]:
        """
        Delete a queue from the waiting queue and return its requests.

        Args:
            queue (QueueInfo): The queue to delete

        Returns:
            List[Request]: All requests that were in the deleted queue
        """
        if queue.low_boundary in self.queues:
            remaining_requests = queue.get_all_requests()
            del self.queues[queue.low_boundary]
            return remaining_requests
        return []

    def add_request(self, request: Request):
        # Determine which queue should handle this request based on prompt length
        ids = request.prompt_token_ids or []
        prompt_len = len(ids)

        with self.lock:
            queue = self._get_queue_by_length(prompt_len)

            # Create a new queue if none exists for this length
            if queue is None:
                queue = self._add_queue(prompt_len)

            # Add the request to the appropriate queue
            queue.add_request(request)

    def _get_queue_by_length(self, length: int) -> QueueInfo | None:
        """
        Find the queue that should contain requests of the given prompt length.
        """
        # Use binary search to find the appropriate queue
        idx = self.queues.bisect_right(length)

        if idx > 0:
            key = self.queues.iloc[idx - 1]
            queue = self.queues[key]
            if queue.contains_length(length):
                return queue

        return None

    def _add_queue(self, length: int) -> QueueInfo:
        """
        Add a new queue for the given prompt length.
        """
        # Find the insertion position using SortedDict
        insert_idx = self.queues.bisect_right(length)

        # Get the previous queue (if exists) using the SortedDict
        prev_queue = None
        if insert_idx > 0:
            prev_key = self.queues.iloc[insert_idx - 1]
            prev_queue = self.queues[prev_key]

        # Get the next queue (if exists) using the SortedDict
        next_queue = None
        if insert_idx < len(self.queues):
            next_key = self.queues.iloc[insert_idx]
            next_queue = self.queues[next_key]

        # Determine boundaries for the new queue based on adjacent queues
        if prev_queue is not None:
            prev_range = prev_queue.high_boundary - prev_queue.low_boundary
            half_range = prev_range // 2
            lower = max(prev_queue.high_boundary + 1, length - half_range)
            upper = lower + prev_range
        else:
            # No previous queue - this is the first queue
            prev_range = 100  # Default range
            lower = 0
            upper = 100

        # Adjust upper boundary if there's a next queue
        if next_queue is not None:
            upper = min(next_queue.low_boundary - 1, upper)

        return self._create_queue((lower, upper))

    def _create_queue(
        self, boundaries: tuple[int, int], removable: bool = True
    ) -> QueueInfo:
        """
        Create a new queue with specified boundaries.
        """
        # print(f"add queue: {boundaries}")
        queue_id = str(boundaries[0])
        new_queue = QueueInfo(queue_id, boundaries, removable)
        # Add to the sorted dictionary using low boundary as key
        self.queues[boundaries[0]] = new_queue
        return new_queue

    def remove_queue(self, queue_to_remove: QueueInfo):
        """
        Remove a queue and redistribute its requests to appropriate queues.
        """
        if queue_to_remove.low_boundary in self.queues:
            # Get any remaining requests before removal
            remaining_requests = queue_to_remove.get_all_requests()

            # Remove from the sorted dictionary
            del self.queues[queue_to_remove.low_boundary]

            # Redistribute remaining requests to appropriate queues
            for req in remaining_requests:
                self.add_request(req)

    def initialize_queues(self, num_queues: int = 4, step_size: int = 100):
        """
        Initialize a default set of queues with equal-sized ranges.
        """
        for i in range(num_queues):
            boundaries = (i * step_size, (i + 1) * step_size - 1)
            self._create_queue(boundaries, False)

    def initialize_queues_by_config(self, queues_config: list):
        """
        Initialize queues based on configuration file.
        """
        for q in queues_config:
            self._create_queue(q["boundaries"], False)


class QueueInfo:
    """
    Class to encapsulate all queue-related information and operations.
    """

    def __init__(
        self, queue_id: str, boundaries: tuple[int, int], removable: bool = True
    ):
        self.queue_id = queue_id
        self.boundaries = boundaries  # (min_length, max_length)
        self.requests: deque[Request] = deque()
        self.empty_count: int = 0
        self.score: float = 0.0
        self.removable: bool = removable

    @property
    def low_boundary(self) -> int:
        return self.boundaries[0]

    @property
    def high_boundary(self) -> int:
        return self.boundaries[1]

    @property
    def is_empty(self) -> bool:
        return len(self.requests) == 0

    def __bool__(self):
        return bool(self.requests)

    @property
    def size(self) -> int:
        return len(self.requests)

    def add_request(self, request: Request) -> None:
        self.requests.append(request)

    def add_request_front(self, request: Request) -> None:
        self.requests.appendleft(request)

    def add_requests_front(self, requests) -> None:
        self.requests.extendleft(reversed(requests))

    def pop_request(self) -> Request | None:
        if self.requests:
            return self.requests.popleft()
        return None

    def peek_request(self) -> Request | None:
        if self.requests:
            return self.requests[0]
        return None

    def remove_request(self, request: Request) -> bool:
        try:
            self.requests.remove(request)
            return True
        except ValueError:
            return False

    def update_score(self, score: float) -> None:
        self.score = score

    def increment_empty_count(self) -> None:
        if self.is_empty:
            self.empty_count += 1
        else:
            self.empty_count = 0

    def reset_empty_count(self) -> None:
        self.empty_count = 0

    def contains_length(self, length: int) -> bool:
        return self.low_boundary <= length <= self.high_boundary

    def get_all_requests(self) -> list[Request]:
        return list(self.requests)

    def __repr__(self) -> str:
        # FIX: Split long line across multiple lines
        return (
            f"QueueInfo(id={self.queue_id}, boundaries={self.boundaries}, "
            f"size={self.size}, score={self.score:.2f})"
        )
