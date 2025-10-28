from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple, Deque, Union

from vllm.v1.request import Request

from typing import Optional, Tuple

from sortedcontainers import SortedDict


class WaitingQueue:
    def __init__(self, lock):
        self.best_queue = None
        self.lock = lock
        self.queues = SortedDict()

    def prepend_request(self, request):
        prompt_len = len(request.prompt_token_ids)
        queue = self._get_queue_by_length(prompt_len)
        if queue is None:
            queue = self._add_queue(prompt_len)
        queue.add_request_front(request)

    def peek_request(self, ):
        return self.best_queue.peek_request()

    def pop_request(self):
        return self.best_queue.pop_request()

    def prepend_requests(self, request_list):
        self.best_queue.add_requests_front(request_list)

    def remove_requests(self, requests_list):
        for req in requests_list:
            found = False
            for queue in self.queues.values():
                if queue.remove_request(req):
                    found = True
                    break
            # if not found:
            #     logger.warning(f"Request {req.request_id} not found in any waiting queue.")

    def __len__(self):
        with self.lock:
            # Sum all requests across all EWSJF queues
            num_waiting = sum(queue.size for queue in self.queues.values())
        return num_waiting

    def add_request(self, request):
        # Determine which queue should handle this request based on prompt length
        prompt_len = len(request.prompt_token_ids)

        with self.lock:
            queue = self._get_queue_by_length(prompt_len)

            # Create a new queue if none exists for this length
            if queue is None:
                queue = self._add_queue(prompt_len)

            # Add the request to the appropriate queue
            queue.add_request(request)

    def _get_queue_by_length(self, length: int) -> Optional[QueueInfo]:
        """
        Find the queue that should contain requests of the given prompt length.

        Uses binary search on the sorted queue boundaries for O(log n) complexity.

        Args:
            length (int): The prompt length to find a queue for

        Returns:
            Optional[QueueInfo]: The queue that handles this length, or None if not found
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

        This method dynamically creates a new queue with appropriate boundaries
        based on existing queues and the specific length requirement.

        Args:
            length (int): The prompt length that triggered queue creation

        Returns:
            QueueInfo: The newly created queue
        """
        # Find the insertion position using SortedDict
        # bisect_right finds the position where length should be inserted
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

        # Ensure minimum range
        if upper - lower < prev_range:
            half_range = prev_range // 2
            lower = max(0, length - half_range)
            upper = lower + prev_range

        return self._create_queue((lower, upper))

    def _create_queue(self, boundaries: Tuple[int, int], removable: bool = True) -> QueueInfo:
        """
        Create a new queue with specified boundaries.

        Args:
            boundaries (Tuple[int, int]): (min_length, max_length) for the queue
            removable (bool, optional): Whether queue can be auto-removed. Defaults to True.

        Returns:
            QueueInfo: The newly created queue
        """
        print(f'add queue: {boundaries}')
        queue_id = str(boundaries[0])
        new_queue = QueueInfo(queue_id, boundaries, removable)
        # Add to the sorted dictionary using low boundary as key
        self.queues[boundaries[0]] = new_queue
        return new_queue

    def remove_queue(self, queue_to_remove: QueueInfo):
        """
        Remove a queue and redistribute its requests to appropriate queues.

        Args:
            queue_to_remove (QueueInfo): The queue to be removed
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

        Args:
            num_queues (int, optional): Number of initial queues to create. Defaults to 4.
        """
        for i in range(num_queues):
            boundaries = (i * step_size, (i + 1) * step_size - 1)
            self._create_queue(boundaries, False)

    def initialize_queues_by_config(self, queues_config: list):
        """
        Initialize queues based on configuration file.

        Args:
            queues_config (list): List of queue configuration dictionaries,
                                each containing 'boundaries' key
        """
        for q in queues_config:
            self._create_queue(q['boundaries'], False)


class QueueInfo:
    """
    Class to encapsulate all queue-related information and operations.

    This class manages individual queues in the EWSJF scheduler, handling request storage,
    queue boundaries, scoring, and various queue operations like adding/removing requests.
    """

    def __init__(self, queue_id: str, boundaries: Tuple[int, int], removable: bool = True):
        """
        Initialize a new queue with specified boundaries and properties.

        Args:
            queue_id (str): Unique identifier for this queue
            boundaries (Tuple[int, int]): (min_length, max_length) defining the range
                                        of prompt lengths this queue handles
            removable (bool, optional): Whether this queue can be automatically removed
                                      when empty. Defaults to True.
        """
        self.queue_id = queue_id
        self.boundaries = boundaries  # (min_length, max_length)
        self.requests: Deque[Request] = deque()
        self.empty_count: int = 0
        self.score: float = 0.0
        self.removable: bool = removable

    @property
    def low_boundary(self) -> int:
        """
        Get the lower boundary of the queue.

        Returns:
            int: Minimum prompt length this queue accepts
        """
        return self.boundaries[0]

    @property
    def high_boundary(self) -> int:
        """
        Get the upper boundary of the queue.

        Returns:
            int: Maximum prompt length this queue accepts
        """
        return self.boundaries[1]

    @property
    def is_empty(self) -> bool:
        """
        Check if the queue is empty.

        Returns:
            bool: True if queue has no requests, False otherwise
        """
        return len(self.requests) == 0

    @property
    def size(self) -> int:
        """
        Get the number of requests in the queue.

        Returns:
            int: Current number of requests in the queue
        """
        return len(self.requests)

    def add_request(self, request: Request) -> None:
        """
        Add a request to the end of the queue.

        Args:
            request (Request): The request object to add to the queue
        """
        self.requests.append(request)

    def add_request_front(self, request: Request) -> None:
        """
        Add a request to the front of the queue (for preempted requests).

        This method is used when a running request gets preempted and needs
        to be prioritized for rescheduling.

        Args:
            request (Request): The preempted request to add to the front
        """
        self.requests.appendleft(request)

    def add_requests_front(self, requests) -> None:
        self.requests.extendleft(reversed(requests))

    def pop_request(self) -> Optional[Request]:
        """
        Remove and return the first request in the queue.

        Returns:
            Optional[Request]: The first request in the queue, or None if empty
        """
        if self.requests:
            return self.requests.popleft()
        return None

    def peek_request(self) -> Optional[Request]:
        """
        Get the first request without removing it.

        Returns:
            Optional[Request]: The first request in the queue, or None if empty
        """
        if self.requests:
            return self.requests[0]
        return None

    def remove_request(self, request: Request) -> bool:
        """
        Remove a specific request from the queue.

        Args:
            request (Request): The request to remove from the queue

        Returns:
            bool: True if the request was found and removed, False otherwise
        """
        try:
            self.requests.remove(request)
            return True
        except ValueError:
            return False

    def update_score(self, score: float) -> None:
        """
        Update the queue's EWSJF score.

        Args:
            score (float): The new score for this queue
        """
        self.score = score

    def increment_empty_count(self) -> None:
        """
        Increment the empty count when queue is empty.

        This is used to track how long a queue has been empty for automatic
        queue removal decisions.
        """
        if self.is_empty:
            self.empty_count += 1
        else:
            self.empty_count = 0

    def reset_empty_count(self) -> None:
        """
        Reset the empty count to zero.

        Called when the queue receives new requests.
        """
        self.empty_count = 0

    def contains_length(self, length: int) -> bool:
        """
        Check if a given prompt length falls within this queue's boundaries.

        Args:
            length (int): The prompt length to check

        Returns:
            bool: True if length is within [low_boundary, high_boundary], False otherwise
        """
        return self.low_boundary <= length <= self.high_boundary

    def get_all_requests(self) -> List[Request]:
        """
        Get all requests in the queue as a list.

        Returns:
            List[Request]: All requests currently in the queue
        """
        return list(self.requests)

    def __repr__(self) -> str:
        """
        String representation of the queue for debugging.

        Returns:
            str: Formatted string showing queue details
        """
        return f"QueueInfo(id={self.queue_id}, boundaries={self.boundaries}, size={self.size}, score={self.score:.2f})"