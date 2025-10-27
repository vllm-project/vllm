from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple, Deque, Union

from vllm.v1.request import Request


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
