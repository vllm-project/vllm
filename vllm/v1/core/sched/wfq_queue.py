# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Weighted Fair Queuing (WFQ) scheduler for LLM serving.

This module implements a fairness-aware scheduling policy based on virtual
time. Requests are assigned weights (default 1.0) and scheduled according to
their virtual finish time, ensuring proportional resource allocation.

The WFQ algorithm provides:
1. Fairness: Resources allocated proportionally to request weights
2. Starvation prevention: All requests eventually make progress
3. Flexibility: Per-request weight configuration via API
"""

import heapq
from collections.abc import Iterable, Iterator

from vllm.logger import init_logger
from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.request import Request

logger = init_logger(__name__)


class WFQRequestQueue(RequestQueue):
    """Weighted Fair Queuing scheduler for LLM requests.

    Implements virtual time-based fairness using a min-heap ordered by
    virtual_finish_time. Requests with higher weights receive proportionally
    more resources.

    Virtual time mechanics:
    - Each request has a weight (default 1.0, configurable via API)
    - virtual_start = max(global_virtual_time, request_arrival_time)
    - virtual_finish = virtual_start + (tokens_needed / weight)
    - Requests scheduled in order of virtual_finish_time (min-heap)

    This ensures:
    1. Higher-weight requests get scheduled earlier
    2. No request is starved indefinitely
    3. Fair allocation under contention

    Args:
        default_weight: Default weight for requests without explicit weight.
            Must be positive. Defaults to 1.0.

    Attributes:
        _heap: Min-heap of requests ordered by virtual_finish_time
        _virtual_time: Global virtual time (monotonically increasing)
        _default_weight: Default weight for requests
    """

    def __init__(self, default_weight: float = 1.0) -> None:
        """Initialize WFQ scheduler.

        Args:
            default_weight: Default weight for requests. Must be positive.

        Raises:
            ValueError: If default_weight is not positive.
        """
        if default_weight <= 0.0:
            raise ValueError(f"default_weight must be positive, got {default_weight}")

        self._heap: list[Request] = []
        self._virtual_time: float = 0.0
        self._default_weight = default_weight

    def add_request(self, request: Request) -> None:
        """Add request and compute virtual start/finish times.

        Computes virtual times based on WFQ algorithm:
        1. Validate and set weight (use default if invalid)
        2. Compute virtual_start_time = max(global_virtual_time, arrival_time)
        3. Estimate tokens_needed from request
        4. Compute virtual_finish_time = virtual_start + tokens_needed / weight
        5. Insert into min-heap ordered by virtual_finish_time

        Args:
            request: Request to add to queue.
        """
        # Validate and set weight
        if not hasattr(request, "weight") or request.weight <= 0.0:
            logger.debug(
                "Request %s has invalid weight %s, using default %s",
                request.request_id,
                getattr(request, "weight", None),
                self._default_weight,
            )
            request.weight = self._default_weight

        # Compute virtual start time
        # Use max of current virtual time and request arrival time
        request.virtual_start_time = max(
            self._virtual_time,
            request.arrival_time,
        )

        # Estimate total tokens needed for this request
        tokens_needed = self._estimate_tokens_needed(request)

        # Compute virtual finish time
        # Higher weight → smaller virtual_finish → scheduled earlier
        request.virtual_finish_time = request.virtual_start_time + (
            tokens_needed / request.weight
        )

        # Insert into heap (min-heap by virtual_finish_time)
        # Request.__lt__() handles comparison
        heapq.heappush(self._heap, request)

        logger.debug(
            "Added request %s: weight=%.2f, tokens=%d, "
            "virtual_start=%.2f, virtual_finish=%.2f",
            request.request_id,
            request.weight,
            tokens_needed,
            request.virtual_start_time,
            request.virtual_finish_time,
        )

    def pop_request(self) -> Request:
        """Pop request with smallest virtual_finish_time.

        Updates global virtual time to the popped request's virtual_start_time,
        ensuring virtual time monotonicity.

        Returns:
            Request with smallest virtual_finish_time.

        Raises:
            IndexError: If queue is empty.
        """
        if not self._heap:
            raise IndexError("pop from empty heap")

        # Pop request with smallest virtual_finish_time
        request = heapq.heappop(self._heap)

        # Advance global virtual time
        # Ensures virtual time is monotonically increasing
        self._virtual_time = max(
            self._virtual_time,
            request.virtual_start_time,
        )

        logger.debug(
            "Popped request %s: virtual_finish=%.2f, global_virtual_time=%.2f",
            request.request_id,
            request.virtual_finish_time,
            self._virtual_time,
        )

        return request

    def peek_request(self) -> Request:
        """Peek at next request without removing it.

        Returns:
            Request with smallest virtual_finish_time.

        Raises:
            IndexError: If queue is empty.
        """
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend request to queue (e.g., after preemption).

        For WFQ, we preserve the request's existing virtual times to maintain
        fairness. If the request was preempted, it should resume with the same
        virtual_finish_time it had before.

        If virtual times are not set (first-time scheduling), calls add_request
        to compute them.

        Args:
            request: Request to prepend.
        """
        # Check if request has been scheduled before (has virtual times)
        if (
            hasattr(request, "virtual_finish_time")
            and request.virtual_finish_time > 0.0
        ):
            # Resuming after preemption - preserve virtual times
            logger.debug(
                "Prepending request %s (preserving virtual_finish=%.2f)",
                request.request_id,
                request.virtual_finish_time,
            )
            heapq.heappush(self._heap, request)
        else:
            # First time scheduling - compute virtual times
            logger.debug(
                "Prepending request %s (computing virtual times)",
                request.request_id,
            )
            self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend multiple requests to queue.

        Args:
            requests: RequestQueue or iterable of requests to prepend.
        """
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove specific request from queue.

        Args:
            request: Request to remove.

        Raises:
            ValueError: If request is not in queue.
        """
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple requests from queue.

        Args:
            requests: Iterable of requests to remove.
        """
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests.

        Returns:
            True if queue is non-empty, False otherwise.
        """
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue.

        Returns:
            Number of requests currently in queue.
        """
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over queue in virtual_finish_time order.

        Yields requests in order of virtual_finish_time (earliest first).
        Does not modify the queue.

        Yields:
            Requests in virtual_finish_time order.
        """
        # Create copy of heap to avoid modifying original
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over queue in reverse virtual_finish_time order.

        Returns:
            Iterator over requests in reverse order.
        """
        return reversed(list(self))

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _estimate_tokens_needed(self, request: Request) -> int:
        """Estimate total tokens needed for request completion.

        Computes the number of tokens remaining to be processed for this
        request, including:
        - Uncomputed prompt tokens
        - Output tokens to be generated

        Args:
            request: Request to estimate tokens for.

        Returns:
            Estimated total tokens needed (guaranteed >= 1).
        """
        # Prompt tokens remaining
        prompt_tokens_remaining = (
            request.num_prompt_tokens - request.num_computed_tokens
        )

        # Output tokens remaining
        output_tokens_remaining = request.max_tokens - request.num_output_tokens

        # Total tokens needed
        tokens_needed = prompt_tokens_remaining + output_tokens_remaining

        # Ensure at least 1 token (avoid division by zero)
        return max(1, tokens_needed)
