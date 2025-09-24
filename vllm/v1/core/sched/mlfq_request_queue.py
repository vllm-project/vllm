# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Optional

from vllm.v1.request import Request, RequestStatus


class MLFQRequestQueue:
    """
    Multi-Level Feedback Queue implementation for vLLM v1 scheduler.
    
    This implements the skip-join MLFQ algorithm from the FastServe paper,
    with multiple priority levels and dynamic job promotion/demotion.
    """
    
    def __init__(
        self,
        num_levels: int = 6,
        base_quantum: int = 1,
        quantum_multiplier: float = 2.0,
        skip_join_base: int = 128,
        starvation_threshold: int = 100,
        eta: int = 2,
    ):
        """
        Initialize MLFQ with configurable parameters.
        
        Args:
            num_levels: Number of priority levels (default 6)
            base_quantum: Base quantum for level 1 (default 1 iteration)
            quantum_multiplier: Multiplier for quantum at each level (default 2.0)
            skip_join_base: Base for skip-join calculation (default 128 tokens)
            starvation_threshold: Iterations before starvation promotion (default 100)
            eta: Skip levels during demotion (default 2)
        """
        self.num_levels = num_levels
        self.base_quantum = base_quantum
        self.quantum_multiplier = quantum_multiplier
        self.skip_join_base = skip_join_base
        self.starvation_threshold = starvation_threshold
        self.eta = eta
        
        # Create priority queues for each level
        # Level 0 is highest priority (shortest quantum)
        self.queues: list[deque[Request]] = [deque() for _ in range(num_levels)]
        
        # Track job attributes for MLFQ logic
        self.job_attributes: dict[str, MLFQJobAttributes] = {}
        
        # Global iteration counter for starvation tracking
        self.global_iteration = 0
        
    def get_quantum(self, level: int) -> int:
        """Get the quantum (max iterations) for a given level."""
        return int(self.base_quantum * (self.quantum_multiplier ** level))
    
    def get_initial_level(self, request: Request) -> int:
        """
        Determine initial priority level using skip-join heuristic.
        
        Based on input length: shorter inputs get higher priority.
        """
        input_length = request.num_prompt_tokens
        
        # Skip-join formula: floor(log2(input_len / base))
        # This ensures longer inputs start at lower priority levels
        if input_length <= self.skip_join_base:
            return 0  # Highest priority for very short inputs
        else:
            level = int(math.floor(math.log2(input_length / self.skip_join_base)))
            return min(level, self.num_levels - 1)
    
    def add_request(self, request: Request) -> None:
        """Add a request to the appropriate MLFQ level."""
        initial_level = self.get_initial_level(request)
        
        # Create job attributes for MLFQ tracking
        self.job_attributes[request.request_id] = MLFQJobAttributes(
            current_level=initial_level,
            attained_iterations=0,
            starve_counter=0,
            last_scheduled_iteration=self.global_iteration,
        )
        
        # Add to the appropriate queue
        self.queues[initial_level].append(request)
    
    def pop_request(self) -> Request:
        """Pop the highest priority request from MLFQ."""
        # Find the highest priority non-empty queue
        for level in range(self.num_levels):
            if self.queues[level]:
                request = self.queues[level].popleft()
                return request
        
        raise IndexError("pop from empty MLFQ")
    
    def peek_request(self) -> Request:
        """Peek at the highest priority request without removing it."""
        for level in range(self.num_levels):
            if self.queues[level]:
                return self.queues[level][0]
        
        raise IndexError("peek from empty MLFQ")
    
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to its current level queue."""
        if request.request_id in self.job_attributes:
            level = self.job_attributes[request.request_id].current_level
            self.queues[level].appendleft(request)
        else:
            # If no attributes, add normally
            self.add_request(request)
    
    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Prepend multiple requests to their respective level queues."""
        for request in requests:
            self.prepend_request(request)
    
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from MLFQ."""
        if request.request_id in self.job_attributes:
            level = self.job_attributes[request.request_id].current_level
            try:
                self.queues[level].remove(request)
            except ValueError:
                pass  # Request not in queue
            del self.job_attributes[request.request_id]
    
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from MLFQ."""
        for request in requests:
            self.remove_request(request)
    
    def update_after_iteration(self, scheduled_requests: list[Request]) -> None:
        """
        Update MLFQ state after each scheduling iteration.
        
        This implements the core MLFQ logic:
        1. Increment attained iterations for scheduled requests
        2. Demote requests that exceed their quantum
        3. Increment starve counters for waiting requests
        4. Promote starved requests
        """
        self.global_iteration += 1
        
        # Update scheduled requests
        # Collect requests to demote to avoid modifying deque during iteration
        requests_to_demote = []
        for request in scheduled_requests:
            if request.request_id in self.job_attributes:
                attrs = self.job_attributes[request.request_id]
                attrs.attained_iterations += 1
                attrs.last_scheduled_iteration = self.global_iteration
                
                # Check for demotion
                quantum = self.get_quantum(attrs.current_level)
                if attrs.attained_iterations > quantum:
                    requests_to_demote.append((request, attrs))
        
        # Perform demotions after iteration
        for request, attrs in requests_to_demote:
            self._demote_request(request, attrs)
        
        # Update starve counters for all waiting requests
        # Collect requests to promote to avoid modifying deque during iteration
        requests_to_promote = []
        for level in range(self.num_levels):
            for request in self.queues[level]:
                if request.request_id in self.job_attributes:
                    attrs = self.job_attributes[request.request_id]
                    attrs.starve_counter += 1
                    
                    # Check for starvation promotion
                    if attrs.starve_counter >= self.starvation_threshold:
                        requests_to_promote.append((request, attrs))
        
        # Perform promotions after iteration
        for request, attrs in requests_to_promote:
            self._promote_request(request, attrs)
    
    def _demote_request(self, request: Request, attrs: MLFQJobAttributes) -> None:
        """Demote a request to a lower priority level."""
        # Skip levels with eta parameter
        new_level = min(attrs.current_level + self.eta, self.num_levels - 1)
        
        if new_level != attrs.current_level:
            # Remove from current level
            try:
                self.queues[attrs.current_level].remove(request)
            except ValueError:
                pass
            
            # Add to new level
            attrs.current_level = new_level
            attrs.attained_iterations = 0  # Reset attained iterations
            self.queues[new_level].append(request)
    
    def _promote_request(self, request: Request, attrs: MLFQJobAttributes) -> None:
        """Promote a starved request to highest priority."""
        # Remove from current level
        try:
            self.queues[attrs.current_level].remove(request)
        except ValueError:
            pass
        
        # Promote to highest priority
        attrs.current_level = 0
        attrs.attained_iterations = 0
        attrs.starve_counter = 0
        self.queues[0].append(request)
    
    def __bool__(self) -> bool:
        """Check if MLFQ has any requests."""
        return any(self.queues)
    
    def __len__(self) -> int:
        """Get total number of requests in MLFQ."""
        return sum(len(queue) for queue in self.queues)
    
    def __iter__(self) -> Iterator[Request]:
        """Iterate over all requests in priority order."""
        for level in range(self.num_levels):
            for request in self.queues[level]:
                yield request
    
    def __reversed__(self) -> Iterator[Request]:
        """Iterate over all requests in reverse priority order."""
        for level in reversed(range(self.num_levels)):
            for request in reversed(self.queues[level]):
                yield request
    
    def get_level_counts(self) -> list[int]:
        """Get the number of requests in each level."""
        return [len(queue) for queue in self.queues]
    
    def get_job_attributes(self, request_id: str) -> Optional[MLFQJobAttributes]:
        """Get MLFQ attributes for a specific request."""
        return self.job_attributes.get(request_id)


class MLFQJobAttributes:
    """Attributes tracked for each job in MLFQ."""
    
    def __init__(
        self,
        current_level: int,
        attained_iterations: int = 0,
        starve_counter: int = 0,
        last_scheduled_iteration: int = 0,
    ):
        self.current_level = current_level
        self.attained_iterations = attained_iterations
        self.starve_counter = starve_counter
        self.last_scheduled_iteration = last_scheduled_iteration
    
    def __repr__(self) -> str:
        return (f"MLFQJobAttributes(level={self.current_level}, "
                f"attained={self.attained_iterations}, "
                f"starve={self.starve_counter})")
