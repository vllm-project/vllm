# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MultiModalRegistry
from vllm.v1.core.sched.mlfq_request_queue import MLFQRequestQueue
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


class MLFQScheduler(Scheduler):
    """
    Multi-Level Feedback Queue Scheduler for vLLM v1.
    
    This implements the skip-join MLFQ algorithm from the FastServe paper,
    designed to reduce head-of-line blocking and improve latency for
    interactive LLM applications.
    """
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MultiModalRegistry(),
        include_finished_set: bool = False,
        log_stats: bool = False,
        # MLFQ-specific parameters
        num_levels: Optional[int] = None,
        base_quantum: Optional[int] = None,
        quantum_multiplier: Optional[float] = None,
        skip_join_base: Optional[int] = None,
        starvation_threshold: Optional[int] = None,
        eta: Optional[int] = None,
    ) -> None:
        """
        Initialize MLFQ Scheduler.
        
        Args:
            vllm_config: vLLM configuration
            kv_cache_config: KV cache configuration
            structured_output_manager: Structured output manager
            mm_registry: Multi-modal registry
            include_finished_set: Whether to include finished request set
            log_stats: Whether to log statistics
            num_levels: Number of MLFQ priority levels
            base_quantum: Base quantum for level 1
            quantum_multiplier: Quantum multiplier between levels
            skip_join_base: Base for skip-join calculation
            starvation_threshold: Iterations before starvation promotion
            eta: Skip levels during demotion
        """
        # Initialize parent scheduler
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )
        
        # MLFQ-specific configuration - use config values if not provided
        self.num_levels = num_levels if num_levels is not None else vllm_config.scheduler_config.mlfq_num_levels
        self.base_quantum = base_quantum if base_quantum is not None else vllm_config.scheduler_config.mlfq_base_quantum
        self.quantum_multiplier = quantum_multiplier if quantum_multiplier is not None else vllm_config.scheduler_config.mlfq_quantum_multiplier
        self.skip_join_base = skip_join_base if skip_join_base is not None else vllm_config.scheduler_config.mlfq_skip_join_base
        self.starvation_threshold = starvation_threshold if starvation_threshold is not None else vllm_config.scheduler_config.mlfq_starvation_threshold
        self.eta = eta if eta is not None else vllm_config.scheduler_config.mlfq_eta
        
        # Replace the standard waiting queue with MLFQ
        self.mlfq = MLFQRequestQueue(
            num_levels=self.num_levels,
            base_quantum=self.base_quantum,
            quantum_multiplier=self.quantum_multiplier,
            skip_join_base=self.skip_join_base,
            starvation_threshold=self.starvation_threshold,
            eta=self.eta,
        )
        
        # Keep reference to original waiting queue for compatibility
        self.waiting = self.mlfq
        
        # Track scheduled requests for MLFQ updates
        self.last_scheduled_requests: list[Request] = []
        
        logger.info(
            f"MLFQ Scheduler initialized with {self.num_levels} levels, "
            f"base_quantum={self.base_quantum}, quantum_multiplier={self.quantum_multiplier}, "
            f"skip_join_base={self.skip_join_base}, starvation_threshold={self.starvation_threshold}"
        )
    
    def schedule(self) -> SchedulerOutput:
        """
        Schedule requests using MLFQ algorithm.
        
        This overrides the parent schedule method to use MLFQ logic
        while maintaining compatibility with vLLM's scheduling interface.
        """
        # Clean up any finished requests from MLFQ queue before scheduling
        self._cleanup_finished_requests_from_mlfq()
        
        # Call parent schedule method to get standard scheduling logic
        scheduler_output = super().schedule()
        
        # Update MLFQ state after scheduling
        self._update_mlfq_after_schedule(scheduler_output)
        
        return scheduler_output
    
    def _cleanup_finished_requests_from_mlfq(self) -> None:
        """Remove any finished or invalid requests from MLFQ queue before scheduling."""
        requests_to_remove = set()
        for request in list(self.mlfq):
            # Remove finished requests
            if request.is_finished():
                requests_to_remove.add(request.request_id)
                logger.debug(f"Found finished request {request.request_id} with status {request.status}")
            # Remove requests that are already RUNNING (should not be in waiting queue)
            elif request.status == RequestStatus.RUNNING:
                requests_to_remove.add(request.request_id)
                logger.debug(f"Found RUNNING request {request.request_id} in MLFQ queue - removing")
        
        if requests_to_remove:
            logger.debug(f"Cleaning up {len(requests_to_remove)} requests from MLFQ")
            # Remove requests from MLFQ queue
            for req_id in requests_to_remove:
                # Get the request object from MLFQ queue directly
                request = None
                for level in range(self.mlfq.num_levels):
                    for req in self.mlfq.queues[level]:
                        if req.request_id == req_id:
                            request = req
                            break
                    if request:
                        break
                
                if request:
                    logger.debug(f"Processing request {req_id}, current status: {request.status}")
                    # Check if request is still in MLFQ before removing
                    if req_id in self.mlfq.job_attributes:
                        level = self.mlfq.job_attributes[req_id].current_level
                        logger.debug(f"Request {req_id} is in MLFQ level {level}")
                        if request in self.mlfq.queues[level]:
                            self.mlfq.queues[level].remove(request)
                            logger.debug(f"Removed request {req_id} from MLFQ level {level}")
                        else:
                            logger.debug(f"Request {req_id} not found in MLFQ level {level} queue")
                        del self.mlfq.job_attributes[req_id]
                    else:
                        logger.debug(f"Request {req_id} not found in MLFQ job attributes")
                else:
                    logger.debug(f"Request {req_id} not found in MLFQ queues")
    
    def _update_mlfq_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Update MLFQ state after scheduling step."""
        # Collect all scheduled requests
        scheduled_requests = []
        
        # Get requests from scheduler output
        for req_data in scheduler_output.scheduled_new_reqs:
            if req_data.req_id in self.requests:
                scheduled_requests.append(self.requests[req_data.req_id])
        
        # Handle cached requests (CachedRequestData object)
        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id in self.requests:
                    scheduled_requests.append(self.requests[req_id])
        
        # Update MLFQ state
        self.mlfq.update_after_iteration(scheduled_requests)
        self.last_scheduled_requests = scheduled_requests
        
        # Log MLFQ statistics if enabled
        if self.log_stats:
            self._log_mlfq_stats()
    
    def _log_mlfq_stats(self) -> None:
        """Log MLFQ statistics for monitoring."""
        level_counts = self.mlfq.get_level_counts()
        total_waiting = sum(level_counts)
        
        if total_waiting > 0:
            logger.debug(
                f"MLFQ Stats - Level counts: {level_counts}, "
                f"Total waiting: {total_waiting}, "
                f"Global iteration: {self.mlfq.global_iteration}"
            )
    
    def add_request(self, request: Request) -> None:
        """Add a new request to MLFQ."""
        # Add to MLFQ instead of standard waiting queue
        self.mlfq.add_request(request)
        self.requests[request.request_id] = request
        
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)
        
        # Log MLFQ-specific information
        if request.request_id in self.mlfq.job_attributes:
            attrs = self.mlfq.job_attributes[request.request_id]
            logger.debug(
                f"Added request {request.request_id} to MLFQ level {attrs.current_level} "
                f"(input_len={request.num_prompt_tokens})"
            )
    
    def finish_requests(
        self,
        request_ids: str | list[str],
        finished_status: RequestStatus,
    ) -> None:
        """Finish requests and clean up MLFQ state."""
        # Call parent method for standard cleanup
        super().finish_requests(request_ids, finished_status)
        
        # Clean up MLFQ attributes
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        
        for req_id in request_ids:
            if req_id in self.mlfq.job_attributes:
                del self.mlfq.job_attributes[req_id]
    
    def get_mlfq_stats(self) -> dict[str, Any]:
        """Get MLFQ-specific statistics."""
        level_counts = self.mlfq.get_level_counts()
        total_waiting = sum(level_counts)
        
        # Calculate average starve counter
        total_starve = 0
        starve_count = 0
        for attrs in self.mlfq.job_attributes.values():
            total_starve += attrs.starve_counter
            starve_count += 1
        
        avg_starve = total_starve / starve_count if starve_count > 0 else 0
        
        return {
            "level_counts": level_counts,
            "total_waiting": total_waiting,
            "global_iteration": self.mlfq.global_iteration,
            "average_starve_counter": avg_starve,
            "num_jobs_with_attributes": len(self.mlfq.job_attributes),
        }
    
    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
    ) -> Optional[Any]:
        """Create scheduler stats including MLFQ information."""
        stats = super().make_stats(spec_decoding_stats)
        
        if stats and self.log_stats:
            mlfq_stats = self.get_mlfq_stats()
            logger.debug(f"MLFQ Stats: {mlfq_stats}")
        
        return stats
    
    def get_request_counts(self) -> tuple[int, int]:
        """Get running and waiting request counts."""
        num_running = len(self.running)
        num_waiting = len(self.mlfq)
        return num_running, num_waiting
    
    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests."""
        return len(self.running) > 0 or len(self.mlfq) > 0
    
    def get_num_unfinished_requests(self) -> int:
        """Get total number of unfinished requests."""
        return len(self.running) + len(self.mlfq)
    
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """
        Update scheduler state from model runner output.
        
        This overrides the parent method to properly handle finished requests
        in the MLFQ queue.
        """
        # Call parent method to get the standard behavior
        engine_core_outputs = super().update_from_output(scheduler_output, model_runner_output)
        
        # Additional cleanup for MLFQ: remove any finished requests from MLFQ queue
        # that might still be in the queue due to status changes
        finished_request_ids = set()
        for request in list(self.mlfq):
            if request.is_finished():
                finished_request_ids.add(request.request_id)
        
        if finished_request_ids:
            # Remove finished requests from MLFQ queue
            for req_id in finished_request_ids:
                if req_id in self.requests:
                    request = self.requests[req_id]
                    self.mlfq.remove_request(request)
                    # Clean up MLFQ attributes
                    if req_id in self.mlfq.job_attributes:
                        del self.mlfq.job_attributes[req_id]
        
        return engine_core_outputs