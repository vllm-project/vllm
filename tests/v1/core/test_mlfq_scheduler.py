# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from unittest.mock import Mock

from vllm.config import (CacheConfig, ModelConfig, SchedulerConfig, VllmConfig)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.mlfq_request_queue import MLFQRequestQueue, MLFQJobAttributes
from vllm.v1.core.sched.mlfq_scheduler import MLFQScheduler
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .utils import EOS_TOKEN_ID, create_requests


def create_mlfq_scheduler(
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    num_blocks: int = 10000,
    block_size: int = 16,
    max_model_len: int = 8192,
    # MLFQ-specific parameters
    mlfq_num_levels: int = 6,
    mlfq_base_quantum: int = 1,
    mlfq_quantum_multiplier: float = 2.0,
    mlfq_skip_join_base: int = 128,
    mlfq_starvation_threshold: int = 100,
    mlfq_eta: int = 2,
) -> MLFQScheduler:
    """Create MLFQ scheduler for testing."""
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        policy="mlfq",
        # MLFQ parameters
        mlfq_num_levels=mlfq_num_levels,
        mlfq_base_quantum=mlfq_base_quantum,
        mlfq_quantum_multiplier=mlfq_quantum_multiplier,
        mlfq_skip_join_base=mlfq_skip_join_base,
        mlfq_starvation_threshold=mlfq_starvation_threshold,
        mlfq_eta=mlfq_eta,
    )
    
    model_config = ModelConfig(
        model="/home/lifd/models/opt-125m/facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
    )
    
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(['layer'],
                             FullAttentionSpec(block_size, 1, 1, torch.float32,
                                               False))
        ],
    )
    
    cache_config.num_gpu_blocks = num_blocks
    
    return MLFQScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def create_test_request(
    request_id: str,
    num_prompt_tokens: int,
    max_tokens: int = 16,
) -> Request:
    """Create a test request with specified parameters."""
    sampling_params = SamplingParams(max_tokens=max_tokens)
    prompt_token_ids = [0] * num_prompt_tokens
    
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        multi_modal_kwargs=None,
        multi_modal_placeholders=None,
        multi_modal_hashes=None,
        eos_token_id=EOS_TOKEN_ID,
        block_hasher=lambda x: hash(str(x)),
    )


class TestMLFQRequestQueue:
    """Test cases for MLFQRequestQueue."""
    
    def test_initialization(self):
        """Test MLFQ queue initialization."""
        queue = MLFQRequestQueue(
            num_levels=4,
            base_quantum=2,
            quantum_multiplier=3.0,
            skip_join_base=64,
            starvation_threshold=50,
            eta=1,
        )
        
        assert queue.num_levels == 4
        assert queue.base_quantum == 2
        assert queue.quantum_multiplier == 3.0
        assert queue.skip_join_base == 64
        assert queue.starvation_threshold == 50
        assert queue.eta == 1
        assert len(queue.queues) == 4
        assert queue.global_iteration == 0
        assert len(queue.job_attributes) == 0
    
    def test_get_quantum(self):
        """Test quantum calculation for different levels."""
        queue = MLFQRequestQueue(
            num_levels=4,
            base_quantum=2,
            quantum_multiplier=3.0,
        )
        
        # Level 0: 2 * (3.0^0) = 2
        assert queue.get_quantum(0) == 2
        # Level 1: 2 * (3.0^1) = 6
        assert queue.get_quantum(1) == 6
        # Level 2: 2 * (3.0^2) = 18
        assert queue.get_quantum(2) == 18
        # Level 3: 2 * (3.0^3) = 54
        assert queue.get_quantum(3) == 54
    
    def test_get_initial_level_skip_join(self):
        """Test skip-join initial level calculation."""
        queue = MLFQRequestQueue(
            num_levels=6,
            skip_join_base=128,
        )
        
        # Very short input (<= base) -> level 0
        request = create_test_request("req1", 64)
        assert queue.get_initial_level(request) == 0
        
        # Short input (base < input <= 2*base) -> level 0
        request = create_test_request("req2", 200)
        assert queue.get_initial_level(request) == 0
        
        # Medium input (2*base < input <= 4*base) -> level 1
        request = create_test_request("req3", 300)
        assert queue.get_initial_level(request) == 1
        
        # Long input (4*base < input <= 8*base) -> level 2
        request = create_test_request("req4", 600)
        assert queue.get_initial_level(request) == 2
        
        # Very long input -> calculate actual level
        request = create_test_request("req5", 2000)
        level = queue.get_initial_level(request)
        # 2000/128 = 15.625, log2(15.625) ≈ 3.97, floor = 3
        # Should be capped at max level (5) if > 5
        expected_level = min(3, queue.num_levels - 1)
        assert level == expected_level
    
    def test_add_request(self):
        """Test adding requests to MLFQ queue."""
        queue = MLFQRequestQueue(num_levels=4, skip_join_base=100)
        
        # Add requests with different input lengths
        short_req = create_test_request("short", 50)  # Should go to level 0
        long_req = create_test_request("long", 500)   # Should go to level 2
        
        queue.add_request(short_req)
        queue.add_request(long_req)
        
        # Check that requests are in correct levels
        assert len(queue.queues[0]) == 1
        assert len(queue.queues[2]) == 1
        assert len(queue.queues[1]) == 0
        assert len(queue.queues[3]) == 0
        
        # Check job attributes
        assert "short" in queue.job_attributes
        assert "long" in queue.job_attributes
        
        short_attrs = queue.job_attributes["short"]
        assert short_attrs.current_level == 0
        assert short_attrs.attained_iterations == 0
        assert short_attrs.starve_counter == 0
    
    def test_pop_request(self):
        """Test popping requests from MLFQ queue."""
        queue = MLFQRequestQueue(num_levels=4, skip_join_base=100)
        
        # Add requests to different levels
        req1 = create_test_request("req1", 50)   # Level 0
        req2 = create_test_request("req2", 200)  # Level 1
        req3 = create_test_request("req3", 500)  # Level 2
        
        queue.add_request(req1)
        queue.add_request(req2)
        queue.add_request(req3)
        
        # Pop should return highest priority (level 0)
        popped = queue.pop_request()
        assert popped.request_id == "req1"
        assert len(queue.queues[0]) == 0
        
        # Next pop should return level 1
        popped = queue.pop_request()
        assert popped.request_id == "req2"
        assert len(queue.queues[1]) == 0
        
        # Next pop should return level 2
        popped = queue.pop_request()
        assert popped.request_id == "req3"
        assert len(queue.queues[2]) == 0
    
    def test_pop_empty_queue(self):
        """Test popping from empty queue raises error."""
        queue = MLFQRequestQueue()
        
        with pytest.raises(IndexError, match="pop from empty MLFQ"):
            queue.pop_request()
    
    def test_peek_request(self):
        """Test peeking at highest priority request."""
        queue = MLFQRequestQueue(num_levels=4, skip_join_base=100)
        
        req1 = create_test_request("req1", 50)   # Level 0
        req2 = create_test_request("req2", 200)  # Level 1
        
        queue.add_request(req1)
        queue.add_request(req2)
        
        # Peek should return highest priority without removing
        peeked = queue.peek_request()
        assert peeked.request_id == "req1"
        assert len(queue.queues[0]) == 1  # Still there
    
    def test_remove_request(self):
        """Test removing specific requests."""
        queue = MLFQRequestQueue(num_levels=4, skip_join_base=100)
        
        req1 = create_test_request("req1", 50)
        req2 = create_test_request("req2", 200)
        
        queue.add_request(req1)
        queue.add_request(req2)
        
        # Remove req1
        queue.remove_request(req1)
        
        assert "req1" not in queue.job_attributes
        assert len(queue.queues[0]) == 0
        assert "req2" in queue.job_attributes
        assert len(queue.queues[1]) == 1
    
    def test_update_after_iteration_demotion(self):
        """Test demotion after exceeding quantum."""
        queue = MLFQRequestQueue(
            num_levels=4,
            base_quantum=1,
            quantum_multiplier=2.0,
            eta=1,
        )
        
        req = create_test_request("req1", 50)
        queue.add_request(req)
        
        # Initially at level 0 with quantum = 1
        attrs = queue.job_attributes["req1"]
        assert attrs.current_level == 0
        assert attrs.attained_iterations == 0
        
        # Schedule for 1 iteration (within quantum)
        queue.update_after_iteration([req])
        assert attrs.attained_iterations == 1
        assert attrs.current_level == 0  # Still at level 0
        
        # Schedule for another iteration (exceeds quantum)
        queue.update_after_iteration([req])
        assert attrs.attained_iterations == 0  # Reset after demotion
        assert attrs.current_level == 1  # Demoted to level 1
    
    def test_update_after_iteration_starvation_promotion(self):
        """Test starvation promotion."""
        queue = MLFQRequestQueue(
            num_levels=4,
            starvation_threshold=3,
        )
        
        req = create_test_request("req1", 500)  # Calculate actual level
        queue.add_request(req)
        
        attrs = queue.job_attributes["req1"]
        # 500/128 = 3.9, log2(3.9) ≈ 1.96, floor = 1
        expected_level = 1
        assert attrs.current_level == expected_level
        assert attrs.starve_counter == 0
        
        # Wait for 3 iterations (starvation threshold)
        for _ in range(3):
            queue.update_after_iteration([])
        
        # After 3 iterations, request should be promoted
        assert attrs.current_level == 0  # Promoted to highest priority
        assert attrs.attained_iterations == 0  # Reset
        assert attrs.starve_counter == 0  # Reset after promotion
    
    def test_level_counts(self):
        """Test getting level counts."""
        queue = MLFQRequestQueue(num_levels=4, skip_join_base=100)
        
        # Add requests to different levels
        queue.add_request(create_test_request("req1", 50))   # Level 0
        queue.add_request(create_test_request("req2", 200))  # Level 1
        queue.add_request(create_test_request("req3", 200))  # Level 1
        queue.add_request(create_test_request("req4", 500))  # Level 2
        
        counts = queue.get_level_counts()
        assert counts == [1, 2, 1, 0]
    
    def test_len_and_bool(self):
        """Test length and boolean conversion."""
        queue = MLFQRequestQueue()
        
        assert len(queue) == 0
        assert not queue
        
        queue.add_request(create_test_request("req1", 50))
        assert len(queue) == 1
        assert queue
        
        queue.add_request(create_test_request("req2", 200))
        assert len(queue) == 2
    
    def test_iteration(self):
        """Test iterating over all requests."""
        queue = MLFQRequestQueue(num_levels=4, skip_join_base=100)
        
        req1 = create_test_request("req1", 50)   # Level 0
        req2 = create_test_request("req2", 200)  # Level 1
        req3 = create_test_request("req3", 500)  # Level 2
        
        queue.add_request(req1)
        queue.add_request(req2)
        queue.add_request(req3)
        
        # Should iterate in priority order
        requests = list(queue)
        assert len(requests) == 3
        assert requests[0].request_id == "req1"  # Highest priority first
        assert requests[1].request_id == "req2"
        assert requests[2].request_id == "req3"


class TestMLFQScheduler:
    """Test cases for MLFQScheduler."""
    
    def test_initialization(self):
        """Test MLFQ scheduler initialization."""
        scheduler = create_mlfq_scheduler(
            mlfq_num_levels=4,
            mlfq_base_quantum=2,
            mlfq_quantum_multiplier=3.0,
        )
        
        assert isinstance(scheduler.mlfq, MLFQRequestQueue)
        assert scheduler.mlfq.num_levels == 4
        assert scheduler.mlfq.base_quantum == 2
        assert scheduler.mlfq.quantum_multiplier == 3.0
        assert scheduler.waiting is scheduler.mlfq  # Reference to MLFQ
    
    def test_add_request(self):
        """Test adding requests to MLFQ scheduler."""
        scheduler = create_mlfq_scheduler()
        
        req1 = create_test_request("req1", 50)
        req2 = create_test_request("req2", 200)
        
        scheduler.add_request(req1)
        scheduler.add_request(req2)
        
        assert "req1" in scheduler.requests
        assert "req2" in scheduler.requests
        assert len(scheduler.mlfq) == 2
        
        # Check that requests are in appropriate levels
        assert "req1" in scheduler.mlfq.job_attributes
        assert "req2" in scheduler.mlfq.job_attributes
    
    def test_schedule_basic(self):
        """Test basic scheduling functionality."""
        scheduler = create_mlfq_scheduler(max_num_seqs=2)
        
        # Add requests with different priorities
        req1 = create_test_request("req1", 50)   # High priority (short)
        req2 = create_test_request("req2", 500)  # Low priority (long)
        
        scheduler.add_request(req1)
        scheduler.add_request(req2)
        
        # Schedule should prioritize short request
        output = scheduler.schedule()
        
        # Short request should be scheduled first
        scheduled_ids = [req.req_id for req in output.scheduled_new_reqs]
        if scheduled_ids:  # If any requests were scheduled
            assert "req1" in scheduled_ids
    
    def test_finish_requests(self):
        """Test finishing requests and cleanup."""
        scheduler = create_mlfq_scheduler()
        
        req1 = create_test_request("req1", 50)
        req2 = create_test_request("req2", 200)
        
        scheduler.add_request(req1)
        scheduler.add_request(req2)
        
        # Finish req1
        scheduler.finish_requests("req1", RequestStatus.FINISHED_ABORTED)
        
        assert "req1" not in scheduler.requests
        assert "req1" not in scheduler.mlfq.job_attributes
        assert "req2" in scheduler.requests
        assert "req2" in scheduler.mlfq.job_attributes
    
    def test_get_mlfq_stats(self):
        """Test MLFQ statistics."""
        scheduler = create_mlfq_scheduler()
        
        # Add some requests
        scheduler.add_request(create_test_request("req1", 50))
        scheduler.add_request(create_test_request("req2", 200))
        
        stats = scheduler.get_mlfq_stats()
        
        assert "level_counts" in stats
        assert "total_waiting" in stats
        assert "global_iteration" in stats
        assert "average_starve_counter" in stats
        assert "num_jobs_with_attributes" in stats
        
        assert stats["total_waiting"] == 2
        assert stats["num_jobs_with_attributes"] == 2
    
    def test_get_request_counts(self):
        """Test getting request counts."""
        scheduler = create_mlfq_scheduler()
        
        # Initially no requests
        running, waiting = scheduler.get_request_counts()
        assert running == 0
        assert waiting == 0
        
        # Add requests
        scheduler.add_request(create_test_request("req1", 50))
        scheduler.add_request(create_test_request("req2", 200))
        
        running, waiting = scheduler.get_request_counts()
        assert running == 0
        assert waiting == 2
    
    def test_has_unfinished_requests(self):
        """Test checking for unfinished requests."""
        scheduler = create_mlfq_scheduler()
        
        assert not scheduler.has_unfinished_requests()
        
        scheduler.add_request(create_test_request("req1", 50))
        assert scheduler.has_unfinished_requests()
        
        scheduler.finish_requests("req1", RequestStatus.FINISHED_ABORTED)
        assert not scheduler.has_unfinished_requests()
    
    def test_get_num_unfinished_requests(self):
        """Test getting number of unfinished requests."""
        scheduler = create_mlfq_scheduler()
        
        assert scheduler.get_num_unfinished_requests() == 0
        
        scheduler.add_request(create_test_request("req1", 50))
        scheduler.add_request(create_test_request("req2", 200))
        
        assert scheduler.get_num_unfinished_requests() == 2
    
    def test_cleanup_finished_requests(self):
        """Test cleanup of finished requests from MLFQ queue."""
        scheduler = create_mlfq_scheduler()
        
        req1 = create_test_request("req1", 50)
        req2 = create_test_request("req2", 200)
        
        scheduler.add_request(req1)
        scheduler.add_request(req2)
        
        # Manually set one request as finished (simulating completion)
        req1.status = RequestStatus.FINISHED_ABORTED
        
        # Schedule should clean up finished requests
        scheduler.schedule()
        
        # req1 should be removed from MLFQ
        assert "req1" not in scheduler.mlfq.job_attributes
        assert "req2" in scheduler.mlfq.job_attributes
    
    def test_mlfq_parameter_validation(self):
        """Test MLFQ parameter validation."""
        # Test with custom MLFQ parameters
        scheduler = create_mlfq_scheduler(
            mlfq_num_levels=3,
            mlfq_base_quantum=2,
            mlfq_quantum_multiplier=1.5,
            mlfq_skip_join_base=64,
            mlfq_starvation_threshold=50,
            mlfq_eta=1,
        )
        
        assert scheduler.mlfq.num_levels == 3
        assert scheduler.mlfq.base_quantum == 2
        assert scheduler.mlfq.quantum_multiplier == 1.5
        assert scheduler.mlfq.skip_join_base == 64
        assert scheduler.mlfq.starvation_threshold == 50
        assert scheduler.mlfq.eta == 1


class TestMLFQIntegration:
    """Integration tests for MLFQ scheduler."""
    
    def test_mlfq_vs_fcfs_priority(self):
        """Test that MLFQ gives priority to short requests over FCFS."""
        # Create MLFQ scheduler
        mlfq_scheduler = create_mlfq_scheduler(max_num_seqs=1)
        
        # Add requests: long first, then short
        long_req = create_test_request("long", 500)  # Low priority
        short_req = create_test_request("short", 50)  # High priority
        
        mlfq_scheduler.add_request(long_req)
        mlfq_scheduler.add_request(short_req)
        
        # Schedule - short request should be prioritized
        output = mlfq_scheduler.schedule()
        
        # In MLFQ, short request should be scheduled first despite arriving later
        if output.scheduled_new_reqs:
            scheduled_id = output.scheduled_new_reqs[0].req_id
            # Short request should be scheduled first due to higher priority
            assert scheduled_id == "short"
    
    def test_demotion_behavior(self):
        """Test that requests are demoted after exceeding quantum."""
        scheduler = create_mlfq_scheduler(
            mlfq_base_quantum=1,
            mlfq_quantum_multiplier=2.0,
            mlfq_eta=1,
        )
        
        req = create_test_request("req1", 50)  # Starts at level 0
        scheduler.add_request(req)
        
        attrs = scheduler.mlfq.job_attributes["req1"]
        initial_level = attrs.current_level
        
        # Simulate scheduling the request multiple times
        for _ in range(3):
            output = scheduler.schedule()
            # Update MLFQ state after scheduling
            scheduled_requests = []
            for req_data in output.scheduled_new_reqs:
                if req_data.req_id in scheduler.requests:
                    scheduled_requests.append(scheduler.requests[req_data.req_id])
            scheduler.mlfq.update_after_iteration(scheduled_requests)
        
        # Request should be demoted after exceeding quantum
        assert attrs.current_level > initial_level
    
    def test_starvation_prevention(self):
        """Test that long-waiting requests are promoted."""
        scheduler = create_mlfq_scheduler(
            mlfq_starvation_threshold=3,
        )
        
        req = create_test_request("req1", 500)  # Starts at low priority
        scheduler.add_request(req)
        
        attrs = scheduler.mlfq.job_attributes["req1"]
        initial_level = attrs.current_level
        
        # Simulate waiting (no scheduling) - exactly 3 iterations to trigger promotion
        for _ in range(3):
            scheduler.mlfq.update_after_iteration([])
        
        # Request should be promoted to highest priority
        assert attrs.current_level == 0  # Highest priority
        # Starve counter should be reset to 0 after promotion
        assert attrs.starve_counter == 0
    
    def test_multiple_requests_scheduling(self):
        """Test scheduling multiple requests with different priorities."""
        scheduler = create_mlfq_scheduler(max_num_seqs=3)
        
        # Add requests with different input lengths
        req1 = create_test_request("req1", 50)   # Highest priority
        req2 = create_test_request("req2", 200)  # Medium priority
        req3 = create_test_request("req3", 500)  # Lowest priority
        
        scheduler.add_request(req1)
        scheduler.add_request(req2)
        scheduler.add_request(req3)
        
        # Schedule should prioritize by input length
        output = scheduler.schedule()
        
        if len(output.scheduled_new_reqs) > 0:
            # Shortest request should be scheduled first
            first_scheduled = output.scheduled_new_reqs[0].req_id
            assert first_scheduled == "req1"


if __name__ == "__main__":
    pytest.main([__file__])
