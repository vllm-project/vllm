import pytest
from vllm.v1.metrics.stats import SlidingWindowStats, FinishedRequestStats


def test_sliding_window_stats_initialization():
    stats = SlidingWindowStats(window_size=100)
    assert stats.window_size == 100
    assert stats.get_metric("latency_ms") == 0.0
    assert stats.get_metric("throughput_tokens_per_sec") == 0.0
    assert stats.get_metric_count("latency_ms") == 0


def test_sliding_window_stats_single_request():
    stats = SlidingWindowStats(window_size=100)
    
    # Create a sample finished request
    finished_req = FinishedRequestStats(
        finish_reason=None,
        e2e_latency=1.5,  # 1.5 seconds
        num_prompt_tokens=10,
        num_generation_tokens=20,
        max_tokens_param=30,
        queued_time=0.2,
        prefill_time=0.3,
        inference_time=1.0,
        decode_time=0.7
    )
    
    stats.add_finished_request(finished_req)
    
    # Check metrics
    assert stats.get_metric("latency_ms") == 1500.0  # 1.5s -> 1500ms
    assert stats.get_metric("prompt_tokens") == 10
    assert stats.get_metric("generation_tokens") == 20
    assert stats.get_metric("queued_time_ms") == 200.0
    assert stats.get_metric("prefill_time_ms") == 300.0
    assert stats.get_metric("decode_time_ms") == 700.0
    
    # Check throughput calculation
    total_tokens = finished_req.num_prompt_tokens + finished_req.num_generation_tokens
    total_time = finished_req.prefill_time + finished_req.decode_time
    expected_throughput = total_tokens / total_time
    assert stats.get_metric("throughput_tokens_per_sec") == expected_throughput


def test_sliding_window_stats_multiple_requests():
    stats = SlidingWindowStats(window_size=3)
    
    # Add three requests with different latencies
    for latency in [1.0, 2.0, 3.0]:
        finished_req = FinishedRequestStats(
            finish_reason=None,
            e2e_latency=latency,
            num_prompt_tokens=10,
            num_generation_tokens=20,
            max_tokens_param=30,
            queued_time=0.1,
            prefill_time=0.2,
            inference_time=0.5,
            decode_time=0.3
        )
        stats.add_finished_request(finished_req)
    
    # Check that we have exactly 3 samples
    assert stats.get_metric_count("latency_ms") == 3
    
    # Average latency should be (1.0 + 2.0 + 3.0) / 3 * 1000
    assert stats.get_metric("latency_ms") == 2000.0
    
    # Add one more request - should push out the first one
    finished_req = FinishedRequestStats(
        finish_reason=None,
        e2e_latency=4.0,
        num_prompt_tokens=10,
        num_generation_tokens=20,
        max_tokens_param=30,
        queued_time=0.1,
        prefill_time=0.2,
        inference_time=0.5,
        decode_time=0.3
    )
    stats.add_finished_request(finished_req)
    
    # Should still have 3 samples
    assert stats.get_metric_count("latency_ms") == 3
    
    # Average latency should now be (2.0 + 3.0 + 4.0) / 3 * 1000
    assert stats.get_metric("latency_ms") == 3000.0


def test_sliding_window_stats_empty_window():
    stats = SlidingWindowStats(window_size=100)
    
    # All metrics should return 0 when no requests have been added
    assert stats.get_metric("latency_ms") == 0.0
    assert stats.get_metric("throughput_tokens_per_sec") == 0.0
    assert stats.get_metric("prompt_tokens") == 0.0
    assert stats.get_metric("generation_tokens") == 0.0
    assert stats.get_metric("queued_time_ms") == 0.0
    assert stats.get_metric("prefill_time_ms") == 0.0
    assert stats.get_metric("decode_time_ms") == 0.0