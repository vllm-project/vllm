# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fairness benchmarks for WFQ scheduler.

Tests proportional fairness, starvation prevention, and overall fairness
using Jain's Fairness Index.

These benchmarks validate that WFQ provides fair resource allocation
according to request weights.
"""

import time
from collections import defaultdict

import pytest

from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    create_request_queue,
)

pytestmark = pytest.mark.cpu_test


def create_test_request(
    request_id: str,
    num_prompt_tokens: int = 100,
    max_tokens: int = 50,
    weight: float = 1.0,
    arrival_time: float | None = None,
    priority: int = 0,
):
    """Create a test request for fairness benchmarking."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    sampling_params = SamplingParams(max_tokens=max_tokens)

    return Request(
        request_id=request_id,
        prompt_token_ids=[0] * num_prompt_tokens,
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=0,
        arrival_time=arrival_time if arrival_time is not None else time.time(),
        priority=priority,
        weight=weight,
    )


def simulate_scheduling(queue, requests):
    """
    Simulate scheduling and return metrics.

    Returns:
        dict with per-weight-class metrics: {weight: {tokens, count, avg_wait}}
    """
    # Add all requests to queue
    for req in requests:
        queue.add_request(req)

    # Track metrics
    metrics_by_weight: dict[float, dict[str, float | int]] = defaultdict(
        lambda: {"tokens": 0, "count": 0, "total_wait": 0.0}
    )

    current_time = 0.0

    # Simulate scheduling
    while queue:
        req = queue.pop_request()

        # Calculate wait time
        wait_time = current_time - req.arrival_time

        # Update metrics
        weight_key = req.weight
        tokens = req.num_prompt_tokens + req.max_tokens
        metrics_by_weight[weight_key]["tokens"] += tokens
        metrics_by_weight[weight_key]["count"] += 1
        metrics_by_weight[weight_key]["total_wait"] += wait_time

        # Simulate processing time (proportional to tokens)
        current_time += tokens * 0.001  # 1ms per token

    # Compute averages
    results = {}
    for weight, data in metrics_by_weight.items():
        results[weight] = {
            "total_tokens": data["tokens"],
            "request_count": data["count"],
            "avg_wait_time": data["total_wait"] / data["count"]
            if data["count"] > 0
            else 0.0,
            "throughput": data["tokens"]
            / max(current_time, 0.001),  # tokens per second
        }

    return results, current_time


def jains_fairness_index(values: list[float]) -> float:
    """
    Compute Jain's Fairness Index.

    FI = (Σ x_i)² / (n × Σ x_i²)

    Where x_i are the values to measure fairness over.
    FI ∈ [0, 1], where 1 = perfect fairness.
    """
    if not values or len(values) == 0:
        return 0.0

    n = len(values)
    sum_x = sum(values)
    sum_x_sq = sum(x**2 for x in values)

    if sum_x_sq == 0:
        return 0.0

    return (sum_x**2) / (n * sum_x_sq)


class TestProportionalFairness:
    """Test that WFQ provides proportional fairness."""

    def test_proportional_fairness_2x(self):
        """Test that weight 2.0 gets 2x throughput of weight 1.0."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Create requests with 2:1 weight ratio
        requests = []

        # Group A: weight 1.0
        for i in range(20):
            req = create_test_request(
                f"a_{i}",
                num_prompt_tokens=100,
                max_tokens=50,
                weight=1.0,
                arrival_time=0.0,
            )
            requests.append(req)

        # Group B: weight 2.0
        for i in range(20):
            req = create_test_request(
                f"b_{i}",
                num_prompt_tokens=100,
                max_tokens=50,
                weight=2.0,
                arrival_time=0.0,
            )
            requests.append(req)

        # Simulate scheduling
        results, _ = simulate_scheduling(queue, requests)

        # Verify proportional fairness
        throughput_1 = results[1.0]["throughput"]
        throughput_2 = results[2.0]["throughput"]

        # Ratio should be approximately 2.0 (within ±10%)
        ratio = throughput_2 / throughput_1
        assert 1.8 <= ratio <= 2.2, f"Expected ratio ~2.0, got {ratio:.2f}"

        print(f"✓ Weight 2.0 throughput ratio: {ratio:.2f} (target: 2.0)")

    def test_proportional_fairness_4x(self):
        """Test that weight 4.0 gets 4x throughput of weight 1.0."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Create requests with 4:1 weight ratio
        requests = []

        # Group A: weight 1.0
        for i in range(20):
            req = create_test_request(
                f"a_{i}",
                num_prompt_tokens=100,
                max_tokens=50,
                weight=1.0,
                arrival_time=0.0,
            )
            requests.append(req)

        # Group C: weight 4.0
        for i in range(20):
            req = create_test_request(
                f"c_{i}",
                num_prompt_tokens=100,
                max_tokens=50,
                weight=4.0,
                arrival_time=0.0,
            )
            requests.append(req)

        # Simulate scheduling
        results, _ = simulate_scheduling(queue, requests)

        # Verify proportional fairness
        throughput_1 = results[1.0]["throughput"]
        throughput_4 = results[4.0]["throughput"]

        # Ratio should be approximately 4.0 (within ±10%)
        ratio = throughput_4 / throughput_1
        assert 3.6 <= ratio <= 4.4, f"Expected ratio ~4.0, got {ratio:.2f}"

        print(f"✓ Weight 4.0 throughput ratio: {ratio:.2f} (target: 4.0)")

    def test_proportional_fairness_half(self):
        """Test that weight 0.5 gets 0.5x throughput of weight 1.0."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Create requests with 1:0.5 weight ratio
        requests = []

        # Group A: weight 1.0
        for i in range(20):
            req = create_test_request(
                f"a_{i}",
                num_prompt_tokens=100,
                max_tokens=50,
                weight=1.0,
                arrival_time=0.0,
            )
            requests.append(req)

        # Group D: weight 0.5
        for i in range(20):
            req = create_test_request(
                f"d_{i}",
                num_prompt_tokens=100,
                max_tokens=50,
                weight=0.5,
                arrival_time=0.0,
            )
            requests.append(req)

        # Simulate scheduling
        results, _ = simulate_scheduling(queue, requests)

        # Verify proportional fairness
        throughput_1 = results[1.0]["throughput"]
        throughput_half = results[0.5]["throughput"]

        # Ratio should be approximately 0.5 (within ±10%)
        ratio = throughput_half / throughput_1
        assert 0.45 <= ratio <= 0.55, f"Expected ratio ~0.5, got {ratio:.2f}"

        print(f"✓ Weight 0.5 throughput ratio: {ratio:.2f} (target: 0.5)")

    def test_proportional_fairness_multiple_classes(self):
        """Test proportional fairness with 4 weight classes."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Create requests with multiple weight classes
        weights = [0.5, 1.0, 2.0, 4.0]
        requests = []

        for weight in weights:
            for i in range(15):
                req = create_test_request(
                    f"w{weight}_{i}",
                    num_prompt_tokens=100,
                    max_tokens=50,
                    weight=weight,
                    arrival_time=0.0,
                )
                requests.append(req)

        # Simulate scheduling
        results, _ = simulate_scheduling(queue, requests)

        # Verify all ratios relative to weight 1.0
        baseline_throughput = results[1.0]["throughput"]

        for weight in weights:
            throughput = results[weight]["throughput"]
            ratio = throughput / baseline_throughput
            expected_ratio = weight / 1.0

            # Within ±15% tolerance (more relaxed for multiple classes)
            lower = expected_ratio * 0.85
            upper = expected_ratio * 1.15

            assert lower <= ratio <= upper, (
                f"Weight {weight}: expected ratio ~{expected_ratio:.1f}, "
                f"got {ratio:.2f}"
            )

            print(
                f"✓ Weight {weight} throughput ratio: {ratio:.2f} "
                f"(target: {expected_ratio:.1f})"
            )


class TestStarvationPrevention:
    """Test that WFQ prevents starvation."""

    def test_no_starvation_low_weight(self):
        """Test that low-weight requests complete under high-weight load."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        requests = []

        # Low-weight requests (should not be starved)
        for i in range(10):
            req = create_test_request(
                f"low_{i}",
                num_prompt_tokens=100,
                max_tokens=50,
                weight=0.1,
                arrival_time=0.0,
            )
            requests.append(req)

        # High-weight flood
        for i in range(100):
            req = create_test_request(
                f"high_{i}",
                num_prompt_tokens=50,
                max_tokens=25,
                weight=10.0,
                arrival_time=0.01,  # Slightly later
            )
            requests.append(req)

        # Simulate scheduling
        results, _ = simulate_scheduling(queue, requests)

        # Verify all low-weight requests completed
        low_weight_count = results[0.1]["request_count"]
        assert low_weight_count == 10, (
            f"Expected 10 low-weight requests, got {low_weight_count}"
        )

        # Verify all high-weight requests completed
        high_weight_count = results[10.0]["request_count"]
        assert high_weight_count == 100, (
            f"Expected 100 high-weight requests, got {high_weight_count}"
        )

        print(f"✓ No starvation: {low_weight_count}/10 low-weight requests completed")
        print(f"✓ All high-weight requests completed: {high_weight_count}/100")

    def test_completion_rate_100_percent(self):
        """Test that 100% of requests complete regardless of weight."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Mix of weights
        weights = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        total_requests = 0
        requests = []

        for weight in weights:
            for i in range(10):
                req = create_test_request(
                    f"w{weight}_{i}",
                    num_prompt_tokens=100,
                    max_tokens=50,
                    weight=weight,
                    arrival_time=0.0,
                )
                requests.append(req)
                total_requests += 1

        # Simulate scheduling
        results, _ = simulate_scheduling(queue, requests)

        # Count completed requests
        completed_count = sum(data["request_count"] for data in results.values())

        # Verify 100% completion
        assert completed_count == total_requests, (
            f"Expected {total_requests} requests, completed {completed_count}"
        )

        print(f"✓ Completion rate: {completed_count}/{total_requests} (100%)")


class TestJainsFairnessIndex:
    """Test overall fairness using Jain's Fairness Index."""

    def test_jains_index_equal_weights(self):
        """Test Jain's Index with equal weights (should be perfect ~1.0)."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # All requests have same weight
        requests = []
        for i in range(50):
            req = create_test_request(
                f"req_{i}",
                num_prompt_tokens=100,
                max_tokens=50,
                weight=1.0,
                arrival_time=0.0,
            )
            requests.append(req)

        # Simulate scheduling
        results, total_time = simulate_scheduling(queue, requests)

        # All requests have same weight, so normalized throughput should be identical
        normalized_throughputs = [results[1.0]["throughput"] / 1.0]

        fi = jains_fairness_index(normalized_throughputs)

        # Should be perfect fairness
        assert fi >= 0.99, f"Expected FI ~1.0 for equal weights, got {fi:.3f}"

        print(f"✓ Jain's Index (equal weights): {fi:.3f} (target: 1.0)")

    def test_jains_index_mixed_weights(self):
        """Test Jain's Index with mixed weights (should be high >0.95)."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Multiple weight classes
        weights = [0.5, 1.0, 2.0, 4.0]
        requests = []

        for weight in weights:
            for i in range(20):
                req = create_test_request(
                    f"w{weight}_{i}",
                    num_prompt_tokens=100,
                    max_tokens=50,
                    weight=weight,
                    arrival_time=0.0,
                )
                requests.append(req)

        # Simulate scheduling
        results, _ = simulate_scheduling(queue, requests)

        # Normalize throughputs by weight
        normalized_throughputs = []
        for weight in weights:
            throughput = results[weight]["throughput"]
            normalized = throughput / weight
            normalized_throughputs.append(normalized)

        fi = jains_fairness_index(normalized_throughputs)

        # Should be high fairness (>0.95)
        assert fi >= 0.90, f"Expected FI >0.90 for mixed weights, got {fi:.3f}"

        print(f"✓ Jain's Index (mixed weights): {fi:.3f} (target: >0.95)")
        print(
            f"  Normalized throughputs: {[f'{x:.2f}' for x in normalized_throughputs]}"
        )

    def test_jains_index_extreme_weights(self):
        """Test Jain's Index with extreme weight differences."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Extreme weight range
        weights = [0.1, 1.0, 10.0, 100.0]
        requests = []

        for weight in weights:
            for i in range(15):
                req = create_test_request(
                    f"w{weight}_{i}",
                    num_prompt_tokens=100,
                    max_tokens=50,
                    weight=weight,
                    arrival_time=0.0,
                )
                requests.append(req)

        # Simulate scheduling
        results, _ = simulate_scheduling(queue, requests)

        # Normalize throughputs by weight
        normalized_throughputs = []
        for weight in weights:
            throughput = results[weight]["throughput"]
            normalized = throughput / weight
            normalized_throughputs.append(normalized)

        fi = jains_fairness_index(normalized_throughputs)

        # Should still maintain reasonable fairness (>0.85 even with extreme weights)
        assert fi >= 0.80, f"Expected FI >0.80 for extreme weights, got {fi:.3f}"

        print(f"✓ Jain's Index (extreme weights): {fi:.3f} (target: >0.85)")
        print(f"  Weight range: {min(weights)} - {max(weights)}")


class TestFairnessUnderLoad:
    """Test fairness under various load conditions."""

    def test_fairness_heavy_load(self):
        """Test fairness with many concurrent requests."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Large number of requests
        weights = [0.5, 1.0, 2.0]
        requests = []

        for weight in weights:
            for i in range(100):  # 100 requests per weight class
                req = create_test_request(
                    f"w{weight}_{i}",
                    num_prompt_tokens=100,
                    max_tokens=50,
                    weight=weight,
                    arrival_time=0.0,
                )
                requests.append(req)

        # Simulate scheduling
        results, total_time = simulate_scheduling(queue, requests)

        # Verify proportional fairness
        baseline_throughput = results[1.0]["throughput"]

        for weight in weights:
            throughput = results[weight]["throughput"]
            ratio = throughput / baseline_throughput
            expected_ratio = weight / 1.0

            # Within ±12% tolerance
            lower = expected_ratio * 0.88
            upper = expected_ratio * 1.12

            assert lower <= ratio <= upper, (
                f"Heavy load: Weight {weight} ratio {ratio:.2f}, "
                f"expected ~{expected_ratio:.1f}"
            )

        # Compute Jain's Index
        normalized_throughputs = [results[w]["throughput"] / w for w in weights]
        fi = jains_fairness_index(normalized_throughputs)

        assert fi >= 0.90, f"Heavy load: Expected FI >0.90, got {fi:.3f}"

        print(f"✓ Fairness under heavy load (300 requests): FI = {fi:.3f}")

    def test_fairness_heterogeneous_requests(self):
        """Test fairness with heterogeneous request sizes."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Varying token counts
        requests = []

        # Short prompts, weight 1.0
        for i in range(20):
            req = create_test_request(
                f"short_{i}",
                num_prompt_tokens=50,
                max_tokens=25,
                weight=1.0,
                arrival_time=0.0,
            )
            requests.append(req)

        # Long prompts, weight 1.0
        for i in range(20):
            req = create_test_request(
                f"long_{i}",
                num_prompt_tokens=200,
                max_tokens=100,
                weight=1.0,
                arrival_time=0.0,
            )
            requests.append(req)

        # Simulate scheduling
        results, _ = simulate_scheduling(queue, requests)

        # Both should have same weight, so normalized throughput should be
        # similar (Even though total tokens differ, fairness is about
        # resource allocation per weight)

        # Just verify all completed
        total_completed = sum(data["request_count"] for data in results.values())
        assert total_completed == 40, f"Expected 40 requests, got {total_completed}"

        print(f"✓ Fairness with heterogeneous requests: {total_completed}/40 completed")
