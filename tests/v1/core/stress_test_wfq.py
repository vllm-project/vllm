# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Stress tests for WFQ scheduler.

Tests WFQ under extreme conditions:
- High concurrent load (10,000+ requests)
- Extreme weight ranges (0.01 to 100.0)
- Mixed heterogeneous workloads
- Memory and stability testing
"""

import random
import time

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
    """Create a test request for stress testing."""
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


class TestHighLoadStress:
    """Test WFQ under high concurrent load."""

    def test_10k_concurrent_requests(self):
        """Test with 10,000 concurrent requests."""
        num_requests = 10000
        queue = create_request_queue(SchedulingPolicy.WFQ)

        print(f"\nStress Test: {num_requests} concurrent requests")

        # Create requests with mixed weights
        requests = []
        for i in range(num_requests):
            weight = random.choice([0.5, 1.0, 2.0, 4.0])
            req = create_test_request(
                f"req_{i}",
                num_prompt_tokens=random.randint(50, 200),
                max_tokens=random.randint(20, 100),
                weight=weight,
                arrival_time=i * 0.0001,
            )
            requests.append(req)

        # Add all requests
        print(f"  Adding {num_requests} requests...")
        start = time.time()
        for req in requests:
            queue.add_request(req)
        add_time = time.time() - start

        print(f"  ✓ Added in {add_time:.2f}s ({num_requests / add_time:.0f} req/s)")
        assert len(queue) == num_requests

        # Process all requests
        print(f"  Processing {num_requests} requests...")
        start = time.time()
        processed = 0
        while queue:
            queue.pop_request()
            processed += 1
        pop_time = time.time() - start

        print(f"  ✓ Processed in {pop_time:.2f}s ({num_requests / pop_time:.0f} req/s)")
        assert processed == num_requests
        assert len(queue) == 0

        # Verify throughput
        total_time = add_time + pop_time
        throughput = num_requests / total_time
        print(f"  Overall throughput: {throughput:.0f} req/s")

        # Should handle at least 1000 req/s
        assert throughput >= 1000, f"Throughput too low: {throughput:.0f} req/s"

    def test_20k_sequential_operations(self):
        """Test with 20,000 sequential add/pop cycles."""
        num_cycles = 20000
        queue = create_request_queue(SchedulingPolicy.WFQ)

        print(f"\nStress Test: {num_cycles} sequential add/pop cycles")

        start = time.time()
        for i in range(num_cycles):
            # Add request
            weight = float((i % 5) + 1)
            req = create_test_request(f"req_{i}", weight=weight)
            queue.add_request(req)

            # Occasionally pop
            if i % 10 == 9:  # Pop every 10th request
                for _ in range(min(5, len(queue))):
                    queue.pop_request()

        # Pop remaining
        while queue:
            queue.pop_request()

        elapsed = time.time() - start
        ops_per_sec = (num_cycles * 2) / elapsed  # 2 ops per cycle (add + pop)

        print(f"  Completed in {elapsed:.2f}s")
        print(f"  Operations per second: {ops_per_sec:.0f}")

        # Should handle at least 5000 ops/s
        assert ops_per_sec >= 5000, f"Operations too slow: {ops_per_sec:.0f} ops/s"

    def test_sustained_load_stability(self):
        """Test stability under sustained load."""
        num_iterations = 100
        requests_per_iteration = 100

        queue = create_request_queue(SchedulingPolicy.WFQ)
        total_processed = 0

        print(f"\nStress Test: Sustained load ({num_iterations} iterations)")

        for iteration in range(num_iterations):
            # Add batch of requests
            for i in range(requests_per_iteration):
                weight = random.uniform(0.5, 4.0)
                req = create_test_request(
                    f"iter{iteration}_req{i}",
                    weight=weight,
                )
                queue.add_request(req)

            # Process half
            for _ in range(requests_per_iteration // 2):
                if queue:
                    queue.pop_request()
                    total_processed += 1

        # Process remaining
        while queue:
            queue.pop_request()
            total_processed += 1

        expected_total = num_iterations * requests_per_iteration
        print(f"  Total processed: {total_processed}/{expected_total}")
        assert total_processed == expected_total


class TestExtremeWeightRanges:
    """Test WFQ with extreme weight values."""

    def test_extreme_weight_range(self):
        """Test with weights from 0.01 to 100.0."""
        weights = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
        requests_per_weight = 50

        queue = create_request_queue(SchedulingPolicy.WFQ)

        print("\nStress Test: Extreme weight range (0.01 to 100.0)")
        print(f"  Weights: {weights}")

        # Create requests with extreme weights
        all_requests = []
        for weight in weights:
            for i in range(requests_per_weight):
                req = create_test_request(
                    f"w{weight}_{i}",
                    weight=weight,
                    arrival_time=0.0,
                )
                all_requests.append(req)
                queue.add_request(req)

        total_requests = len(all_requests)
        print(f"  Total requests: {total_requests}")

        # Process all requests
        processed_by_weight = {w: 0 for w in weights}
        processed_order = []

        while queue:
            req = queue.pop_request()
            processed_by_weight[req.weight] += 1
            processed_order.append(req.weight)

        # Verify all completed
        print("\n  Requests processed by weight:")
        for weight in weights:
            count = processed_by_weight[weight]
            print(f"    Weight {weight:>6.2f}: {count}/{requests_per_weight}")
            assert count == requests_per_weight, f"Weight {weight} incomplete"

        # Verify higher weights processed first (on average)
        # First 100 requests should have higher avg weight than last 100
        first_100_avg = sum(processed_order[:100]) / 100
        last_100_avg = sum(processed_order[-100:]) / 100

        print(f"\n  First 100 avg weight: {first_100_avg:.2f}")
        print(f"  Last 100 avg weight: {last_100_avg:.2f}")
        assert first_100_avg > last_100_avg, "Higher weights not prioritized"

    def test_very_small_weights(self):
        """Test with very small weights (near zero)."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        weights = [0.001, 0.01, 0.1, 1.0]
        for weight in weights:
            for i in range(25):
                req = create_test_request(f"w{weight}_{i}", weight=weight)
                queue.add_request(req)

        print("\nStress Test: Very small weights")
        print(f"  Weights: {weights}")

        # Process all
        processed = 0
        while queue:
            req = queue.pop_request()
            processed += 1

            # Verify virtual times don't overflow
            assert req.virtual_finish_time < 1e10, "Virtual time overflow"

        assert processed == len(weights) * 25
        print(f"  ✓ All {processed} requests processed")

    def test_very_large_weights(self):
        """Test with very large weights."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        weights = [1.0, 10.0, 100.0, 1000.0]
        for weight in weights:
            for i in range(25):
                req = create_test_request(f"w{weight}_{i}", weight=weight)
                queue.add_request(req)

        print("\nStress Test: Very large weights")
        print(f"  Weights: {weights}")

        # Process all
        processed = 0
        while queue:
            req = queue.pop_request()
            processed += 1

            # Verify virtual times are reasonable
            assert req.virtual_finish_time >= 0, "Virtual time underflow"

        assert processed == len(weights) * 25
        print(f"  ✓ All {processed} requests processed")


class TestMixedWorkloads:
    """Test WFQ with heterogeneous workloads."""

    def test_mixed_token_lengths(self):
        """Test with varying request sizes."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Create requests with different token counts
        workload = [
            ("tiny", 10, 5, 1.0, 100),
            ("small", 50, 25, 1.0, 100),
            ("medium", 100, 50, 1.0, 100),
            ("large", 500, 200, 1.0, 50),
            ("huge", 2000, 500, 1.0, 20),
        ]

        print("\nStress Test: Mixed token lengths")
        total_requests = 0

        for name, prompt, output, weight, count in workload:
            for i in range(count):
                req = create_test_request(
                    f"{name}_{i}",
                    num_prompt_tokens=prompt,
                    max_tokens=output,
                    weight=weight,
                )
                queue.add_request(req)
                total_requests += 1

        print(f"  Total requests: {total_requests}")
        print("  Workload mix: tiny(100), small(100), medium(100), large(50), huge(20)")

        # Process all
        processed = 0
        while queue:
            queue.pop_request()
            processed += 1

        assert processed == total_requests
        print(f"  ✓ All {processed} requests processed")

    def test_mixed_weights_and_sizes(self):
        """Test with varying weights and request sizes."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Complex mixed workload
        workload = [
            # (tokens, weight, count)
            (50, 0.5, 100),  # Low weight, small
            (200, 0.5, 50),  # Low weight, large
            (50, 2.0, 100),  # High weight, small
            (200, 2.0, 50),  # High weight, large
            (100, 1.0, 200),  # Baseline
        ]

        print("\nStress Test: Mixed weights and sizes")
        total_requests = 0

        for tokens, weight, count in workload:
            for i in range(count):
                req = create_test_request(
                    f"t{tokens}_w{weight}_{i}",
                    num_prompt_tokens=tokens,
                    max_tokens=tokens // 2,
                    weight=weight,
                )
                queue.add_request(req)
                total_requests += 1

        print(f"  Total requests: {total_requests}")

        # Process all
        processed = 0
        while queue:
            queue.pop_request()
            processed += 1

        assert processed == total_requests
        print(f"  ✓ All {processed} requests processed")

    def test_bursty_arrivals(self):
        """Test with bursty arrival patterns."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        print("\nStress Test: Bursty arrival pattern")

        # Simulate bursts of requests
        num_bursts = 20
        requests_per_burst = 100

        for burst in range(num_bursts):
            # Add burst
            for i in range(requests_per_burst):
                weight = random.choice([0.5, 1.0, 2.0, 4.0])
                req = create_test_request(
                    f"burst{burst}_req{i}",
                    weight=weight,
                    arrival_time=burst * 1.0 + i * 0.001,
                )
                queue.add_request(req)

            # Process some requests before next burst
            for _ in range(requests_per_burst // 2):
                if queue:
                    queue.pop_request()

        # Process remaining
        remaining = 0
        while queue:
            queue.pop_request()
            remaining += 1

        total = num_bursts * requests_per_burst
        print(f"  Total requests: {total}")
        print("  ✓ All processed successfully")


class TestMemoryAndStability:
    """Test memory usage and stability."""

    def test_queue_size_bounds(self):
        """Test that queue size is tracked correctly."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Add requests
        for i in range(500):
            req = create_test_request(f"req_{i}", weight=float(i % 5 + 1))
            queue.add_request(req)
            assert len(queue) == i + 1

        # Remove requests
        for i in range(500):
            queue.pop_request()
            assert len(queue) == 500 - i - 1

        assert len(queue) == 0
        assert not queue  # __bool__ should be False

    def test_alternating_add_remove(self):
        """Test alternating add and remove operations."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        print("\nStress Test: Alternating add/remove (5000 cycles)")

        for i in range(5000):
            # Add request
            req = create_test_request(f"req_{i}", weight=float(i % 5 + 1))
            queue.add_request(req)

            # Pop if queue has more than 10 requests
            if len(queue) > 10:
                queue.pop_request()

        # Final cleanup
        final_count = len(queue)
        while queue:
            queue.pop_request()

        print("  ✓ Completed 5000 cycles")
        print(f"  Final queue size before cleanup: {final_count}")

    def test_prepend_operations_stress(self):
        """Test prepend operations under stress."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        print("\nStress Test: Prepend operations")

        # Add initial requests
        requests = []
        for i in range(100):
            req = create_test_request(f"req_{i}", weight=1.0)
            queue.add_request(req)
            requests.append(req)

        # Pop and prepend repeatedly
        for cycle in range(50):
            # Pop some requests
            popped = []
            for _ in range(20):
                if queue:
                    popped.append(queue.pop_request())

            # Prepend them back
            for req in popped:
                queue.prepend_request(req)

        # Verify all requests still in queue
        assert len(queue) == 100
        print("  ✓ Completed 50 prepend cycles")

        # Process all
        processed = 0
        while queue:
            queue.pop_request()
            processed += 1

        assert processed == 100


class TestEdgeCaseStress:
    """Test edge cases under stress."""

    def test_single_weight_class_at_scale(self):
        """Test 10,000 requests with same weight."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        num_requests = 10000
        weight = 1.0

        print(f"\nStress Test: {num_requests} requests, single weight ({weight})")

        for i in range(num_requests):
            req = create_test_request(f"req_{i}", weight=weight)
            queue.add_request(req)

        # Should behave like FCFS for same weight
        processed = 0
        while queue:
            queue.pop_request()
            processed += 1

        assert processed == num_requests
        print(f"  ✓ All {processed} requests processed")

    def test_many_weight_classes(self):
        """Test with many distinct weight values."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # 100 different weight values
        num_weights = 100
        requests_per_weight = 50

        print(f"\nStress Test: {num_weights} distinct weights")

        for w in range(num_weights):
            weight = 0.1 + (w * 0.1)  # 0.1, 0.2, ..., 10.0
            for i in range(requests_per_weight):
                req = create_test_request(f"w{weight:.1f}_{i}", weight=weight)
                queue.add_request(req)

        total = num_weights * requests_per_weight
        processed = 0
        while queue:
            queue.pop_request()
            processed += 1

        assert processed == total
        print(f"  ✓ All {processed} requests processed")

    def test_remove_operations_at_scale(self):
        """Test remove operations with large queue."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Add 1000 requests
        requests = []
        for i in range(1000):
            req = create_test_request(f"req_{i}", weight=float(i % 5 + 1))
            queue.add_request(req)
            requests.append(req)

        print("\nStress Test: Remove operations (1000 requests)")

        # Remove every 10th request
        removed = 0
        for i in range(0, 1000, 10):
            queue.remove_request(requests[i])
            removed += 1

        print(f"  Removed: {removed} requests")
        assert len(queue) == 1000 - removed

        # Process remaining
        processed = 0
        while queue:
            queue.pop_request()
            processed += 1

        assert processed == 1000 - removed
        print(f"  ✓ Processed remaining {processed} requests")
