# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Performance benchmarks for WFQ scheduler.

Tests throughput, latency, and queue operation performance.
Compares WFQ against FCFS and Priority scheduling policies.
"""

import statistics
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
    """Create a test request for performance benchmarking."""
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


class TestThroughputComparison:
    """Compare throughput across scheduling policies."""

    def test_throughput_wfq_vs_fcfs(self):
        """Compare WFQ and FCFS throughput."""
        num_requests = 100

        # Create identical requests for both policies
        def create_requests():
            requests = []
            for i in range(num_requests):
                req = create_test_request(
                    f"req_{i}",
                    num_prompt_tokens=100,
                    max_tokens=50,
                    weight=1.0 if i % 2 == 0 else 2.0,  # Mixed weights for WFQ
                    arrival_time=i * 0.001,
                )
                requests.append(req)
            return requests

        # Test WFQ
        wfq_queue = create_request_queue(SchedulingPolicy.WFQ)
        wfq_requests = create_requests()

        start = time.time()
        for req in wfq_requests:
            wfq_queue.add_request(req)
        for _ in range(num_requests):
            wfq_queue.pop_request()
        wfq_time = time.time() - start

        # Test FCFS
        fcfs_queue = create_request_queue(SchedulingPolicy.FCFS)
        fcfs_requests = create_requests()

        start = time.time()
        for req in fcfs_requests:
            fcfs_queue.add_request(req)
        for _ in range(num_requests):
            fcfs_queue.pop_request()
        fcfs_time = time.time() - start

        # Calculate throughput
        wfq_throughput = num_requests / wfq_time
        fcfs_throughput = num_requests / fcfs_time

        print(f"\nThroughput Comparison ({num_requests} requests):")
        print(f"  WFQ:  {wfq_throughput:.2f} req/s ({wfq_time * 1000:.2f} ms total)")
        print(f"  FCFS: {fcfs_throughput:.2f} req/s ({fcfs_time * 1000:.2f} ms total)")
        print(f"  Ratio (WFQ/FCFS): {wfq_throughput / fcfs_throughput:.2f}")

        # WFQ should be within 20% of FCFS
        # (may be slightly slower due to heap ops)
        assert wfq_throughput >= fcfs_throughput * 0.80, (
            f"WFQ throughput too low: {wfq_throughput:.2f} "
            f"vs FCFS {fcfs_throughput:.2f}"
        )

    def test_throughput_wfq_vs_priority(self):
        """Compare WFQ and Priority throughput."""
        num_requests = 100

        # Create requests with priorities/weights
        def create_wfq_requests():
            requests = []
            for i in range(num_requests):
                req = create_test_request(
                    f"req_{i}",
                    num_prompt_tokens=100,
                    max_tokens=50,
                    weight=float(i % 5 + 1),  # Weights 1-5
                    arrival_time=i * 0.001,
                )
                requests.append(req)
            return requests

        def create_priority_requests():
            requests = []
            for i in range(num_requests):
                req = create_test_request(
                    f"req_{i}",
                    num_prompt_tokens=100,
                    max_tokens=50,
                    priority=i % 5,  # Priorities 0-4
                    arrival_time=i * 0.001,
                )
                requests.append(req)
            return requests

        # Test WFQ
        wfq_queue = create_request_queue(SchedulingPolicy.WFQ)
        wfq_requests = create_wfq_requests()

        start = time.time()
        for req in wfq_requests:
            wfq_queue.add_request(req)
        for _ in range(num_requests):
            wfq_queue.pop_request()
        wfq_time = time.time() - start

        # Test Priority
        priority_queue = create_request_queue(SchedulingPolicy.PRIORITY)
        priority_requests = create_priority_requests()

        start = time.time()
        for req in priority_requests:
            priority_queue.add_request(req)
        for _ in range(num_requests):
            priority_queue.pop_request()
        priority_time = time.time() - start

        # Calculate throughput
        wfq_throughput = num_requests / wfq_time
        priority_throughput = num_requests / priority_time

        print(f"\nThroughput Comparison ({num_requests} requests):")
        print(
            f"  WFQ:      {wfq_throughput:.2f} req/s ({wfq_time * 1000:.2f} ms total)"
        )
        print(
            f"  Priority: {priority_throughput:.2f} req/s "
            f"({priority_time * 1000:.2f} ms total)"
        )
        print(f"  Ratio (WFQ/Priority): {wfq_throughput / priority_throughput:.2f}")

        # WFQ and Priority should have similar throughput (both use heaps)
        assert wfq_throughput >= priority_throughput * 0.80, (
            f"WFQ throughput too low: {wfq_throughput:.2f} "
            f"vs Priority {priority_throughput:.2f}"
        )

    def test_throughput_scaling(self):
        """Test throughput with increasing request counts."""
        sizes = [10, 50, 100, 500, 1000]
        results = []

        for size in sizes:
            wfq_queue = create_request_queue(SchedulingPolicy.WFQ)

            # Create requests
            requests = [
                create_test_request(f"req_{i}", weight=float(i % 5 + 1))
                for i in range(size)
            ]

            # Measure throughput
            start = time.time()
            for req in requests:
                wfq_queue.add_request(req)
            for _ in range(size):
                wfq_queue.pop_request()
            elapsed = time.time() - start

            throughput = size / elapsed
            results.append((size, throughput, elapsed))

        print("\nThroughput Scaling:")
        print(f"{'Size':<10} {'Throughput':<15} {'Time':<15}")
        print("-" * 40)
        for size, throughput, elapsed in results:
            print(f"{size:<10} {throughput:>10.2f} req/s {elapsed * 1000:>10.2f} ms")

        # Verify throughput doesn't degrade significantly with scale
        # (Should remain relatively constant for O(log n) operations)
        throughputs = [t for _, t, _ in results]
        min_throughput = min(throughputs)
        max_throughput = max(throughputs)

        # Max should not be more than 3x min
        # (accounts for startup overhead at small sizes)
        assert max_throughput <= min_throughput * 3.0, (
            f"Throughput degrades significantly: "
            f"{min_throughput:.2f} to {max_throughput:.2f}"
        )


class TestLatencyDistribution:
    """Test latency characteristics."""

    def test_latency_by_weight_class(self):
        """Measure latency distribution by weight class."""
        num_requests_per_weight = 50
        weights = [0.5, 1.0, 2.0, 4.0]

        queue = create_request_queue(SchedulingPolicy.WFQ)
        latencies_by_weight = defaultdict(list)

        # Create and add all requests
        all_requests = []
        for weight in weights:
            for i in range(num_requests_per_weight):
                req = create_test_request(
                    f"w{weight}_{i}",
                    weight=weight,
                    arrival_time=0.0,
                )
                all_requests.append(req)
                queue.add_request(req)

        # Process requests and measure latency
        current_time = 0.0
        while queue:
            req = queue.pop_request()

            # Latency = time from arrival to start of processing
            latency = current_time - req.arrival_time
            latencies_by_weight[req.weight].append(latency)

            # Simulate processing time
            tokens = req.num_prompt_tokens + req.max_tokens
            current_time += tokens * 0.001  # 1ms per token

        # Compute statistics
        print("\nLatency Distribution by Weight Class:")
        print(
            f"{'Weight':<10} {'P50 (ms)':<12} {'P95 (ms)':<12} "
            f"{'P99 (ms)':<12} {'Mean (ms)':<12}"
        )
        print("-" * 60)

        for weight in weights:
            latencies = [
                latency * 1000 for latency in latencies_by_weight[weight]
            ]  # Convert to ms
            p50 = statistics.median(latencies)
            p95 = (
                statistics.quantiles(latencies, n=20)[18]
                if len(latencies) > 20
                else max(latencies)
            )
            p99 = (
                statistics.quantiles(latencies, n=100)[98]
                if len(latencies) > 100
                else max(latencies)
            )
            mean = statistics.mean(latencies)

            print(
                f"{weight:<10.1f} {p50:>10.2f} {p95:>10.2f} {p99:>10.2f} {mean:>10.2f}"
            )

        # Verify higher weights have lower latency
        for i in range(len(weights) - 1):
            mean_low = statistics.mean(latencies_by_weight[weights[i]])
            mean_high = statistics.mean(latencies_by_weight[weights[i + 1]])

            # Higher weight should have lower mean latency
            assert mean_high <= mean_low, (
                f"Weight {weights[i + 1]} should have lower latency than {weights[i]}"
            )

    def test_p99_latency_overhead(self):
        """Verify WFQ P99 latency is within 2x of FCFS."""
        num_requests = 200

        # Create identical requests
        def create_requests():
            return [
                create_test_request(f"req_{i}", weight=1.0, arrival_time=0.0)
                for i in range(num_requests)
            ]

        # Test WFQ
        wfq_queue = create_request_queue(SchedulingPolicy.WFQ)
        wfq_requests = create_requests()
        wfq_latencies = []

        for req in wfq_requests:
            wfq_queue.add_request(req)

        current_time = 0.0
        while wfq_queue:
            req = wfq_queue.pop_request()
            latency = current_time - req.arrival_time
            wfq_latencies.append(latency)
            current_time += 0.15  # 150ms processing time

        # Test FCFS
        fcfs_queue = create_request_queue(SchedulingPolicy.FCFS)
        fcfs_requests = create_requests()
        fcfs_latencies = []

        for req in fcfs_requests:
            fcfs_queue.add_request(req)

        current_time = 0.0
        while fcfs_queue:
            req = fcfs_queue.pop_request()
            latency = current_time - req.arrival_time
            fcfs_latencies.append(latency)
            current_time += 0.15

        # Compute P99
        wfq_p99 = (
            statistics.quantiles(wfq_latencies, n=100)[98]
            if len(wfq_latencies) > 100
            else max(wfq_latencies)
        )
        fcfs_p99 = (
            statistics.quantiles(fcfs_latencies, n=100)[98]
            if len(fcfs_latencies) > 100
            else max(fcfs_latencies)
        )

        print(f"\nP99 Latency Comparison ({num_requests} requests):")
        print(f"  WFQ P99:  {wfq_p99 * 1000:.2f} ms")
        print(f"  FCFS P99: {fcfs_p99 * 1000:.2f} ms")
        print(f"  Ratio (WFQ/FCFS): {wfq_p99 / fcfs_p99:.2f}")

        # WFQ P99 should be within 2x of FCFS
        assert wfq_p99 <= fcfs_p99 * 2.0, (
            f"WFQ P99 latency too high: {wfq_p99 * 1000:.2f} ms "
            f"vs FCFS {fcfs_p99 * 1000:.2f} ms"
        )


class TestQueueOperationPerformance:
    """Benchmark individual queue operations."""

    def test_add_request_performance(self):
        """Benchmark add_request operation."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Pre-create requests
        requests = [
            create_test_request(f"req_{i}", weight=float(i % 5 + 1))
            for i in range(1000)
        ]

        # Measure add_request time
        start = time.time()
        for req in requests:
            queue.add_request(req)
        elapsed = time.time() - start

        avg_time_ms = (elapsed / len(requests)) * 1000

        print("\nadd_request Performance:")
        print(f"  Total time: {elapsed * 1000:.2f} ms")
        print(f"  Requests: {len(requests)}")
        print(f"  Avg time per request: {avg_time_ms:.4f} ms")
        print("  Target: < 1.0 ms")

        # Should be well under 1ms per operation
        assert avg_time_ms < 1.0, f"add_request too slow: {avg_time_ms:.4f} ms"

    def test_pop_request_performance(self):
        """Benchmark pop_request operation."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Add requests first
        num_requests = 1000
        for i in range(num_requests):
            req = create_test_request(f"req_{i}", weight=float(i % 5 + 1))
            queue.add_request(req)

        # Measure pop_request time
        start = time.time()
        while queue:
            queue.pop_request()
        elapsed = time.time() - start

        avg_time_ms = (elapsed / num_requests) * 1000

        print("\npop_request Performance:")
        print(f"  Total time: {elapsed * 1000:.2f} ms")
        print(f"  Requests: {num_requests}")
        print(f"  Avg time per request: {avg_time_ms:.4f} ms")
        print("  Target: < 1.0 ms")

        # Should be well under 1ms per operation
        assert avg_time_ms < 1.0, f"pop_request too slow: {avg_time_ms:.4f} ms"

    def test_peek_request_performance(self):
        """Benchmark peek_request operation."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Add requests
        for i in range(100):
            req = create_test_request(f"req_{i}", weight=float(i % 5 + 1))
            queue.add_request(req)

        # Measure peek_request time
        num_peeks = 10000
        start = time.time()
        for _ in range(num_peeks):
            queue.peek_request()
        elapsed = time.time() - start

        avg_time_us = (elapsed / num_peeks) * 1000000  # microseconds

        print("\npeek_request Performance:")
        print(f"  Total time: {elapsed * 1000:.2f} ms")
        print(f"  Operations: {num_peeks}")
        print(f"  Avg time per operation: {avg_time_us:.2f} µs")
        print("  Target: < 100 µs")

        # Should be very fast (< 100 microseconds)
        assert avg_time_us < 100, f"peek_request too slow: {avg_time_us:.2f} µs"

    def test_remove_request_performance(self):
        """Benchmark remove_request operation."""
        queue = create_request_queue(SchedulingPolicy.WFQ)

        # Add requests
        requests = []
        for i in range(500):
            req = create_test_request(f"req_{i}", weight=float(i % 5 + 1))
            queue.add_request(req)
            requests.append(req)

        # Measure remove_request time
        num_removes = 100
        start = time.time()
        for i in range(num_removes):
            queue.remove_request(requests[i])
        elapsed = time.time() - start

        avg_time_ms = (elapsed / num_removes) * 1000

        print("\nremove_request Performance:")
        print(f"  Total time: {elapsed * 1000:.2f} ms")
        print(f"  Operations: {num_removes}")
        print(f"  Avg time per operation: {avg_time_ms:.2f} ms")
        print("  Target: < 10 ms (O(n) operation)")

        # Remove is O(n), so more relaxed requirement
        assert avg_time_ms < 10.0, f"remove_request too slow: {avg_time_ms:.2f} ms"


class TestComplexityVerification:
    """Verify algorithmic complexity."""

    def test_add_logarithmic_complexity(self):
        """Verify add_request is O(log n)."""
        sizes = [100, 500, 1000, 5000]
        times = []

        for size in sizes:
            queue = create_request_queue(SchedulingPolicy.WFQ)
            requests = [
                create_test_request(f"req_{i}", weight=float(i % 5 + 1))
                for i in range(size)
            ]

            start = time.time()
            for req in requests:
                queue.add_request(req)
            elapsed = time.time() - start

            avg_time_ms = (elapsed / size) * 1000
            times.append((size, avg_time_ms))

        print("\nAdd Request Complexity Verification:")
        print(f"{'Size':<10} {'Avg Time (ms)':<15} {'Expected if O(log n)':<25}")
        print("-" * 50)

        # For O(log n), doubling size should increase time by constant factor
        for i, (size, avg_time) in enumerate(times):
            if i == 0:
                base_size, base_time = size, avg_time
            expected = base_time * (size / base_size) ** 0.5  # Rough O(log n) estimate
            print(f"{size:<10} {avg_time:>10.4f} {expected:>20.4f}")

        # Verify times don't grow linearly (would indicate O(n))
        # Time for 5000 should be < 10x time for 100 (log(5000)/log(100) ≈ 1.85)
        assert times[-1][1] < times[0][1] * 10, (
            "add_request appears to be O(n), not O(log n)"
        )

    def test_pop_logarithmic_complexity(self):
        """Verify pop_request is O(log n)."""
        sizes = [100, 500, 1000, 5000]
        times = []

        for size in sizes:
            queue = create_request_queue(SchedulingPolicy.WFQ)

            # Add requests first
            for i in range(size):
                req = create_test_request(f"req_{i}", weight=float(i % 5 + 1))
                queue.add_request(req)

            # Measure pop time
            start = time.time()
            count = 0
            while queue and count < size:
                queue.pop_request()
                count += 1
            elapsed = time.time() - start

            avg_time_ms = (elapsed / size) * 1000
            times.append((size, avg_time_ms))

        print("\nPop Request Complexity Verification:")
        print(f"{'Size':<10} {'Avg Time (ms)':<15}")
        print("-" * 30)

        for size, avg_time in times:
            print(f"{size:<10} {avg_time:>10.4f}")

        # Verify times don't grow linearly
        assert times[-1][1] < times[0][1] * 10, (
            "pop_request appears to be O(n), not O(log n)"
        )
