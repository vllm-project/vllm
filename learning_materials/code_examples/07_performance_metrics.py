"""
Example 07: Performance Metrics Collection

Demonstrates collecting and analyzing performance metrics.

Usage:
    python 07_performance_metrics.py
"""

import time
import numpy as np
from typing import List
from vllm import LLM, SamplingParams


class PerformanceTracker:
    """Track inference performance metrics."""

    def __init__(self):
        self.latencies: List[float] = []
        self.start_time = time.time()

    def record_latency(self, latency: float) -> None:
        """Record a single request latency."""
        self.latencies.append(latency)

    def get_summary(self) -> dict:
        """Get performance summary."""
        if not self.latencies:
            return {}

        latencies_ms = [l * 1000 for l in self.latencies]
        return {
            "count": len(self.latencies),
            "avg_latency_ms": np.mean(latencies_ms),
            "p50_latency_ms": np.percentile(latencies_ms, 50),
            "p95_latency_ms": np.percentile(latencies_ms, 95),
            "p99_latency_ms": np.percentile(latencies_ms, 99),
            "min_latency_ms": np.min(latencies_ms),
            "max_latency_ms": np.max(latencies_ms),
            "throughput_rps": len(self.latencies) / (time.time() - self.start_time),
        }


def main():
    """Demo performance tracking."""
    print("=== Performance Metrics Demo ===\n")

    llm = LLM(model="facebook/opt-125m", trust_remote_code=True)
    tracker = PerformanceTracker()

    prompts = [f"Test prompt {i}" for i in range(50)]
    sampling_params = SamplingParams(temperature=0.8, max_tokens=50)

    print("Running inference with metrics collection...\n")

    for prompt in prompts:
        start = time.time()
        llm.generate([prompt], sampling_params)
        latency = time.time() - start
        tracker.record_latency(latency)

    summary = tracker.get_summary()

    print("Performance Summary:")
    print(f"  Total requests: {summary['count']}")
    print(f"  Average latency: {summary['avg_latency_ms']:.2f}ms")
    print(f"  P50 latency: {summary['p50_latency_ms']:.2f}ms")
    print(f"  P95 latency: {summary['p95_latency_ms']:.2f}ms")
    print(f"  P99 latency: {summary['p99_latency_ms']:.2f}ms")
    print(f"  Throughput: {summary['throughput_rps']:.2f} req/s")


if __name__ == "__main__":
    main()
