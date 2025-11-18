"""Lab 07: Performance Monitoring - Complete Solution"""

import time
from typing import List, Dict
import numpy as np
import torch


class MetricsCollector:
    """Collect and analyze performance metrics."""

    def __init__(self):
        self.latencies = []
        self.start_time = time.time()

    def start_request(self) -> float:
        """Start timing a request."""
        return time.time()

    def end_request(self, start_time: float) -> float:
        """End timing and record latency."""
        latency = time.time() - start_time
        self.latencies.append(latency)
        return latency

    def get_metrics(self) -> Dict[str, float]:
        """Calculate metrics summary."""
        latencies_ms = [l * 1000 for l in self.latencies]

        return {
            "avg_latency_ms": np.mean(latencies_ms),
            "p50_latency_ms": np.percentile(latencies_ms, 50),
            "p95_latency_ms": np.percentile(latencies_ms, 95),
            "p99_latency_ms": np.percentile(latencies_ms, 99),
            "throughput": len(self.latencies) / (time.time() - self.start_time),
        }


def monitor_gpu_metrics() -> Dict[str, float]:
    """Monitor GPU memory and utilization."""
    if torch.cuda.is_available():
        return {
            "gpu_memory_used_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    return {}


def main():
    """Main monitoring demo."""
    print("=== Performance Monitoring Lab ===\n")

    collector = MetricsCollector()

    # Simulate requests
    for _ in range(100):
        start = collector.start_request()
        time.sleep(0.01)  # Simulate processing
        collector.end_request(start)

    metrics = collector.get_metrics()
    print("Metrics Summary:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")

    gpu_metrics = monitor_gpu_metrics()
    for key, value in gpu_metrics.items():
        print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()
