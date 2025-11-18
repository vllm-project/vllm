"""Lab 07: Performance Monitoring - Starter Code"""

import time
from typing import List, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    latencies: List[float]
    throughput: float
    gpu_memory_used: float
    gpu_utilization: float


class MetricsCollector:
    """Collect and analyze performance metrics."""

    def __init__(self):
        self.latencies = []
        self.start_time = None

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
        # TODO 1: Calculate P50, P95, P99 latencies
        # TODO 2: Calculate average latency
        # TODO 3: Calculate throughput
        pass


def monitor_gpu_metrics() -> Dict[str, float]:
    """Monitor GPU memory and utilization."""
    # TODO 4: Implement GPU monitoring
    # Hint: Use pynvml or torch.cuda
    pass


def main():
    """Main monitoring demo."""
    print("=== Performance Monitoring Lab ===\n")

    collector = MetricsCollector()

    # TODO 5: Collect metrics from sample requests
    # TODO 6: Display metrics summary


if __name__ == "__main__":
    main()
