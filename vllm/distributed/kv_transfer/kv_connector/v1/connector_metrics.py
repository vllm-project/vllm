# Copyright (c) 2024, vLLM Contributors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Connector metrics for KV transfer operations."""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

from .base import KVConnectorStats


@dataclass 
class TimestampedStats:
    """KVConnectorStats with timestamp for rolling window tracking."""
    stats: KVConnectorStats
    timestamp: float


@dataclass
class MetricAggregation:
    """Aggregated statistics for a single metric."""
    avg: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p99: float = 0.0


@dataclass
class AggregatedKVConnectorStats:
    """Aggregated connector stats over a rolling time window with percentiles."""
    stats_count: int = 0
    metrics: Dict[str, MetricAggregation] = field(default_factory=dict)


class KVConnectorMetrics:
    """Aggregates KVConnectorStats using a rolling time window.
    
    This class maintains a rolling window of KVConnectorStats and continuously
    removes samples older than the specified time interval.
    """
    
    def __init__(self, time_interval_seconds: float = 60.0):
        """Initialize the metrics aggregator.
        
        Args:
            time_interval_seconds: Time interval for the rolling window (default: 60 seconds)
        """
        self.time_interval = time_interval_seconds
        
        # Rolling window of timestamped stats
        self.rolling_stats: deque[TimestampedStats] = deque()
    
    def observe(self, stats: KVConnectorStats) -> None:
        """Observe new connector stats and add to the rolling window.
        
        Args:
            stats: KVConnectorStats instance to observe and add to metrics
        """
        current_time = time.time()
        
        # Remove old samples outside the time interval
        self._cleanup_old_samples(current_time)
        
        # Add new timestamped stats
        timestamped_stats = TimestampedStats(stats=stats, timestamp=current_time)
        self.rolling_stats.append(timestamped_stats)
    
    def _cleanup_old_samples(self, current_time: float) -> None:
        """Remove samples older than the time interval."""
        cutoff_time = current_time - self.time_interval
        
        # Remove old stats from rolling window
        while self.rolling_stats and self.rolling_stats[0].timestamp < cutoff_time:
            self.rolling_stats.popleft()
    
    def get_current_aggregated_stats(self) -> Optional[AggregatedKVConnectorStats]:
        """Get aggregated stats for the current rolling window."""
        current_time = time.time()
        self._cleanup_old_samples(current_time)
        
        if not self.rolling_stats:
            return None
        
        # Aggregate metrics from all stats samples
        aggregated_metrics = self._calculate_metric_aggregations()
        
        return AggregatedKVConnectorStats(
            stats_count=len(self.rolling_stats),
            metrics=aggregated_metrics
        )
    
    def _calculate_metric_aggregations(self) -> Dict[str, MetricAggregation]:
        """Calculate avg, p50, p90, p99 for each metric across all samples."""
        if not self.rolling_stats:
            return {}
        
        # Collect all metric keys from all samples
        all_metric_keys = set()
        for sample in self.rolling_stats:
            all_metric_keys.update(sample.stats.keys())
        
        aggregated_metrics = {}
        
        # For each metric, collect all values and calculate percentiles
        for metric_key in all_metric_keys:
            values = []
            for sample in self.rolling_stats:
                # Get the value for this metric, use 0 as default if metric doesn't exist in this sample
                metric_value = sample.stats.get(metric_key, 0)
                values.append(metric_value)
            
            # Sort values for percentile calculation
            values.sort()
            n = len(values)
            
            # Calculate aggregations
            avg = sum(values) / n if values else 0.0
            p50 = self._calculate_percentile(values, 50)
            p90 = self._calculate_percentile(values, 90)
            p99 = self._calculate_percentile(values, 99)
            
            aggregated_metrics[metric_key] = MetricAggregation(
                avg=avg,
                p50=p50,
                p90=p90,
                p99=p99
            )
        
        return aggregated_metrics
    
    def _calculate_percentile(self, sorted_values: list[int], percentile: int) -> float:
        """Calculate the specified percentile from sorted values."""
        if not sorted_values:
            return 0.0
        
        n = len(sorted_values)
        if n == 1:
            return float(sorted_values[0])
        
        # Use linear interpolation method
        index = (percentile / 100.0) * (n - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, n - 1)
        
        if lower_index == upper_index:
            return float(sorted_values[lower_index])
        
        # Linear interpolation
        weight = index - lower_index
        return float(sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight)
    
    def reset(self) -> None:
        """Reset all metrics and clear the rolling window."""
        self.rolling_stats.clear()
