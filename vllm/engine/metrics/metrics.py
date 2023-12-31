from typing import Optional, List

from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.metrics.metrics_registry import METRIC_REGISTRY
from vllm.engine.metrics.metrics_utils import (
    PrometheusMetric, GaugeMetric,
    IterationStats, SystemStats, Stats
)

labels = {}
            
def add_global_metrics_labels(**kwargs):
    labels.update(kwargs)

class MetricLogger:
    def __init__(self, logging_interval: float):
        self.logging_interval = logging_interval
        self.last_logging_time = 0.0
        
        # Iteration level stats that we save to log periodically.
        self.iteration_stats = IterationStats()

        # Gauge Metrics from Registry.
        self.metrics: List[PrometheusMetric] = [
            metric_class(
                prometheus_metric=prometheus_metric, labels=labels
            ) for prometheus_metric, metric_class in METRIC_REGISTRY
        ]

    def should_log(self, now: float) -> bool:
        return now - self.last_logging_time >= self.logging_interval

    def update_iteration_stats(
        self,
        now: float,
        scheduler_outputs: SchedulerOutputs
    ) -> None:
        self.iteration_stats.update(now=now, scheduler_outputs=scheduler_outputs)

    def log_stats(
        self, 
        now: float,
        system_stats: SystemStats
    ) -> List[str]:
        # List of strings to log locally.
        log_strings: List[str] = []

        # Compute metrics and log to loggers.
        for metric in self.metrics:
            metric.compute(stats=Stats(
                system_stats=system_stats, 
                iteration_stats=self.iteration_stats,
            ))
            # To prometheus.
            metric.log()
            
            # Save for local logger.
            if isinstance(metric, GaugeMetric):
                log_strings.append(metric.to_str())

        # Reset iteration stats for next logging window.
        self.iteration_stats.reset()
        self.last_logging_time = now

        return log_strings
