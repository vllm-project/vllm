import time
from logging import Logger
from typing import Optional, List
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.metrics.metrics_registry import METRIC_REGISTRY
from vllm.engine.metrics.metrics_utils import SystemStats, IterationStats, PrometheusMetric

labels = {}
            
def add_global_metrics_labels(**kwargs):
    labels.update(kwargs)

class MetricLogger:
    def __init__(
        self,
        local_logger: Logger,
        logging_interval_sec: int,
    ):
        self.local_logger = local_logger
        self.logging_interval_sec = logging_interval_sec
        self.last_logging_time = 0.0
        
        # Iteration level stats that we save to log periodically.
        self.iteration_stats = IterationStats(
            logging_interval_sec = self.logging_interval_sec
        )

        # Gauge Metrics from Registry.
        self.metrics: List[PrometheusMetric] = METRIC_REGISTRY

    def should_log(self, now: float):
        return now - self.last_logging_time >= self.logging_interval_sec

    def log_stats(
        self,
        now: float,
        scheduler_outputs: SchedulerOutputs,
        system_stats: Optional[SystemStats],
    ) -> None:
        # Update the logged iteration stats.
        self.iteration_stats.update(now=now, scheduler_outputs=scheduler_outputs)

        # Actually log every logging_interval seconds. 
        if not self.should_log(now=now):
            return
        assert system_stats is not None, "system_stats should not be none when should_log"

        # Update metrics and log to loggers.
        for metric in self.metrics:
            metric.update(
                now=now,
                system_stats=system_stats, 
                iteration_stats=self.iteration_stats,
            )
            # To prometheus.
            metric.log()
            # To stdout.
            if metric.should_local_log:
                self.local_logger(metric.to_str())

        # Reset iteration level data for next logging window.
        self.iteration_stats.reset()
        self.last_logging_time = now
