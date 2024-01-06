import time
import logging
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union, List, Dict
from aioprometheus import Counter, Gauge, Histogram

labels = {}


def add_global_metrics_labels(**kwargs):
    labels.update(kwargs)


@dataclass
class Stats:
    # System stats.
    num_running: int
    num_waiting: int
    num_swapped: int
    gpu_cache_usage: float
    cpu_cache_usage: float

    # Raw stats from last model iteration.
    num_prompt_tokens: int
    num_generation_tokens: int
    time_to_first_tokens: List[float]
    time_per_output_tokens: List[float]
    time_e2e_requests: List[float]


class TrackedStats:
    """Data to compute metrics for logging to StdOut"""

    def __init__(self) -> None:
        self.stats: List[Union[int, float]] = []
        self.last_reset_time = time.monotonic()

    def append(self, stat: Union[int, float]) -> None:
        self.stats.append(stat)

    def reset(self, now: float) -> None:
        self.stats = []
        self.last_reset_time = now

    def get_throughput(self, now: float) -> float:
        return float(np.sum(self.stats) / (now - self.last_reset_time))

    def get_mean(self) -> float:
        return float(np.mean(self.stats)) if len(self.stats) > 0 else 0

    def get_last(self) -> Union[int, float]:
        return self.stats[-1] if len(self.stats) > 0 else 0


class PrometheusMetric(ABC):
    """Log Stats to Prometheus and TrackStats"""

    def __init__(self,
                 attr: str,
                 template: str,
                 log_local: bool = True) -> None:
        # Attribute to get correct attibute from Stats.
        self.attr = attr
        # String template with $metric for local logging.
        self.template = template
        # Tracked Stats for Logging Locally
        self.tracked_stats = TrackedStats()
        # Whether this metric should log locally.
        self.log_local = log_local

    @abstractmethod
    # Log to Prometheus and local TrackedStats.
    def log(self, stats: Stats, labels: Dict[str, str], now: float) -> None:
        raise NotImplementedError

    @abstractmethod
    # Compute metric from local TrackedStats.
    def _compute_metric(self, now: float) -> Union[int, float]:
        raise NotImplementedError

    # Compute metric and convert to string.
    def get_str(self, now: float) -> str:
        return self.template.format(self._compute_metric(now))

    # Track stat for local logging.
    def track_stat(self, stat: Union[int, float]) -> None:
        if self.log_local:
            self.tracked_stats.append(stat)

    # Reset tracked stats for next local logging interval.
    def reset_stats(self, now: float) -> None:
        self.tracked_stats.reset(now)


class CounterMetric(PrometheusMetric):
    """CounterMetric measures thoughput (# / time)."""

    def __init__(self, counter: Counter, *args, **kwargs) -> None:
        self.counter = counter
        super().__init__(*args, **kwargs)

    def log(self, stats: Stats, labels: Dict[str, str], now: float) -> None:
        stat = getattr(stats, self.attr, 0)
        self.counter.add(labels, stat)
        self.track_stat(stat)

    def _compute_metric(self, now: float) -> Union[int, float]:
        return self.tracked_stats.get_throughput(now)


class GaugeMetric(PrometheusMetric):
    """GaugeMetric measures a metric at a moment in time."""

    def __init__(self, gauge: Gauge, *args, **kwargs) -> None:
        self.gauge = gauge
        super().__init__(*args, **kwargs)

    def log(self, stats: Stats, labels: Dict[str, str], now: float) -> None:
        stat = getattr(stats, self.attr, 0)
        self.gauge.set(labels, stat)
        self.track_stat(stat)

    def _compute_metric(self, now: float) -> Union[int, float]:
        return self.tracked_stats.get_last()


class HistogramMetric(PrometheusMetric):
    """HistogramMetric measures an average of many requests."""

    def __init__(self, histogram: Histogram, *args, **kwargs) -> None:
        self.histogram = histogram
        super().__init__(*args, **kwargs)

    def log(self, stats: Stats, labels: Dict[str, str], now: float) -> None:
        for stat in getattr(stats, self.attr, []):
            self.histogram.observe(labels, stat)
            self.track_stat(stat)

    def _compute_metric(self, now: float) -> Union[int, float]:
        return self.tracked_stats.get_mean()


class MetricsLogger:
    """Used LLMEngine to log metrics to Promethus and Stdout."""

    def __init__(self, metrics: List[PrometheusMetric],
                 local_logger: logging.Logger, local_interval: float) -> None:
        self.metrics = metrics
        self.local_logger = local_logger
        self.last_local_log = time.monotonic()
        self.local_interval = local_interval

    def _interval_elasped(self, now: float):
        return now - self.last_local_log > self.local_interval

    def log(self, now: float, stats: Stats) -> None:
        # Log locally if local_interval sec elapsed.
        log_local = self._interval_elasped(now)
        log_str = ""

        for metric in self.metrics:
            # Log to Prometheus.
            metric.log(labels=labels, stats=stats, now=now)

            # Log to StdOut and reset local tracked stats.
            if log_local and metric.log_local:
                log_str += f", {metric.get_str(now=now)}"
                metric.reset_stats(now=now)

        if log_local:
            self.local_logger.info(log_str)
            self.last_local_log = now
