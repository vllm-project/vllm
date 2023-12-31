from aioprometheus import Counter, Gauge, Histogram
from abc import ABC
from dataclasses import dataclass
from typing import Union, Dict, List
from vllm.core.scheduler import SchedulerOutputs

@dataclass
class SystemStats:
    """System Stats hold a snapshot of the system state at a given time."""
    num_total_gpu_blocks: int
    num_total_cpu_blocks: int
    num_free_gpu_blocks: int
    num_free_cpu_blocks: int
    num_running: int
    num_waiting: int
    num_swapped: int

class IterationStats:
    """IterationStats holds iteration level stats for logging_interval window."""
    def __init__(self):
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []
        self.time_to_first_token: List[float] = []
        self.inter_token_latency: List[float] = []

    def update(self, now: float, scheduler_outputs: SchedulerOutputs) -> None:
        """Updates the Tracked Stats based on the SchedulerOutput."""
        # Update iteration timings for each SequenceGroup.
        timings = [
            seq_group.update_latency_timing(
                now=now, 
                prompt_run=scheduler_outputs.prompt_run
            ) for seq_group in scheduler_outputs.scheduled_seq_groups
        ]

        # Update TrackedStats.
        if scheduler_outputs.prompt_run:
            # Prefill Related Stats.
            self.num_prompt_tokens.append(scheduler_outputs.num_batched_tokens)
            self.time_to_first_token.extend(timings)
        else:
            # Decode Related Stats.
            self.num_generation_tokens.append(scheduler_outputs.num_batched_tokens)
            self.inter_token_latency.extend(timings)

    def reset(self) -> None:
        """Reset stats for next logging window."""
        self.num_prompt_tokens = []
        self.num_generation_tokens = []
        self.time_to_first_token = []
        self.inter_token_latency = []

@dataclass
class Stats:
    system_stats: SystemStats
    iteration_stats: IterationStats

class PrometheusMetric(ABC):
    can_log_local: bool = False

    """Metric holds a Prometheus Metric and logic for converting Stats --> Metric"""    
    def log(self) -> None:
        """Push metric to Prometheus client."""
        raise NotImplementedError

    def compute(self, stats: Stats) -> None:
        """Compute metric based on stats."""
        raise NotImplementedError

class CounterMetric(PrometheusMetric):
    def __init__(self, prometheus_metric: Counter, labels: Dict[str,str]) -> None:
        super().__init__()

        self.counter = prometheus_metric
        self.metric: Union[float, int] = 0
        self.labels = labels
    
    def log(self) -> None:
        # Increment counter by N if "something happend" (metric > 0).
        if self.metric > 0:
            self.counter.add(self.labels, self.metric)
        self.metric = 0

class GaugeMetric(PrometheusMetric):
    def __init__(self, prometheus_metric: Gauge, labels: Dict[str,str]) -> None:
        super().__init__()

        self.gauge = prometheus_metric
        self.metric: Union[float, int] = 0
        self.labels = labels
        self.can_log_local = True
    
    def log(self) -> None:
        # Set gauge.
        self.gauge.set(self.labels, self.metric)

    def to_str(self) -> str:
        raise NotImplementedError

class HistogramMetric(PrometheusMetric):
    def __init__(self, prometheus_metric: Histogram, labels: Dict[str,str]) -> None:
        super().__init__()

        self.histogram = prometheus_metric
        self.metrics: List[Union[float, int]] = []
        self.labels = labels
    
    def log(self) -> None:
        # Log each metric.
        for metric in self.metrics:
            self.histogram.observe(self.labels, metric)
