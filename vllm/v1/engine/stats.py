from dataclasses import dataclass, field
import time
from typing import Dict

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.metrics_types import StatLoggerBase, Stats

_LOCAL_LOGGING_INTERVAL_SEC = 5  # TODO(rickyx): Make this configurable.


@dataclass
class KVCacheStats:
    #   KV Cache Usage in %
    gpu_cache_usage_sys: float = 0.0
    cpu_cache_usage_sys: float = 0.0
    #   Prefix caching block hit rate
    cpu_prefix_cache_hit_rate: float = 0.0
    gpu_prefix_cache_hit_rate: float = 0.0


@dataclass
class SchedulerStats:
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0

    kv_cache_stats: KVCacheStats = field(default_factory=KVCacheStats)


@dataclass
class EngineCoreStats:
    """
    A stateless snapshot of the engine's current stats.
    """

    # Current queue size.
    # TODO(rickyx): Add this.
    scheduler_stats: SchedulerStats = field(default_factory=SchedulerStats)


def make_stats(engine_core_stats: EngineCoreStats) -> Stats:
    # TODO(rickyx): we should deprecate those V0 only metrics when we migrate.
    now = time.time()
    return Stats(
        now=now,
        # System stats
        #   Scheduler State
        num_running_sys=engine_core_stats.scheduler_stats.num_running_reqs,
        num_swapped_sys=0,
        num_waiting_sys=engine_core_stats.scheduler_stats.num_waiting_reqs,
        #   KV Cache Usage in %
        gpu_cache_usage_sys=engine_core_stats.scheduler_stats.kv_cache_stats.gpu_cache_usage_sys,
        cpu_cache_usage_sys=engine_core_stats.scheduler_stats.kv_cache_stats.cpu_cache_usage_sys,
        #   Prefix Cache Hit Rate
        cpu_prefix_cache_hit_rate=engine_core_stats.scheduler_stats.kv_cache_stats.cpu_prefix_cache_hit_rate,
        gpu_prefix_cache_hit_rate=engine_core_stats.scheduler_stats.kv_cache_stats.gpu_prefix_cache_hit_rate,
        # Iteration stats
        num_prompt_tokens_iter=0,
        num_generation_tokens_iter=0,
        num_tokens_iter=0,
        time_to_first_tokens_iter=[],
        time_per_output_tokens_iter=[],
        spec_decode_metrics=None,
        num_preemption_iter=0,
        # Request stats
        #   Latency
        time_e2e_requests=[],
        time_queue_requests=[],
        time_inference_requests=[],
        time_prefill_requests=[],
        time_decode_requests=[],
        time_in_queue_requests=[],
        model_forward_time_requests=[],
        model_execute_time_requests=[],
        #   Metadata
        num_prompt_tokens_requests=0,
        num_generation_tokens_requests=0,
        max_num_generation_tokens_requests=0,
        n_requests=0,
        max_tokens_requests=0,
        finished_reason_requests=[],
        max_lora="0",
        waiting_lora_adapters=[],
        running_lora_adapters=[],
    )


def initialize_stats_loggers(config: VllmConfig) -> Dict[str, StatLoggerBase]:
    """
    Initialize the stats loggers.
    """
    from vllm.engine.metrics import LoggingStatLogger

    stat_loggers = {
        "logging": LoggingStatLogger(
            local_interval=_LOCAL_LOGGING_INTERVAL_SEC
        ),
        # TODO(rickyx): Add prometheus stats logger.
        # "prometheus": PrometheusStatLogger(
        #     local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
        #     labels=dict(model_name=self.model_config.served_model_name),
        #     max_model_len=self.model_config.max_model_len,
        # ),
    }
    return stat_loggers
