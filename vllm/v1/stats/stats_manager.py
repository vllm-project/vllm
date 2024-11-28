from abc import abstractmethod
from typing import Dict, Optional, Protocol

from vllm.config import VllmConfig
from vllm.engine.metrics_types import StatLoggerBase, Stats
from vllm.logger import init_logger
from vllm.v1.stats.common import (EngineStatsSnapshot,
                                  initialize_stats_loggers, ms_to_s)

logger = init_logger(__name__)


class EngineStatsManagerBase(Protocol):

    @abstractmethod
    def add_request(self, request_id: str):
        raise NotImplementedError

    @abstractmethod
    def update_snapshot(self, other_snapshot: EngineStatsSnapshot):
        raise NotImplementedError

    @abstractmethod
    def make_stats(self) -> Stats:
        raise NotImplementedError

    @abstractmethod
    def log_stats(self, stats: Stats) -> None:
        raise NotImplementedError


class EngineStatsManager:
    """
    This is responsible for aggregating EngineStatsSnapshot from
    EngineStatsAgent, and produce a snapshot of the engine's stats.

    This manager should be owned by the engine (i.e. AsyncLLM or LLMEngine),
    and the engine should be responsible for getting the snapshot updates from
    the EngineStatsAgent.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ):
        self._vllm_config = vllm_config

        self._stat_loggers = stat_loggers or initialize_stats_loggers(
            vllm_config)
        logger.info("Logging stats to: %s", list(self._stat_loggers.keys()))

        # Represents the most recent snapshot of the engine's stats.
        self._cur_snapshot = None

    def add_request(self, request_id: str):
        """
        Add a request to the stats snapshot.
        """
        if self._cur_snapshot is None:
            self._cur_snapshot = EngineStatsSnapshot()
        self._cur_snapshot.record_arrival_request(request_id)

    def update_snapshot(self, other_snapshot: EngineStatsSnapshot):
        """
        Update the current snapshot with the new snapshot.
        """
        self._cur_snapshot.merge(other_snapshot)

    def make_stats(self) -> Stats:
        """
        Make the engine stats from the current snapshot.
        """
        # TODO(rickyx): calculate the remaining stats.
        # Make the States
        snapshot = self._cur_snapshot
        stats = Stats(
            now=ms_to_s(snapshot.last_updated_ts_ms),
            # System stats
            #   Scheduler State
            num_running_sys=snapshot.scheduler_stats.num_running_reqs,
            num_swapped_sys=0,
            num_waiting_sys=snapshot.scheduler_stats.num_waiting_reqs,
            #   KV Cache Usage in %
            gpu_cache_usage_sys=snapshot.scheduler_stats.kv_cache_stats.
            gpu_cache_usage_sys,
            cpu_cache_usage_sys=snapshot.scheduler_stats.kv_cache_stats.
            cpu_cache_usage_sys,
            #   Prefix Cache Hit Rate
            cpu_prefix_cache_hit_rate=snapshot.scheduler_stats.kv_cache_stats.
            cpu_prefix_cache_hit_rate,
            gpu_prefix_cache_hit_rate=snapshot.scheduler_stats.kv_cache_stats.
            gpu_prefix_cache_hit_rate,
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
        print(stats)
        return stats

    def log_stats(self, stats: Stats) -> None:
        """
        Log the engine stats.
        """

        for stat_logger in self._stat_loggers.values():
            stat_logger.log(stats)


class NoopEngineStatsManager(EngineStatsManagerBase):

    def __init__(self):
        logger.info("Stats logging is disabled.")

    def add_request(self, request_id: str):
        pass

    def update_snapshot(self, other_snapshot: EngineStatsSnapshot):
        pass

    def log_stats(self) -> None:
        pass
