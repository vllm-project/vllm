import time
from typing import Dict

from vllm.engine.metrics_types import Stats
from vllm.v1.stats.common import (EngineStatsUpdate, RequestStats,
                                  RequestStatsUpdate, ms_to_s, s_to_ms)


class EngineStatsManager:
    """
    This is responsible for aggregating EngineStatsUpdate from
    EngineStatsAgent, and produce a snapshot of the engine's stats.

    This manager should be owned by the engine (i.e. AsyncLLM or LLMEngine),
    and the engine should be responsible for getting the snapshot updates from
    the EngineStatsAgent.
    """

    def __init__(self):
        # A mapping from request ID to the request's stats.
        self._requests_stats: Dict[str, RequestStats] = {}

    def add_request(self, request_id: str):
        """
        Add a request to the stats snapshot.
        """
        assert request_id not in self._requests_stats

        self._requests_stats[request_id] = RequestStats(
            request_id=request_id,
            last_updated_ts_ms=s_to_ms(time.time()),
            arrival_ts_ms=s_to_ms(time.time()),
        )

    def _remove_request(self, request_id: str):
        # Remove a request from tracking when it finishes or aborts.
        del self._requests_stats[request_id]

    def _update_request(self, req_stat: RequestStats,
                        update: RequestStatsUpdate):
        req_stat.last_updated_ts_ms = update.ts_ms
        if update.update_type == "waiting":
            req_stat.waiting_ts_ms = update.ts_ms
        elif update.update_type == "scheduled":
            req_stat.num_tokens_to_compute = update.num_tokens_to_compute
            req_stat.num_tokens_cached = update.num_tokens_cached
            req_stat.scheduled_ts_ms_lst.append(update.ts_ms)
        elif update.update_type == "finished":
            req_stat.finished_ts_ms = update.ts_ms
        elif update.update_type == "aborted":
            req_stat.aborted_ts_ms = update.ts_ms
        elif update.update_type == "model_forward_end":
            req_stat.model_forward_end_ts_ms_lst.append(update.ts_ms)
        elif update.update_type == "model_execute_end":
            req_stat.model_execute_end_ts_ms_lst.append(update.ts_ms)
        else:
            raise ValueError(f"Unknown update type: {update.update_type}")

    def make_engine_stats(self, update: EngineStatsUpdate) -> Stats:
        """Finalize a snapshot of the engine's current stats: creating
        a new snapshot and returning the current snapshot."""

        # Aggregate the requests stats updates.
        print(f"make_engine_stats: {update}")
        for req_update in update.request_updates:
            assert req_update.request_id in self._requests_stats
            req_stat = self._requests_stats[req_update.request_id]
            self._update_request(req_stat, req_update)

        # TODO(rickyx): calculate the remaining stats.

        # Make the States
        stats = Stats(
            now=ms_to_s(update.updated_ts_ms),
            # System stats
            #   Scheduler State
            num_running_sys=update.scheduler_stats.num_running_reqs,
            num_swapped_sys=0,
            num_waiting_sys=update.scheduler_stats.num_waiting_reqs,
            #   KV Cache Usage in %
            gpu_cache_usage_sys=update.scheduler_stats.kv_cache_stats.
            gpu_cache_usage_sys,
            cpu_cache_usage_sys=update.scheduler_stats.kv_cache_stats.
            cpu_cache_usage_sys,
            #   Prefix Cache Hit Rate
            cpu_prefix_cache_hit_rate=update.scheduler_stats.kv_cache_stats.
            cpu_prefix_cache_hit_rate,
            gpu_prefix_cache_hit_rate=update.scheduler_stats.kv_cache_stats.
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

        # Update the requests stats for finished/aborted requests.
        for req_stat in self._requests_stats.values():
            # Remove the request from tracking if it's finished or aborted.
            if (req_stat.finished_ts_ms is not None
                    or req_stat.aborted_ts_ms is not None):
                self._remove_request(req_stat.request_id)

        return stats
