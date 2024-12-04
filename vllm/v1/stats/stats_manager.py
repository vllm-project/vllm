import threading
import time
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Protocol, Tuple

from vllm.config import VllmConfig
from vllm.engine.metrics_types import StatLoggerBase, Stats
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.stats.common import (EngineStatsSnapshot, RequestStats,
                                  RequestStatsUpdate, initialize_stats_loggers)

logger = init_logger(__name__)


class EngineStatsManagerBase(Protocol):

    @abstractmethod
    def add_request(self, request_id: str):
        raise NotImplementedError

    @abstractmethod
    def make_stats(self, engine_core_snapshot: EngineStatsSnapshot) -> Stats:
        raise NotImplementedError

    @abstractmethod
    def log_stats(self, stats: Stats) -> None:
        raise NotImplementedError

    @abstractmethod
    def record_engine_input(self, engine_core_req: EngineCoreRequest):
        raise NotImplementedError

    @abstractmethod
    def record_request_output(self, request_output: RequestOutput):
        raise NotImplementedError

    @abstractmethod
    def record_engine_output(self, engine_core_output: EngineCoreOutput):
        raise NotImplementedError

    @abstractmethod
    def record_decoded(self, request_output: RequestOutput):
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
        # A mapping from request ID to the current request stats.
        self._request_stats: Dict[str, RequestStats] = {}
        # A list of request updates to be merged into the current request stats.
        self._request_updates: List[RequestStatsUpdate] = []

    def add_request(self, request_id: str):
        self._request_stats[request_id] = RequestStats(request_id=request_id)
        self._request_updates.append(
            RequestStatsUpdate(
                type="arrived",
                request_id=request_id,
            ))

    def record_engine_input(self, engine_core_req: EngineCoreRequest):
        self._request_updates.append(
            RequestStatsUpdate(
                type="input_processed",
                request_id=engine_core_req.request_id,
                engine_request=engine_core_req,
            ))

    def record_engine_output(self, engine_core_output: EngineCoreOutput):
        self._request_updates.append(
            RequestStatsUpdate(
                type="decoded",
                request_id=engine_core_output.request_id,
                # So we can compute the latency of each output token.
                token_perf_ts_ns=time.perf_counter_ns(),
                num_new_tokens=len(engine_core_output.new_token_ids),
                finish_reason=engine_core_output.finish_reason,
            ))

    def record_request_output(self, request_output: RequestOutput):
        assert (request_output.outputs
                ), "Request must have at least one output when detokenized."
        self._request_updates.append(
            RequestStatsUpdate(
                type="detokenized",
                request_id=request_output.request_id,
                finish_reason=request_output.outputs[0].finish_reason,
            ))

    @staticmethod
    def _get_num_new_tokens(
        req_stats: RequestStats,
        prev_num_computed_tokens: int,
    ) -> Tuple[int, int]:
        if req_stats.num_prompt_tokens is None:
            return 0, 0

        if prev_num_computed_tokens > req_stats.num_computed_tokens:
            # There's a preemption and we are now backtracking.
            return 0, 0

        cur_num_computed_tokens = req_stats.num_computed_tokens
        # Compute the number of new decode/prefill tokens.
        num_new_tokens = cur_num_computed_tokens - prev_num_computed_tokens

        num_unprefilled_tokens = max(
            0, req_stats.num_prompt_tokens - prev_num_computed_tokens)
        num_new_prefill_tokens = min(num_new_tokens, num_unprefilled_tokens)
        num_new_decode_tokens = num_new_tokens - num_new_prefill_tokens

        return num_new_decode_tokens, num_new_prefill_tokens

    def _build_system_stats(
        self,
        stats: Stats,
        snapshot: EngineStatsSnapshot,
    ) -> None:
        stats.num_running_sys = snapshot.scheduler_stats.num_running_reqs
        stats.num_waiting_sys = snapshot.scheduler_stats.num_waiting_reqs
        stats.gpu_cache_usage_sys = (
            snapshot.scheduler_stats.kv_cache_stats.gpu_cache_usage_sys)
        stats.gpu_prefix_cache_hit_rate = (
            snapshot.scheduler_stats.kv_cache_stats.gpu_prefix_cache_hit_rate)

    def _build_requests_stats(
        self,
        stats: Stats,
        request_updates: Dict[str, List[RequestStatsUpdate]],
    ) -> None:
        """
        Build the request stats from the request updates.

        TODO(rickyx): below stats are currently not yet logged.
        - model_forward_time_requests
        - model_execute_time_requests
        - waiting_lora_adapters
        - running_lora_adapters
        - spec_decode_metrics

        """
        # Update the iteration stats.
        for req_id, req_updates in request_updates.items():
            if req_id not in self._request_stats:
                # This could happen if the request update from the engine core
                # arrives later than a request was finished on the engine
                # frontend.
                continue

            r: RequestStats = self._request_stats[req_id]

            # Intermediate states before the request stats is updated.
            prev_num_computed_tokens = r.num_computed_tokens
            was_scheduled = r.first_scheduled_ts_s is not None
            had_first_token = r.first_token_ts_s is not None
            prev_output_token_latency_s_lst = r.output_token_latency_s_lst
            prev_num_preemption = len(r.preempted_ts_s_lst)

            # Materialize the updates.
            for r_update in req_updates:
                r.update_from(r_update)

            # Compute the new number of decoded and prefill tokens.
            new_num_decoded_tokens, new_num_prefill_tokens = (
                EngineStatsManager._get_num_new_tokens(
                    r, prev_num_computed_tokens))

            stats.num_prompt_tokens_iter += new_num_prefill_tokens
            stats.num_generation_tokens_iter += new_num_decoded_tokens
            stats.num_tokens_iter += (new_num_decoded_tokens +
                                      new_num_prefill_tokens)

            # If any request was preempted.
            if len(r.preempted_ts_s_lst) > prev_num_preemption:
                stats.num_preemption_iter += (len(r.preempted_ts_s_lst) -
                                              prev_num_preemption)

            # If it's first time scheduled.
            if not was_scheduled and r.first_scheduled_ts_s is not None:
                # TODO(rickyx): right now we only account for the duration in
                # the queue before the request is *first* scheduled, but it
                # might be better to also take into account duration when the
                # request was preempted and then rescheduled.
                assert r.queue_duration_s is not None
                stats.time_queue_requests.append(r.queue_duration_s)
                stats.time_in_queue_requests.append(r.queue_duration_s)

            # If first token was generated in the update.
            if not had_first_token and r.first_token_ts_s is not None:
                assert r.first_token_ts_s is not None
                assert r.prefill_latency_s is not None
                assert r.num_prompt_tokens is not None
                stats.time_to_first_tokens_iter.append(r.first_token_ts_s)
                stats.time_prefill_requests.append(r.prefill_latency_s)
                stats.num_prompt_tokens_requests.append(r.num_prompt_tokens)

            # If the request is just finished.
            # NOTE(rickyx): we will prune the finished requests after making the
            # stats so we wouldn't double-log a finished request.
            if r.is_finished:
                assert r.e2e_latency_s is not None
                assert r.inference_latency_s is not None
                assert r.decode_latency_s is not None
                assert r.finish_reason is not None
                assert r.sampling_params is not None
                stats.time_e2e_requests.append(r.e2e_latency_s)
                stats.time_inference_requests.append(r.inference_latency_s)
                stats.time_decode_requests.append(r.decode_latency_s)
                stats.model_forward_time_requests.append(
                    r.model_forward_duration_s)
                stats.model_execute_time_requests.append(
                    r.model_execute_duration_s)
                stats.num_generation_tokens_requests.append(
                    r.num_output_tokens)
                stats.n_requests.append(r.sampling_params.n)
                stats.max_tokens_requests.append(r.sampling_params.max_tokens)
                stats.finished_reason_requests.append(r.finish_reason)

            # Update the new output token stats from the update.
            if len(r.output_token_latency_s_lst) > len(
                    prev_output_token_latency_s_lst):
                stats.time_per_output_tokens_iter.extend(
                    r.output_token_latency_s_lst[
                        len(prev_output_token_latency_s_lst):])
        if stats.num_generation_tokens_requests:
            stats.max_num_generation_tokens_requests.append(
                max(stats.num_generation_tokens_requests))

    def make_stats(self, engine_core_snapshot: EngineStatsSnapshot) -> Stats:
        """
        Make the engine stats from the current snapshot.
        """
        # Computed metrics from this iteration (the current iteration is
        # captured in the new snapshot).
        stats = Stats(now=time.time())
        self._build_system_stats(stats, engine_core_snapshot)

        # Get the latest timestamp from the snapshot.
        latest_snapshot_ts_s = 0
        for req_stats_update in engine_core_snapshot.requests_stats_updates:
            latest_snapshot_ts_s = max(latest_snapshot_ts_s,
                                       req_stats_update.ts_s)

        # Filter out the updates that are older than the latest snapshot.
        updates_iter = []
        updates_remained = []
        for update in self._request_updates:
            if update.ts_s <= latest_snapshot_ts_s:
                updates_iter.append(update)
            else:
                updates_remained.append(update)

        self._request_updates = updates_remained

        # Merge the updates for this iteration.
        req_updates_iter = (engine_core_snapshot.requests_stats_updates +
                            updates_iter)

        # Group by the request id.
        req_updates_by_id: Dict[str,
                                List[RequestStatsUpdate]] = defaultdict(list)
        for r_update in req_updates_iter:
            req_updates_by_id[r_update.request_id].append(r_update)

        # Sort by the timestamp.
        for _, req_updates in req_updates_by_id.items():
            req_updates.sort(key=lambda x: x.ts_s)

        self._build_requests_stats(stats, req_updates_by_id)

        # Prune the finished requests.
        finished_req_ids = set()
        for req_id, r in self._request_stats.items():
            if r.is_finished:
                finished_req_ids.add(req_id)

        for req_id in finished_req_ids:
            del self._request_stats[req_id]

        return stats

    def log_stats(self, stats: Stats) -> None:
        """
        Log the engine stats.
        """

        for stat_logger in self._stat_loggers.values():
            stat_logger.log(stats)


class ThreadSafeEngineStatsManager(EngineStatsManager):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()

    def add_request(self, request_id: str):
        with self._lock:
            super().add_request(request_id)

    def record_engine_input(self, engine_core_req: EngineCoreRequest):
        with self._lock:
            super().record_engine_input(engine_core_req)

    def record_engine_output(self, engine_core_output: EngineCoreOutput):
        with self._lock:
            super().record_engine_output(engine_core_output)

    def record_request_output(self, request_output: RequestOutput):
        with self._lock:
            super().record_request_output(request_output)

    def make_stats(self, engine_core_snapshot: EngineStatsSnapshot) -> Stats:
        with self._lock:
            return super().make_stats(engine_core_snapshot)

    def log_stats(self, stats: Stats) -> None:
        with self._lock:
            super().log_stats(stats)


class NoopEngineStatsManager(EngineStatsManagerBase):

    def __init__(self):
        logger.info("Stats logging is disabled.")

    def add_request(self, request_id: str):
        pass

    def record_engine_input(self, engine_core_req: EngineCoreRequest):
        pass

    def make_stats(self, engine_core_snapshot: EngineStatsSnapshot) -> Stats:
        return Stats()

    def log_stats(self, stats: Stats) -> None:
        pass

    def record_engine_output(self, engine_core_output: EngineCoreOutput):
        pass

    def record_request_output(self, request_output: RequestOutput):
        pass

    def record_decoded(self, request_output: RequestOutput):
        pass
