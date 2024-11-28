import time
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Dict, List, Optional

import msgspec
from msgspec import field as msgspec_field

from vllm.config import VllmConfig
from vllm.engine.metrics_types import StatLoggerBase
from vllm.v1.request import Request

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
    """Stats associated with the scheduler."""

    # Number of requests currently running.
    num_running_reqs: int = 0
    # Number of requests currently waiting.
    num_waiting_reqs: int = 0

    kv_cache_stats: KVCacheStats = dataclass_field(
        default_factory=KVCacheStats)


@dataclass
class EngineCoreProcessStats:
    """Stats associated with the engine core process."""

    # Number of requests currently in the input queue. None if the engine core
    # is not running in multiprocess mode.
    input_queue_size: Optional[int] = None
    # Number of outputs currently in the output queue. None if the engine core
    # is not running in multiprocess mode.
    output_queue_size: Optional[int] = None


@dataclass
class RequestStats:
    """Stats associated with a request.

    A request would go through the following lifecycles upon arriving
    the llm engine:
    - Arrival: when the request is first added to the llm engine.
    - Waiting: added to the waiting queue of the scheduler in the EngineCore.
    - Scheduled: when the request is scheduled by the scheduler.
    - Model forward ends: when the request forward pass finishes.
    - Model execute ends: when the request finishes the model execute function.
    - [Preempted]: a request could be temporarily unscheduled by the scheduler
                   under contention of resources. This will go back to the
                   waiting queue of the scheduler, and the request will be
                   scheduled again.
    - Finished: a request is finished (aborted or stopped)
    """

    # Metadata
    request_id: str

    # Timestamp when the request was last updated.
    last_updated_ts_ms: Optional[float] = None

    # Timestamp when the request arrived at the llm engine.
    arrival_ts_ms: Optional[float] = None

    # Number of tokens cached. When part of the request prefix is cached,
    # this will be set.
    num_cached_tokens: Optional[int] = None

    # Number of tokens computed.
    num_computed_tokens: Optional[int] = None

    # The timestamp when the request was first added to the scheduler, waiting
    # in the queue.
    waiting_ts_ms: Optional[float] = None

    # A list of timestamps when the request was scheduled to run.
    running_ts_ms_lst: List[float] = dataclass_field(default_factory=list)

    # TODO(rickyx): we need model runner to surface these.
    # # A list of timestamps when the request finished the model forward pass.
    # # This is used to calculate the model forward time.
    # model_forward_end_ts_ms_lst: List[float] = dataclass_field(
    #     default_factory=list
    # )
    # # A list of timestamps when the request finished the model execute
    # # function.
    # # This is used to calculate the model execute time, model executing
    # # includes model forward, block/sync across workers, cpu-gpu sync time
    # # and sampling time.
    # model_execute_end_ts_ms_lst: List[float] = dataclass_field(
    #     default_factory=list
    # )

    # A list of timestamps when the request was preempted at the scheduler.
    preempted_ts_ms_lst: List[float] = dataclass_field(default_factory=list)
    # Timestamp when the first token was generated at the engine core.
    first_token_ts_ms: Optional[float] = None
    # Timestamp when the request was finished at the engine core.
    finished_ts_ms: Optional[float] = None

    def merge(self, other: "RequestStats"):
        assert self.request_id == other.request_id

        self.last_updated_ts_ms = other.last_updated_ts_ms
        if other.num_cached_tokens is not None:
            self.num_cached_tokens = other.num_cached_tokens
        if other.num_computed_tokens is not None:
            self.num_computed_tokens = other.num_computed_tokens
        if other.waiting_ts_ms is not None:
            self.waiting_ts_ms = other.waiting_ts_ms
        if other.running_ts_ms_lst:
            self.running_ts_ms_lst.extend(other.running_ts_ms_lst)
        if other.preempted_ts_ms_lst:
            self.preempted_ts_ms_lst.extend(other.preempted_ts_ms_lst)
        if other.finished_ts_ms is not None:
            self.finished_ts_ms = other.finished_ts_ms
        if other.first_token_ts_ms is not None:
            self.first_token_ts_ms = other.first_token_ts_ms


class EngineStatsSnapshot(msgspec.Struct,
                          array_like=True,
                          omit_defaults=True,
                          gc=False):
    """
    A snapshot of the engine's current stats.
    This represents a snapshot of the current engine core's stats over a
    period of time.

    A snapshot is created periodically (e.g. every 5 seconds) on the frontend of
    the engine, and engine core stats would be gathered from the engine core:
    including the current state of the scheduler, the requests updated since
    the last snapshot.

    This decouples stats collection from actual processing of the requests such
    that:
        1. Stats collection is lightweight and could be aligned with the same
        interval as the upper level stats logging (e.g. Prometheus scraping
        time, logging interval, etc.).
        2. Stats collection could happen independently of the request processing
        so even if no requests were processed, stats would still be propagated
        reliably.
    """

    # Timestamp of the snapshot last updated. None if the snapshot is just
    # created.
    last_updated_ts_ms: Optional[float] = None

    # Timestamp of the snapshot when created.
    created_ts_ms: float = msgspec_field(
        default_factory=lambda: ms_to_s(time.time()))

    # Snapshot of the scheduler stats.
    scheduler_stats: SchedulerStats = msgspec_field(
        default_factory=SchedulerStats)

    # Per request stats.
    requests_stats: Dict[str, RequestStats] = msgspec_field(
        default_factory=lambda: defaultdict(RequestStats))

    # Engine core's queue stats.
    engine_core_process_stats: EngineCoreProcessStats = msgspec_field(
        default_factory=EngineCoreProcessStats)

    # TODO(rickyx): Add other components' stats,
    # e.g. model runner/worker and etc.

    def _get_or_create_request_stats(self, request_id: str) -> RequestStats:
        if request_id not in self.requests_stats:
            self.requests_stats[request_id] = RequestStats(
                request_id=request_id, )
        return self.requests_stats[request_id]

    def record_arrival_request(self, request_id: str):
        assert request_id not in self.requests_stats
        now_ms = ms_to_s(time.time())
        self.last_updated_ts_ms = now_ms
        request_stats = self._get_or_create_request_stats(request_id)
        request_stats.arrival_ts_ms = now_ms
        request_stats.last_updated_ts_ms = now_ms

    def record_running_request(
        self,
        request: Request,
        num_computed_tokens: int,
        num_cached_tokens: Optional[int],
    ):
        now_ms = ms_to_s(time.time())
        self.last_updated_ts_ms = now_ms
        request_stats = self._get_or_create_request_stats(request.request_id)
        request_stats.running_ts_ms_lst.append(now_ms)
        request_stats.num_computed_tokens = num_computed_tokens
        request_stats.last_updated_ts_ms = now_ms
        if num_cached_tokens is not None:
            request_stats.num_cached_tokens = num_cached_tokens

    def record_finished_request(self, request: Request):
        now_ms = ms_to_s(time.time())
        self.last_updated_ts_ms = now_ms
        request_stats = self._get_or_create_request_stats(request.request_id)
        request_stats.finished_ts_ms = now_ms
        request_stats.last_updated_ts_ms = now_ms

    def record_waiting_request(self, request: Request):
        now_ms = ms_to_s(time.time())
        self.last_updated_ts_ms = now_ms
        request_stats = self._get_or_create_request_stats(request.request_id)
        request_stats.waiting_ts_ms = now_ms
        request_stats.last_updated_ts_ms = now_ms

    def record_preempted_request(self, request: Request):
        now_ms = ms_to_s(time.time())
        self.last_updated_ts_ms = now_ms
        request_stats = self._get_or_create_request_stats(request.request_id)
        request_stats.preempted_ts_ms_lst.append(now_ms)
        request_stats.last_updated_ts_ms = now_ms

    def record_first_token_ts_ms(self, request: Request):
        now_ms = ms_to_s(time.time())
        self.last_updated_ts_ms = now_ms
        request_stats = self._get_or_create_request_stats(request.request_id)
        request_stats.first_token_ts_ms = now_ms
        request_stats.last_updated_ts_ms = now_ms

    def merge(self, snapshot: "EngineStatsSnapshot"):
        self.last_updated_ts_ms = snapshot.last_updated_ts_ms
        for request_id, target in snapshot.requests_stats.items():
            assert request_id in self.requests_stats
            source = self._get_or_create_request_stats(request_id)
            source.merge(target)

        # Just copy over the scheduler stats.
        self.scheduler_stats = snapshot.scheduler_stats

        # Just copy over the engine core process stats.
        self.engine_core_process_stats = snapshot.engine_core_process_stats

    def prune(self):
        # Prune the requests stats that are finished.
        requests_to_prune = [
            request_stats for request_stats in self.requests_stats.values()
            if request_stats.finished_ts_ms is not None
        ]
        for request_stats in requests_to_prune:
            del self.requests_stats[request_stats.request_id]


def ms_to_s(ms: float) -> float:
    return ms / 1000.0


def s_to_ms(s: float) -> float:
    return s * 1000.0


def initialize_stats_loggers(config: VllmConfig) -> Dict[str, StatLoggerBase]:
    """
    Initialize the stats loggers.
    """
    from vllm.engine.metrics import LoggingStatLogger

    stat_loggers = {
        "logging":
        LoggingStatLogger(local_interval=_LOCAL_LOGGING_INTERVAL_SEC),
        # TODO(rickyx): Add prometheus stats logger.
        # "prometheus": PrometheusStatLogger(
        #     local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
        #     labels=dict(model_name=self.model_config.served_model_name),
        #     max_model_len=self.model_config.max_model_len,
        # ),
    }
    return stat_loggers
