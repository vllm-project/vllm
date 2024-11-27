from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Literal, Optional

import msgspec
from msgspec import field as msgspec_field

from vllm.config import VllmConfig
from vllm.engine.metrics_types import StatLoggerBase

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

    kv_cache_stats: KVCacheStats = dataclass_field(default_factory=KVCacheStats)


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
    - Finished: a request is finished.
    - Aborted: a request is aborted.
    """

    # Metadata
    request_id: str

    # Timestamp when the request was last updated.
    last_updated_ts_ms: float

    # Timestamp when the request arrived at the llm engine.
    arrival_ts_ms: float

    # Number of tokens sheduled to compute. This includes both the encoder and
    # decoder tokens. Only set when the request is scheduled, and before
    # the request forward pass finishes.
    num_tokens_to_compute: Optional[int] = None

    # Number of tokens cached. When part of the request prefix is cached,
    # this will be set.
    num_tokens_cached: Optional[int] = None

    # The timestamp when the request was first added to the scheduler, waiting
    # in the queue.
    waiting_ts_ms: Optional[float] = None
    # A list of timestamps when the request was scheduled.
    scheduled_ts_ms_lst: List[float] = dataclass_field(default_factory=list)
    # A list of timestamps when the request finished the model forward pass.
    # This is used to calculate the model forward time.
    model_forward_end_ts_ms_lst: List[float] = dataclass_field(
        default_factory=list
    )
    # A list of timestamps when the request finished the model execute function.
    # This is used to calculate the model execute time, model executing includes
    # model forward, block/sync across workers, cpu-gpu sync time and sampling
    # time.
    model_execute_end_ts_ms_lst: List[float] = dataclass_field(
        default_factory=list
    )

    # A list of timestamps when the request was preempted.
    preempted_ts_ms_lst: List[float] = dataclass_field(default_factory=list)
    # Timestamp when the first token was generated.
    first_token_ts_ms: Optional[float] = None
    # Timestamp when the request was finished.
    finished_ts_ms: Optional[float] = None
    # Timestamp when the request was aborted.
    aborted_ts_ms: Optional[float] = None


@dataclass
class EngineStatsSnapshot:
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
        interval as the upper level stats logging (e.g. Prometheus scraping time,
        logging interval, etc.).
        2. Stats collection could happen independently of the request processing
        so even if no requests were processed, stats would still be propogated
        reliably.
    """

    # Timestamp of the snapshot when created.
    created_ts_ms: float
    # Timestamp of the snapshot last updated.
    last_updated_ts_ms: float

    # Snapshot of the scheduler stats.
    scheduler_stats: SchedulerStats = dataclass_field(
        default_factory=SchedulerStats
    )

    # Per request that's active in the engine.
    requests_stats: Dict[str, RequestStats] = dataclass_field(
        default_factory=dict
    )

    # Engine core's queue stats.
    engine_core_process_stats: EngineCoreProcessStats = dataclass_field(
        default_factory=EngineCoreProcessStats
    )

    # TODO(rickyx): Add other components' stats,
    # e.g. model runner/worker and etc.


class RequestStatsUpdate(
    msgspec.Struct, array_like=True, omit_defaults=True, gc=False
):
    """ """

    request_id: str

    update_type: Literal[
        # Timestamp when the request was first added to the scheduler, waiting
        # in the queue.
        "waiting",
        # Timestamp when the request is scheduled by the scheduler.
        "scheduled",
        # Timestamp when the request is preempted by the scheduler.
        "preempted",
        # Timestamp when the request is finished.
        "finished",
        # Timestamp when the request is aborted.
        "aborted",
        # Timestamps when the request finishes the model forward pass.
        "model_forward_end",
        # Timestamps when the request finishes the model execute function.
        # (This includes model forward, block/sync across workers,
        # cpu-gpu sync time and sampling time.)
        "model_execute_end",
    ]

    # Timestamp of the update occurred.
    ts_ms: float

    # Number of tokens cached. Only set when the request is prefilled.
    num_tokens_cached: Optional[int] = None

    # Number of tokens to compute. Only set when the request is scheduled, and
    # before the request forward pass finishes.
    num_tokens_to_compute: Optional[int] = None

    def __post_init__(self):
        if self.update_type == "scheduled":
            assert self.num_tokens_to_compute is not None
            assert self.num_tokens_cached is not None


class EngineStatsUpdate(
    msgspec.Struct, array_like=True, omit_defaults=True, gc=False
):
    """
    An update of the engine's current snapshot of stats.
    """

    # Update timestamp.
    updated_ts_ms: float

    # Updates of the requests' stats.
    request_updates: List[RequestStatsUpdate] = msgspec_field(
        default_factory=list
    )

    # Current state of the engine's scheduler stats.
    scheduler_stats: SchedulerStats = msgspec_field(
        default_factory=SchedulerStats
    )

    # Queue stats (relevant for multi-process architecture)
    engine_core_process_stats: EngineCoreProcessStats = msgspec_field(
        default_factory=EngineCoreProcessStats
    )


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
