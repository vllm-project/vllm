import time
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Dict, List, Literal, Optional, Union

import msgspec
from msgspec import field as msgspec_field

from vllm.config import VllmConfig
from vllm.engine.metrics_types import StatLoggerBase
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

_LOCAL_LOGGING_INTERVAL_SEC = 1  # TODO(rickyx): Make this configurable.


def ns_to_s(ns: int) -> float:
    return ns / 1e9


@dataclass
class KVCacheStats:
    #   KV Cache Usage in %
    gpu_cache_usage_sys: float = 0.0
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
    - Inputs processed: when the input processor is completed.
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

    ############################################################
    # Metadata
    ############################################################
    request_id: str
    # The original request object from the engine core.
    engine_request: Optional[EngineCoreRequest] = None

    ############################################################
    # Metrics and Stats
    ############################################################
    # Timestamp when the request was last updated.
    last_updated_ts_s: Optional[float] = None

    # Timestamp when the request arrived at the llm engine.
    arrival_ts_s: Optional[float] = None

    # Number of tokens cached. When part of the request prefix is cached,
    # this will be set.
    num_cached_tokens: int = 0

    # Number of tokens computed.
    num_computed_tokens: int = 0

    # The timestamp when the request was first added to the scheduler, waiting
    # in the queue.
    waiting_ts_s: Optional[float] = None

    # When the input processor is completed.
    input_processor_end_ts_s: Optional[float] = None

    # A sorted list of timestamps when the request was scheduled to run.
    running_ts_s_lst: List[float] = dataclass_field(default_factory=list)

    # A sorted list of perf counter timestamps for each output token.
    output_token_perf_counter_ns_lst: List[int] = dataclass_field(
        default_factory=list)

    # First token's timestamp.
    first_token_ts_s: Optional[float] = None

    # TODO(rickyx): we need model runner to surface these.
    model_forward_duration_s: float = 0.0
    # Includes model forward, block/sync across workers, cpu-gpu sync time
    # and sampling time.
    model_execute_duration_s: float = 0.0

    # A sorted list of timestamps when the request was preempted at the
    # scheduler.
    preempted_ts_s_lst: List[float] = dataclass_field(default_factory=list)

    # Timestamp when the request was finished at the engine core.
    finished_ts_s: Optional[float] = None

    # Finish reason.
    finish_reason: Optional[str] = None

    ############################################################
    # Derived properties.
    ############################################################
    @property
    def num_prompt_tokens(self) -> Optional[int]:
        return (len(self.engine_request.prompt_token_ids)
                if self.engine_request else None)

    @property
    def first_scheduled_ts_s(self) -> Optional[float]:
        return self.running_ts_s_lst[0] if self.running_ts_s_lst else None

    @property
    def e2e_latency_s(self) -> Optional[float]:
        if self.finished_ts_s is None or self.arrival_ts_s is None:
            return None
        assert self.finished_ts_s >= self.arrival_ts_s
        return self.finished_ts_s - self.arrival_ts_s

    @property
    def queue_duration_s(self) -> Optional[float]:
        if self.first_scheduled_ts_s is None or self.arrival_ts_s is None:
            return None
        assert self.first_scheduled_ts_s >= self.arrival_ts_s
        return self.first_scheduled_ts_s - self.arrival_ts_s

    @property
    def inference_latency_s(self) -> Optional[float]:
        if self.e2e_latency_s is None or self.queue_duration_s is None:
            return None
        assert self.e2e_latency_s >= self.queue_duration_s
        return self.e2e_latency_s - self.queue_duration_s

    @property
    def first_token_latency_s(self) -> Optional[float]:
        if self.first_token_ts_s is None or self.arrival_ts_s is None:
            return None
        assert self.first_token_ts_s >= self.arrival_ts_s
        return self.first_token_ts_s - self.arrival_ts_s

    @property
    def prefill_latency_s(self) -> Optional[float]:
        if self.first_token_ts_s is None or self.first_scheduled_ts_s is None:
            return None
        assert self.first_token_ts_s >= self.first_scheduled_ts_s
        return self.first_token_ts_s - self.first_scheduled_ts_s

    @property
    def decode_latency_s(self) -> Optional[float]:
        if self.e2e_latency_s is None or self.first_token_latency_s is None:
            return None
        assert self.e2e_latency_s >= self.first_token_latency_s
        return self.e2e_latency_s - self.first_token_latency_s

    @property
    def output_token_latency_s_lst(self) -> List[float]:
        if len(self.output_token_perf_counter_ns_lst) == 0:
            return []
        latency_s_lst = []
        for i in range(1, len(self.output_token_perf_counter_ns_lst)):
            assert (self.output_token_perf_counter_ns_lst[i] >=
                    self.output_token_perf_counter_ns_lst[i - 1])
            latency_s = ns_to_s(self.output_token_perf_counter_ns_lst[i] -
                                self.output_token_perf_counter_ns_lst[i - 1])
            latency_s_lst.append(latency_s)
        return latency_s_lst

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_perf_counter_ns_lst)

    @property
    def is_finished(self) -> bool:
        return self.finished_ts_s is not None

    @property
    def sampling_params(self) -> Optional[SamplingParams]:
        return (self.engine_request.sampling_params
                if self.engine_request else None)

    def update_from(self, update: "RequestStatsUpdate"):
        ts = update.ts_s
        self.last_updated_ts_s = ts
        if update.type == "arrived":
            self.arrival_ts_s = ts
        elif update.type == "input_processed":
            self.input_processor_end_ts_s = ts
            self.engine_request = update.engine_request
        elif update.type == "queued":
            self.waiting_ts_s = ts
        elif update.type == "running":
            assert (update.was_running is not None
                    and update.num_computed_tokens is not None)
            self._record_running(
                update.num_computed_tokens,
                update.was_running,
                ts,
                update.num_cached_tokens,
            )
        elif update.type == "preempted":
            self._reset_for_preemption(ts)
        elif update.type == "decoded":
            assert update.token_perf_ts_ns is not None
            self._record_engine_output(
                ts,
                update.token_perf_ts_ns,
                update.num_new_tokens,
                update.finish_reason,
            )
        elif update.type == "detokenized":
            self._record_request_output(update.finish_reason, ts)
        else:
            raise ValueError(f"Unknown update type: {update.type}")

    def _record_running(
        self,
        num_computed_tokens: int,
        was_running: bool,
        ts_s: float,
        num_cached_tokens: Optional[int] = None,
    ):
        if not was_running:
            # Was preempted or newly run.
            self.running_ts_s_lst.append(ts_s)
            self.num_cached_tokens = num_cached_tokens

        self.num_computed_tokens = num_computed_tokens

    def _record_engine_output(
        self,
        ts_s: float,
        perf_ts_ns: int,
        num_new_tokens: int,
        finish_reason: Optional[str],
    ):
        # Handle the output token timestamps.
        if len(self.output_token_perf_counter_ns_lst) > 0:
            self._check_sorted(self.output_token_perf_counter_ns_lst,
                               perf_ts_ns)

        # Update if first output token is generated.
        if len(self.output_token_perf_counter_ns_lst) == 0:
            self.first_token_ts_s = ts_s
            assert self.first_scheduled_ts_s is not None

        self.output_token_perf_counter_ns_lst.extend([perf_ts_ns] *
                                                     num_new_tokens)

        # Update if the request is finished.
        if finish_reason is not None:
            self.finished_ts_s = ts_s
            self.finish_reason = finish_reason

    def _record_request_output(self, finish_reason: Optional[str],
                               ts_s: float):
        if finish_reason is not None and self.finished_ts_s is None:
            self.finished_ts_s = ts_s
            self.finish_reason = finish_reason

    def _reset_for_preemption(self, ts_s: float):
        self.preempted_ts_s_lst.append(ts_s)
        self.num_computed_tokens = 0
        self.num_cached_tokens = 0
        self.output_token_perf_counter_ns_lst.clear()
        self.model_forward_duration_s = 0.0
        self.model_execute_duration_s = 0.0
        self.first_token_ts_s = None

    @staticmethod
    def _check_sorted(lst: List[float], elem_or_sorted_lst: Union[List[float],
                                                                  float]):
        """
        Check if the list is sorted and the last element is less than the
        first element of the list to be appended or extended.

        It's assumed that the when `elem_or_sorted_lst` is a list, it's sorted.
        """
        if not lst:
            return

        if isinstance(elem_or_sorted_lst, list):
            assert lst[-1] <= elem_or_sorted_lst[0], (
                f"Output token timestamps must be sorted: {lst[-1]} and "
                f"{elem_or_sorted_lst[0]}")
        else:
            assert lst[-1] <= elem_or_sorted_lst, (
                f"Output token timestamps must be sorted: {lst[-1]} and "
                f"{elem_or_sorted_lst}")


class RequestStatsUpdate(msgspec.Struct,
                         array_like=True,
                         omit_defaults=True,
                         gc=False):
    """
    An update to the request stats.

    NOTE:
    - We should try to keep the size of this struct minimal by avoiding
      keeping references to additional objects if not necessary, especially
      when the referenced object could have been GCed already if not for
      this reference (examples include per decoded token RequestOutput,
      EngineCoreOutput, etc.).
    """

    request_id: str

    type: Literal[
        # Request arrived at the engine frontend.
        "arrived",
        # Input processed by the input processor.
        "input_processed",
        # Queued on the engine core.
        "queued",
        # Scheduled running by the scheduler.
        "running",
        # Preempted by the scheduler.
        "preempted",
        # Token decoded by the engine.
        "decoded",
        # Token detokenized by the detokenizer.
        "detokenized", ]

    # Timestamp when the update is recorded. This is used to record time
    # intervals between events.
    ts_s: float = msgspec_field(default_factory=lambda: time.monotonic())

    # Metadata associated with the update.
    # For input_processed.
    engine_request: Optional[EngineCoreRequest] = None

    # For running.
    # If the request was already running, we don't record the timestamp.
    was_running: Optional[bool] = None
    # Number of tokens computed.
    num_computed_tokens: Optional[int] = None
    # Number of cached tokens.
    num_cached_tokens: Optional[int] = None

    # For decoded.
    # The perf counter timestamp for each output token.
    token_perf_ts_ns: Optional[int] = None
    # The number of output tokens.
    num_new_tokens: Optional[int] = None

    # For both detokenized and decoded.
    # Finished reason.
    finish_reason: Optional[str] = None


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

    # Snapshot of the scheduler stats.
    scheduler_stats: SchedulerStats = msgspec_field(
        default_factory=SchedulerStats)

    # Per request stats updates.
    requests_stats_updates: List[RequestStatsUpdate] = msgspec_field(
        default_factory=list)

    # Engine core's queue stats.
    engine_core_process_stats: EngineCoreProcessStats = msgspec_field(
        default_factory=EngineCoreProcessStats)

    # TODO(rickyx): Add other components' stats,
    # e.g. model runner/worker and etc.


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
