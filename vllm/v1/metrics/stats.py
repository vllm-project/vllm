# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.spec_decode.metrics import SpecDecodingStats

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreEvent, EngineCoreOutput, FinishReason


@dataclass
class BaseCacheStats:
    """Stores cache hit statistics."""

    reset: bool = False
    """Whether the cache was reset."""

    requests: int = 0
    """The number of requests in this update."""

    queries: int = 0
    """The number of queries in these requests."""

    hits: int = 0
    """The number of hits in these requests."""


class CachingMetrics:
    """Metrics for caching with a hit rate of the most recent N requests.
    Args:
        interval: The number of the most recent requests to aggregate.
            Defaults to 1000.
    """

    def __init__(self, max_recent_requests: int = 1000) -> None:
        super().__init__()

        self.max_recent_requests = max_recent_requests
        # The current aggregated values.
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0

        # A deque of (requests, queries, hits) for the most recent requests.
        self.query_queue = deque[tuple[int, int, int]]()

    def observe(self, stats: BaseCacheStats):
        """Observe the prefix caching for a set of requests.

        This function is called with information gathered when new requests
        are being scheduled and are looking for computed blocks.

        When there are more than `max_recent_requests` requests, the oldest set
        of requests are removed from the metrics.

        Args:
            stats: The prefix cache stats.
        """
        # reset_prefix_cache was invoked before the current update.
        # Reset the metrics before aggregating the current stats.
        if stats.reset:
            self.reset()

        # DO NOT appending empty stats to avoid helpful info get kicked out
        # due to sliding window.
        if stats.requests == 0:
            return

        # Update the metrics.
        self.query_queue.append((stats.requests, stats.queries, stats.hits))
        self.aggregated_requests += stats.requests
        self.aggregated_query_total += stats.queries
        self.aggregated_query_hit += stats.hits

        # Remove the oldest stats until number of requests does not exceed
        # the limit.
        # NOTE: We preserve the latest added stats regardless.
        while (
            len(self.query_queue) > 1
            and self.aggregated_requests > self.max_recent_requests
        ):
            old_requests, old_queries, old_hits = self.query_queue.popleft()
            self.aggregated_requests -= old_requests
            self.aggregated_query_total -= old_queries
            self.aggregated_query_hit -= old_hits

    def reset(self):
        """Reset the metrics."""
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0
        self.query_queue.clear()

    @property
    def empty(self) -> bool:
        """Return true if no requests have been observed."""
        return self.aggregated_requests == 0

    @property
    def hit_rate(self) -> float:
        """Calculate the hit rate for the past N requests."""
        if self.aggregated_query_total == 0:
            return 0.0
        return self.aggregated_query_hit / self.aggregated_query_total


@dataclass
class PrefixCacheStats(BaseCacheStats):
    """
    Stores prefix cache hit statistics.
    - `reset`: Whether `reset_prefix_cache` was invoked.
    - `queries`: Refers to the number of tokens that were queried.
    """

    preempted_requests: int = 0
    """The number of previously preempted requests in this update."""

    preempted_queries: int = 0
    """The `queries` number for preempted requests."""

    preempted_hits: int = 0
    """The `hits` number for preempted requests."""

    def record(self, num_tokens: int, num_hits: int, preempted: bool) -> None:
        """Aggregate request information into the stats."""
        if preempted:
            # Previously preempted request
            self.preempted_requests += 1
            self.preempted_queries += num_tokens
            self.preempted_hits += num_hits
        else:
            # New request
            self.requests += 1
            self.queries += num_tokens
            self.hits += num_hits


@dataclass
class MultiModalCacheStats(BaseCacheStats):
    """
    Stores multi-modal cache hit statistics.
    - `reset`: Whether `reset_mm_cache` was invoked.
    - `queries`: Refers to the number of multi-modal data items
      that were queried.
    """


@dataclass
class KVCacheEvictionEvent:
    """Single KV cache block eviction sample."""

    lifetime_seconds: float
    idle_seconds: float
    reuse_gaps_seconds: tuple[float, ...]


@dataclass
class SchedulerStats:
    """Stats associated with the scheduler."""

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0

    # These are used for internal DP load-balancing.
    step_counter: int = 0
    current_wave: int = 0

    kv_cache_usage: float = 0.0
    encoder_cache_usage: float = 0.0

    prefix_cache_stats: PrefixCacheStats = field(default_factory=PrefixCacheStats)
    connector_prefix_cache_stats: PrefixCacheStats | None = None

    kv_cache_eviction_events: list[KVCacheEvictionEvent] = field(default_factory=list)

    spec_decoding_stats: SpecDecodingStats | None = None
    kv_connector_stats: dict[str, Any] | None = None

    waiting_lora_adapters: dict[str, int] = field(default_factory=dict)
    running_lora_adapters: dict[str, int] = field(default_factory=dict)

    cudagraph_stats: CUDAGraphStat | None = None

    perf_stats: PerfStats | None = None


@dataclass
class RequestStateStats:
    """Stats that need to be tracked across delta updates."""

    num_generation_tokens: int = 0

    # This is an engine frontend timestamp (wall-clock)
    arrival_time: float = 0.0

    # These are engine core timestamps (monotonic)
    queued_ts: float = 0.0
    scheduled_ts: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0

    # first token latency
    first_token_latency: float = 0.0

    # Track if this request is corrupted (NaNs in logits)
    is_corrupted: bool = False


@dataclass
class FinishedRequestStats:
    """Stats associated with a finished request."""

    finish_reason: "FinishReason"
    e2e_latency: float = 0.0
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    max_tokens_param: int | None = None
    queued_time: float = 0.0
    prefill_time: float = 0.0
    inference_time: float = 0.0
    decode_time: float = 0.0
    mean_time_per_output_token: float = 0.0
    is_corrupted: bool = False
    num_cached_tokens: int = 0


class IterationStats:
    """Stats associated with a single set of EngineCoreOutputs."""

    def __init__(self):
        self.iteration_timestamp = time.time()
        self.num_generation_tokens = 0
        self.num_prompt_tokens = 0
        self.num_preempted_reqs = 0
        self.finished_requests: list[FinishedRequestStats] = []
        self.max_num_generation_tokens_iter: list[int] = []
        self.n_params_iter: list[int] = []
        self.time_to_first_tokens_iter: list[float] = []
        self.inter_token_latencies_iter: list[float] = []
        self.num_corrupted_reqs: int = 0

    def __repr__(self) -> str:
        field_to_value_str = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({field_to_value_str})"

    def _time_since(self, start: float) -> float:
        """Calculate an interval relative to this iteration's timestamp."""
        return self.iteration_timestamp - start

    def update_from_output(
        self,
        output: "EngineCoreOutput",
        engine_core_timestamp: float,
        is_prefilling: bool,
        prompt_len: int,
        req_stats: RequestStateStats,
        lora_states: "LoRARequestStates",
        lora_name: str | None,
    ):
        num_new_generation_tokens = len(output.new_token_ids)

        self.num_generation_tokens += num_new_generation_tokens
        if is_prefilling:
            self.num_prompt_tokens += prompt_len

            first_token_latency = self._time_since(req_stats.arrival_time)
            self.time_to_first_tokens_iter.append(first_token_latency)
            req_stats.first_token_latency = first_token_latency

        req_stats.num_generation_tokens += num_new_generation_tokens

        # Track if this request is corrupted (only check once per request)
        # Early exit if already marked as corrupted to avoid redundant checks
        if (
            envs.VLLM_COMPUTE_NANS_IN_LOGITS
            and not req_stats.is_corrupted
            and output.num_nans_in_logits > 0
        ):
            req_stats.is_corrupted = True

        # Process request-level engine core events
        if output.events is not None:
            self.update_from_events(
                output.request_id,
                output.events,
                is_prefilling,
                req_stats,
                lora_states,
                lora_name,
            )

        # Process the batch-level "new tokens" engine core event
        if is_prefilling:
            req_stats.first_token_ts = engine_core_timestamp
        else:
            itl = engine_core_timestamp - req_stats.last_token_ts
            self.inter_token_latencies_iter.append(itl)

        req_stats.last_token_ts = engine_core_timestamp

    def update_from_events(
        self,
        req_id: str,
        events: list["EngineCoreEvent"],
        is_prefilling: bool,
        req_stats: RequestStateStats,
        lora_states: "LoRARequestStates",
        lora_name: str | None,
    ):
        # Avoid circular dependency
        from vllm.v1.engine import EngineCoreEventType

        for event in events:
            if event.type == EngineCoreEventType.QUEUED:
                req_stats.queued_ts = event.timestamp
                lora_states.request_waiting(req_id, lora_name)
            elif event.type == EngineCoreEventType.SCHEDULED:
                if req_stats.scheduled_ts == 0.0:  # ignore preemptions
                    req_stats.scheduled_ts = event.timestamp
                lora_states.request_running(req_id, lora_name)
            elif event.type == EngineCoreEventType.PREEMPTED:
                self.num_preempted_reqs += 1
                lora_states.request_waiting(req_id, lora_name)

    def update_from_finished_request(
        self,
        finish_reason: "FinishReason",
        num_prompt_tokens: int,
        max_tokens_param: int | None,
        req_stats: RequestStateStats,
        num_cached_tokens: int = 0,
    ):
        e2e_latency = self._time_since(req_stats.arrival_time)

        # Queued interval is from first QUEUED event to first SCHEDULED
        queued_time = req_stats.scheduled_ts - req_stats.queued_ts

        # Prefill interval is from first SCHEDULED to first NEW_TOKEN
        # Any preemptions during prefill is included in the interval
        prefill_time = req_stats.first_token_ts - req_stats.scheduled_ts

        # Decode interval is from first NEW_TOKEN to last NEW_TOKEN
        # Any preemptions during decode are included
        decode_time = req_stats.last_token_ts - req_stats.first_token_ts

        # Inference interval is from first SCHEDULED to last NEW_TOKEN
        # Any preemptions during prefill or decode are included
        inference_time = req_stats.last_token_ts - req_stats.scheduled_ts

        # Do not count the token generated by the prefill phase
        mean_time_per_output_token = (
            decode_time / (req_stats.num_generation_tokens - 1)
            if req_stats.num_generation_tokens - 1 > 0
            else 0
        )

        finished_req = FinishedRequestStats(
            finish_reason=finish_reason,
            e2e_latency=e2e_latency,
            num_prompt_tokens=num_prompt_tokens,
            num_generation_tokens=req_stats.num_generation_tokens,
            max_tokens_param=max_tokens_param,
            queued_time=queued_time,
            prefill_time=prefill_time,
            inference_time=inference_time,
            decode_time=decode_time,
            mean_time_per_output_token=mean_time_per_output_token,
            is_corrupted=req_stats.is_corrupted,
            num_cached_tokens=num_cached_tokens,
        )
        self.finished_requests.append(finished_req)

        # Count corrupted requests when they finish (only once per request)
        if req_stats.is_corrupted:
            self.num_corrupted_reqs += 1


class LoRAStats:
    """Tracks waiting and running request IDs for a single LoRA."""

    def __init__(self):
        self.waiting: set[str] = set()
        self.running: set[str] = set()

    def update(self, req_id: str, waiting: bool, running: bool):
        assert not (waiting and running)
        if waiting:
            self.waiting.add(req_id)
        else:
            self.waiting.discard(req_id)

        if running:
            self.running.add(req_id)
        else:
            self.running.discard(req_id)

    @property
    def empty(self) -> bool:
        return not (self.waiting or self.running)


class LoRARequestStates:
    """A per-LoRA count of running and waiting requests."""

    def __init__(self, log_stats: bool = False):
        self.log_stats = log_stats
        self.requests: defaultdict[str, LoRAStats] = defaultdict(LoRAStats)

    def _request_update(
        self, req_id: str, lora_name: str | None, waiting: bool, running: bool
    ):
        if not self.log_stats or lora_name is None:
            return

        lora_stats = self.requests[lora_name]
        lora_stats.update(req_id, waiting, running)
        if lora_stats.empty:
            del self.requests[lora_name]

    def request_waiting(self, req_id: str, lora_name: str | None):
        self._request_update(req_id, lora_name, waiting=True, running=False)

    def request_running(self, req_id: str, lora_name: str | None):
        self._request_update(req_id, lora_name, waiting=False, running=True)

    def request_finished(self, req_id: str, lora_name: str | None):
        self._request_update(req_id, lora_name, waiting=False, running=False)

    def update_scheduler_stats(self, scheduler_stats: SchedulerStats | None):
        if not self.log_stats or scheduler_stats is None:
            return
        for lora_name, stats in self.requests.items():
            scheduler_stats.waiting_lora_adapters[lora_name] = len(stats.waiting)
            scheduler_stats.running_lora_adapters[lora_name] = len(stats.running)
