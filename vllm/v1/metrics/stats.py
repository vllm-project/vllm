# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.v1.spec_decode.metrics import SpecDecodingStats

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreEvent, EngineCoreOutput, FinishReason
    from vllm.v1.engine.output_processor import RequestState


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

    block_residency_time: float = 0.0
    """Sum of lifetimes (seconds) for cached blocks evicted in this update."""

    evicted_blocks: int = 0
    """The number of cached blocks evicted in this update."""

    request_residency_time: float = 0.0
    """Sum of lifetimes (seconds) for request prefixes evicted in this update."""

    evicted_requests: int = 0
    """The number of requests whose cached prefixes fully evicted."""


@dataclass
class MultiModalCacheStats(BaseCacheStats):
    """
    Stores multi-modal cache hit statistics.
    - `reset`: Whether `reset_mm_cache` was invoked.
    - `queries`: Refers to the number of multi-modal data items
      that were queried.
    """


@dataclass
class SchedulerStats:
    """Stats associated with the scheduler."""

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0

    # These are used for internal DP load-balancing.
    step_counter: int = 0
    current_wave: int = 0

    kv_cache_usage: float = 0.0

    prefix_cache_stats: PrefixCacheStats = field(default_factory=PrefixCacheStats)

    spec_decoding_stats: SpecDecodingStats | None = None
    kv_connector_stats: dict[str, Any] | None = None

    num_corrupted_reqs: int = 0


@dataclass
class LoRAStats:
    waiting_requests: set[str] = field(default_factory=set)
    running_requests: set[str] = field(default_factory=set)


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
        self.waiting_lora_adapters: dict[str, int] = {}
        self.running_lora_adapters: dict[str, int] = {}

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
        lora_stats: LoRAStats | None,
    ):
        num_new_generation_tokens = len(output.new_token_ids)

        self.num_generation_tokens += num_new_generation_tokens
        if is_prefilling:
            self.num_prompt_tokens += prompt_len

            first_token_latency = self._time_since(req_stats.arrival_time)
            self.time_to_first_tokens_iter.append(first_token_latency)
            req_stats.first_token_latency = first_token_latency

        req_stats.num_generation_tokens += num_new_generation_tokens

        # Process request-level engine core events
        if output.events is not None:
            self.update_from_events(
                output.request_id, output.events, is_prefilling, req_stats, lora_stats
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
        lora_stats: LoRAStats | None,
    ):
        # Avoid circular dependency
        from vllm.v1.engine import EngineCoreEventType

        for event in events:
            if event.type == EngineCoreEventType.QUEUED:
                req_stats.queued_ts = event.timestamp
                if lora_stats is not None:
                    lora_stats.waiting_requests.add(req_id)
            elif event.type == EngineCoreEventType.SCHEDULED:
                if req_stats.scheduled_ts == 0.0:  # ignore preemptions
                    req_stats.scheduled_ts = event.timestamp
                LoRARequestStates.scheduled_request(lora_stats, req_id)
            elif event.type == EngineCoreEventType.PREEMPTED:
                self.num_preempted_reqs += 1
                LoRARequestStates.preempted_request(lora_stats, req_id)

    def update_from_finished_request(
        self,
        finish_reason: "FinishReason",
        num_prompt_tokens: int,
        max_tokens_param: int | None,
        req_stats: RequestStateStats,
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
        )
        self.finished_requests.append(finished_req)


class LoRARequestStates:
    """Per-LoRA request state stats."""

    def __init__(self):
        self.lora_name_to_stats: dict[str, LoRAStats] = {}

    def get_stats(self, req_state: "RequestState") -> LoRAStats | None:
        if req_state.lora_name is None:
            return None
        if req_state.lora_name not in self.lora_name_to_stats:
            self.lora_name_to_stats[req_state.lora_name] = LoRAStats()
        return self.lora_name_to_stats[req_state.lora_name]

    def add_request(self, req_state: "RequestState"):
        if (lora_stats := self.get_stats(req_state)) is not None:
            lora_stats.waiting_requests.add(req_state.request_id)

    def finish_request(self, req_state: "RequestState"):
        if req_state.lora_name is None:
            return
        lora_stats = self.lora_name_to_stats[req_state.lora_name]
        lora_stats.running_requests.remove(req_state.request_id)

    def abort_request(self, req_state: "RequestState"):
        if req_state.lora_name is None:
            return
        lora_stats = self.lora_name_to_stats[req_state.lora_name]
        lora_stats.waiting_requests.discard(req_state.request_id)
        lora_stats.running_requests.discard(req_state.request_id)

    # Break the pattern for this lifecycle methods so we can
    # call this from IterationStats.update_from_events()
    @staticmethod
    def scheduled_request(lora_stats: LoRAStats | None, request_id: str):
        if lora_stats is None:
            return
        lora_stats.waiting_requests.remove(request_id)
        lora_stats.running_requests.add(request_id)

    @staticmethod
    def preempted_request(lora_stats: LoRAStats | None, request_id: str):
        if lora_stats is None:
            return
        lora_stats.running_requests.remove(request_id)
        lora_stats.waiting_requests.add(request_id)

    def update_iteration_stats(self, iteration_stats: IterationStats | None):
        if iteration_stats is None:
            return
        for lora_name, stats in self.lora_name_to_stats.items():
            if stats.waiting_requests:
                iteration_stats.waiting_lora_adapters[lora_name] = len(
                    stats.waiting_requests
                )
            if stats.running_requests:
                iteration_stats.running_lora_adapters[lora_name] = len(
                    stats.running_requests
                )
