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
class ScheduledTiming:
    """Timing for a request that was scheduled but didn't generate tokens.

    This occurs when a request is aborted during prefill or fails due to
    errors like KV load failures.
    """

    queued_time: float


@dataclass
class CompletedTiming:
    """Timing for a request that generated at least one token."""

    queued_time: float
    prefill_time: float
    decode_time: float
    inference_time: float
    # None if request generated only a single token
    mean_time_per_output_token: float | None = None


RequestTiming = ScheduledTiming | CompletedTiming | None


@dataclass
class FinishedRequestStats:
    """Stats associated with a finished request.

    The timing field uses its type to encode how far the
    request progressed before finishing:
    - None: rejected before scheduling (e.g., cache_threshold)
    - ScheduledTiming: scheduled but no tokens generated (abort/error during prefill)
    - CompletedTiming: generated at least one token (normal completion)
    """

    finish_reason: "FinishReason"
    e2e_latency: float = 0.0
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    max_tokens_param: int | None = None
    timing: RequestTiming = None
    is_corrupted: bool = False
    num_cached_tokens: int = 0


@dataclass
class PromptTokenStats:
    """Breakdown of prompt tokens by source.

    Fields:
        computed: Tokens prefilled locally (actual compute work).
        local_cache_hit: Tokens from local prefix cache.
        external_kv_transfer: Tokens from external KV transfer.
        cached_tokens: Tokens skipped during prefill (from scheduler).
        recomputed_tokens: Cached tokens that were recomputed (see below).
        total: Total prompt tokens.

    Invariants:
        computed + local_cache_hit + external_kv_transfer - recomputed_tokens = total
        local_cache_hit + external_kv_transfer - recomputed_tokens = cached_tokens
    """

    ALL_SOURCES: tuple[str, ...] = (
        "local_compute",
        "local_cache_hit",
        "external_kv_transfer",
    )

    computed: int = 0
    local_cache_hit: int = 0
    external_kv_transfer: int = 0
    cached_tokens: int = 0
    recomputed_tokens: int = 0
    total: int = 0

    def update_from_output(
        self,
        num_cached_tokens: int,
        num_external_computed_tokens: int,
        prompt_len: int,
    ) -> None:
        """Update stats from a prefill output."""
        # When all tokens are cached, the scheduler reduces num_cached_tokens
        # by 1 to force the model to recompute the last token, since the model
        # needs at least one input token to run a forward pass.
        recomputed = 1 if (num_cached_tokens + 1 == prompt_len) else 0

        self.computed += prompt_len - num_cached_tokens
        self.external_kv_transfer += num_external_computed_tokens
        self.local_cache_hit += (
            num_cached_tokens + recomputed - num_external_computed_tokens
        )
        self.cached_tokens += num_cached_tokens
        self.recomputed_tokens += recomputed
        self.total += prompt_len

    def get_by_source(self, source: str) -> int:
        """Get token count by source label."""
        source_map = {
            "local_compute": self.computed,
            "local_cache_hit": self.local_cache_hit,
            "external_kv_transfer": self.external_kv_transfer,
        }
        if source not in source_map:
            raise ValueError(f"Unknown source: {source}")
        return source_map[source]


class IterationStats:
    """Stats associated with a single set of EngineCoreOutputs."""

    def __init__(self):
        self.iteration_timestamp = time.time()
        self.num_generation_tokens = 0
        self.prompt_token_stats = PromptTokenStats()
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

    @property
    def num_prompt_tokens(self) -> int:
        """Total prompt tokens (for backward compatibility)."""
        return self.prompt_token_stats.total

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
            self.prompt_token_stats.update_from_output(
                num_cached_tokens=output.num_cached_tokens,
                num_external_computed_tokens=output.num_external_computed_tokens,
                prompt_len=prompt_len,
            )

        # Only record first token latency when a token was actually generated.
        # is_prefilling can be True even when no tokens are produced (e.g.,
        # KV-cache load failures, aborts during prefill).
        if is_prefilling and num_new_generation_tokens > 0:
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

        # Process the batch-level "new tokens" engine core event.
        # Only update timestamps when tokens were actually generated.
        if num_new_generation_tokens > 0:
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

        # build timing based on how far the request progressed
        timing: RequestTiming = None
        was_queued = req_stats.queued_ts > 0
        was_scheduled = req_stats.scheduled_ts > 0
        got_first_token = req_stats.first_token_ts > 0
        got_last_token = req_stats.last_token_ts > 0

        if was_queued and was_scheduled:
            # queued: from first QUEUED event to first SCHEDULED
            queued_time = req_stats.scheduled_ts - req_stats.queued_ts

            if got_first_token and got_last_token:
                # request generated tokens - full timing available

                # prefill: from first SCHEDULED to first NEW_TOKEN
                # (any preemptions during prefill are included)
                prefill_time = req_stats.first_token_ts - req_stats.scheduled_ts

                # decode: from first NEW_TOKEN to last NEW_TOKEN
                # (any preemptions during decode are included)
                decode_time = req_stats.last_token_ts - req_stats.first_token_ts

                # inference: from first SCHEDULED to last NEW_TOKEN
                # (any preemptions during prefill or decode are included)
                inference_time = req_stats.last_token_ts - req_stats.scheduled_ts

                # don't count the token generated by the prefill phase
                mean_tpot = (
                    decode_time / (req_stats.num_generation_tokens - 1)
                    if req_stats.num_generation_tokens > 1
                    else None
                )
                timing = CompletedTiming(
                    queued_time=queued_time,
                    prefill_time=prefill_time,
                    decode_time=decode_time,
                    inference_time=inference_time,
                    mean_time_per_output_token=mean_tpot,
                )
            else:
                # scheduled but no tokens (abort during prefill, KV error, etc.)
                timing = ScheduledTiming(queued_time=queued_time)
        # else: timing stays None (rejected before scheduling)

        finished_req = FinishedRequestStats(
            finish_reason=finish_reason,
            e2e_latency=e2e_latency,
            num_prompt_tokens=num_prompt_tokens,
            num_generation_tokens=req_stats.num_generation_tokens,
            max_tokens_param=max_tokens_param,
            timing=timing,
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
