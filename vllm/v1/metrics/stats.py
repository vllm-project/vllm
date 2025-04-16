# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from vllm.v1.spec_decode.metrics import SpecDecodingStats

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreEvent, EngineCoreOutput, FinishReason
    from vllm.v1.engine.output_processor import RequestState


@dataclass
class PrefixCacheStats:
    """Stores prefix cache hit statistics."""
    # Whether reset_prefix_cache was invoked.
    reset: bool = False
    # The number of requests in this update.
    requests: int = 0
    # The number of queries in these requests. Note that "queries" here
    # means the number of blocks that were queried from the cache.
    queries: int = 0
    # The number of hits in these requests.
    hits: int = 0


@dataclass
class SchedulerStats:
    """Stats associated with the scheduler."""

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0

    gpu_cache_usage: float = 0.0

    prefix_cache_stats: PrefixCacheStats = field(
        default_factory=PrefixCacheStats)

    spec_decoding_stats: Optional[SpecDecodingStats] = None


@dataclass
class LoRAStats:
    waiting_requests: set[str] = field(default_factory=set)
    running_requests: set[str] = field(default_factory=set)


@dataclass
class RequestStateStats:
    """Stats that need to be tracked across delta updates."""

    num_generation_tokens: int = 0

    # This is a engine frontend timestamp (wall-clock)
    arrival_time: float = 0.0

    # These are engine core timestamps (monotonic)
    queued_ts: float = 0.0
    scheduled_ts: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0


@dataclass
class FinishedRequestStats:
    """Stats associated with a finished request."""

    finish_reason: "FinishReason"
    e2e_latency: float = 0.0
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    max_tokens_param: Optional[int] = None
    queued_time: float = 0.0
    prefill_time: float = 0.0
    inference_time: float = 0.0
    decode_time: float = 0.0


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
        self.time_per_output_tokens_iter: list[float] = []
        self.waiting_lora_adapters: dict[str, int] = {}
        self.running_lora_adapters: dict[str, int] = {}

    def _time_since(self, start: float) -> float:
        """Calculate an interval relative to this iteration's timestamp."""
        return self.iteration_timestamp - start

    def update_from_output(self, output: "EngineCoreOutput",
                           engine_core_timestamp: float, is_prefilling: bool,
                           prompt_len: int, req_stats: RequestStateStats,
                           lora_stats: Optional[LoRAStats]):
        num_new_generation_tokens = len(output.new_token_ids)

        self.num_generation_tokens += num_new_generation_tokens
        if is_prefilling:
            assert num_new_generation_tokens > 0
            self.num_prompt_tokens += prompt_len

            first_token_latency = self._time_since(req_stats.arrival_time)
            self.time_to_first_tokens_iter.append(first_token_latency)

        req_stats.num_generation_tokens += num_new_generation_tokens

        # Process request-level engine core events
        if output.events is not None:
            self.update_from_events(output.request_id, output.events,
                                    is_prefilling, req_stats, lora_stats)

        # Process the batch-level "new tokens" engine core event
        if is_prefilling:
            req_stats.first_token_ts = engine_core_timestamp
        else:
            tpot = engine_core_timestamp - req_stats.last_token_ts
            self.time_per_output_tokens_iter.append(tpot)

        req_stats.last_token_ts = engine_core_timestamp

    def update_from_events(self, req_id: str, events: list["EngineCoreEvent"],
                           is_prefilling: bool, req_stats: RequestStateStats,
                           lora_stats: Optional[LoRAStats]):
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

    def update_from_finished_request(self, finish_reason: "FinishReason",
                                     num_prompt_tokens: int,
                                     max_tokens_param: Optional[int],
                                     req_stats: RequestStateStats):
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

        finished_req = \
            FinishedRequestStats(finish_reason=finish_reason,
                                 e2e_latency=e2e_latency,
                                 num_prompt_tokens=num_prompt_tokens,
                                 num_generation_tokens=req_stats.num_generation_tokens,
                                 max_tokens_param=max_tokens_param,
                                 queued_time=queued_time,
                                 prefill_time=prefill_time,
                                 inference_time=inference_time,
                                 decode_time=decode_time)
        self.finished_requests.append(finished_req)


class LoRARequestStates:
    """Per-LoRA request state stats."""

    def __init__(self):
        self.lora_name_to_stats: dict[str, LoRAStats] = {}

    def get_stats(self, req_state: 'RequestState') -> Optional[LoRAStats]:
        if req_state.lora_name is None:
            return None
        if req_state.lora_name not in self.lora_name_to_stats:
            self.lora_name_to_stats[req_state.lora_name] = LoRAStats()
        return self.lora_name_to_stats[req_state.lora_name]

    def add_request(self, req_state: 'RequestState'):
        if (lora_stats := self.get_stats(req_state)) is not None:
            lora_stats.waiting_requests.add(req_state.request_id)

    def finish_request(self, req_state: 'RequestState'):
        if req_state.lora_name is None:
            return
        lora_stats = self.lora_name_to_stats[req_state.lora_name]
        lora_stats.running_requests.remove(req_state.request_id)

    def abort_request(self, req_state: 'RequestState'):
        if req_state.lora_name is None:
            return
        lora_stats = self.lora_name_to_stats[req_state.lora_name]
        lora_stats.waiting_requests.discard(req_state.request_id)
        lora_stats.running_requests.discard(req_state.request_id)

    # Break the pattern for this lifecycle methods so we can
    # call this from IterationStats.update_from_events()
    @staticmethod
    def scheduled_request(lora_stats: Optional[LoRAStats], request_id: str):
        if lora_stats is None:
            return
        lora_stats.waiting_requests.remove(request_id)
        lora_stats.running_requests.add(request_id)

    @staticmethod
    def preempted_request(lora_stats: Optional[LoRAStats], request_id: str):
        if lora_stats is None:
            return
        lora_stats.running_requests.remove(request_id)
        lora_stats.waiting_requests.add(request_id)

    def update_iteration_stats(self,
                               iteration_stats: Optional[IterationStats]):
        if iteration_stats is None:
            return
        for lora_name, stats in self.lora_name_to_stats.items():
            if stats.waiting_requests:
                iteration_stats.waiting_lora_adapters[lora_name] = \
                    len(stats.waiting_requests)
            if stats.running_requests:
                iteration_stats.running_lora_adapters[lora_name] = \
                    len(stats.running_requests)
