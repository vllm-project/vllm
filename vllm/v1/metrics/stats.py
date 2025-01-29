import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreOutput


@dataclass
class SchedulerStats:
    """Stats associated with the scheduler."""

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0

    # gpu_cache_usage: float = 0.0
    # gpu_prefix_cache_hit_rate: float = 0.0


@dataclass
class RequestStateStats:
    """Stats that need to be tracked across delta updates."""

    num_generation_tokens: int = 0
    last_token_time: float = 0.0


@dataclass
class FinishedRequestStats:
    """Stats associated with a finished request."""

    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0


class IterationStats:
    """Stats associated with a single set of EngineCoreOutputs."""

    def __init__(self, log_stats: bool):
        self.log_stats = log_stats
        self.num_generation_tokens = 0
        self.num_prompt_tokens = 0
        self.finished_requests: List[FinishedRequestStats] = []
        self.time_to_first_tokens_iter: List[float] = []
        self.time_per_output_tokens_iter: List[float] = []

    def update_from_output(self, output: "EngineCoreOutput",
                           is_prefilling: bool, prompt_len: int,
                           request_state_stats: RequestStateStats):
        if not self.log_stats:
            return

        num_new_generation_tokens = len(output.new_token_ids)
        now = time.time()
        last_token_latency = now - request_state_stats.last_token_time

        self.num_generation_tokens += num_new_generation_tokens
        if is_prefilling:
            # This relies on the invariant that EngineCore does
            # not stream outputs for partially completed prefills
            # (scheduler.update_from_output makes EngineCoreOutput
            # iff num_computed_tokens == num_tokens).
            assert (num_new_generation_tokens > 0)
            self.num_prompt_tokens += prompt_len

            self.time_to_first_tokens_iter.append(last_token_latency)
        else:
            self.time_per_output_tokens_iter.append(last_token_latency)

        request_state_stats.num_generation_tokens += num_new_generation_tokens
        request_state_stats.last_token_time = now

    def update_from_finished_request(self, request_output: "RequestOutput",
                                     request_state_stats: RequestStateStats):
        self.finished_requests.append(
            FinishedRequestStats(len(request_output.prompt_token_ids),
                                 request_state_stats.num_generation_tokens))
