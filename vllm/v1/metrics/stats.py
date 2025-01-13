from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutput


@dataclass
class SchedulerStats:
    """Stats associated with the scheduler."""

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0

    # gpu_cache_usage: float = 0.0
    # gpu_prefix_cache_hit_rate: float = 0.0


class IterationStats:
    """Stats associated with a single set of EngineCoreOutputs."""

    def __init__(self, log_stats: bool):
        self.log_stats = log_stats
        self.num_generation_tokens = 0
        self.num_prompt_tokens = 0

    def update_from_output(self, output: "EngineCoreOutput",
                           is_prefilling: bool, prompt_len: int):
        if not self.log_stats:
            return

        self.num_generation_tokens += len(output.new_token_ids)
        if is_prefilling:
            # This relies on the invariant that EngineCore does
            # not stream outputs for partially completed prefills
            # (scheduler.update_from_output makes EngineCoreOutput
            # iff num_computed_tokens == num_tokens).
            assert (len(output.new_token_ids) > 0)
            self.num_prompt_tokens += prompt_len
