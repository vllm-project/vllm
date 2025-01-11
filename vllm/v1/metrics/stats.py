from dataclasses import dataclass


@dataclass
class SchedulerStats:
    """Stats associated with the scheduler."""

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0

    # gpu_cache_usage: float = 0.0
    # gpu_prefix_cache_hit_rate: float = 0.0


@dataclass
class IterationStats:
    """Stats associated with a single iteration"""

    num_generation_tokens: int = 0
    num_prompt_tokens: int = 0
