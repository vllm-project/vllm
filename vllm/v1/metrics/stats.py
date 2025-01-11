from dataclasses import dataclass

@dataclass
class SchedulerStats:
    """Stats associated with the scheduler."""

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0

    # gpu_cache_usage: float = 0.0
    # gpu_prefix_cache_hit_rate: float = 0.0
