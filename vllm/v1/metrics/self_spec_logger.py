# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Custom stat logger for tracking self-speculative decoding metrics over time.
"""

from typing import Optional

from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats


class SelfSpecStatLogger(StatLoggerBase):
    """Stat logger that tracks self-spec metrics history across iterations."""

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        self.vllm_config = vllm_config
        self.engine_index = engine_index

        # History tracking lists
        self.total_scheduled_tokens_history: list[int] = []
        self.num_scheduled_reqs_history: list[int] = []
        self.num_cached_reqs_in_accumulating_history: list[int] = []
        self.num_cached_reqs_in_verifying_history: list[int] = []

    def record(
        self,
        scheduler_stats: Optional[SchedulerStats],
        iteration_stats: Optional[IterationStats],
        engine_idx: int = 0,
    ):
        """Record self-spec metrics from scheduler stats."""
        if scheduler_stats is None:
            return

        # Track number of running requests (scheduled requests)
        self.num_scheduled_reqs_history.append(
            scheduler_stats.num_running_reqs)

        # Track self-spec state counts
        self.num_cached_reqs_in_accumulating_history.append(
            scheduler_stats.num_cached_reqs_in_accumulating)
        self.num_cached_reqs_in_verifying_history.append(
            scheduler_stats.num_cached_reqs_in_verifying)

        # Track total scheduled tokens from iteration stats
        if iteration_stats is not None:
            total_tokens = (iteration_stats.num_prompt_tokens +
                            iteration_stats.num_generation_tokens)
            self.total_scheduled_tokens_history.append(total_tokens)
        else:
            # If no iteration stats, append 0
            self.total_scheduled_tokens_history.append(0)

    def log_engine_initialized(self):
        """Called when engine is initialized."""
        pass

    def log(self):
        """Periodic logging - not used for history tracking."""
        pass
