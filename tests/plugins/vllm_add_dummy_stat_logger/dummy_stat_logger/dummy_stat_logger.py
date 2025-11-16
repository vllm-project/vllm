# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.metrics.loggers import StatLoggerBase


class DummyStatLogger(StatLoggerBase):
    """
    A dummy stat logger for testing purposes.
    Implements the minimal interface expected by StatLoggerManager.
    """

    def __init__(self, vllm_config, engine_idx=0):
        self.vllm_config = vllm_config
        self.engine_idx = engine_idx
        self.recorded = []
        self.logged = False
        self.engine_initialized = False

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats, engine_idx):
        self.recorded.append(
            (scheduler_stats, iteration_stats, mm_cache_stats, engine_idx)
        )

    def log(self):
        self.logged = True

    def log_engine_initialized(self):
        self.engine_initialized = True
