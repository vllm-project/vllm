# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
These types are defined in this file to avoid importing vllm.engine.metrics
and therefore importing prometheus_client.

This is required due to usage of Prometheus multiprocess mode to enable 
metrics after splitting out the uvicorn process from the engine process.

Prometheus multiprocess mode requires setting PROMETHEUS_MULTIPROC_DIR
before prometheus_client is imported. Typically, this is done by setting
the env variable before launch, but since we are a library, we need to
do this in Python code and lazily import prometheus_client.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from vllm.config import SupportsMetricsInfo, VllmConfig


@dataclass
class Stats:
    """Created by LLMEngine for use by StatLogger."""
    now: float

    # System stats (should have _sys suffix)
    #   Scheduler State
    num_running_sys: int
    num_waiting_sys: int
    num_swapped_sys: int
    #   KV Cache Usage in %
    gpu_cache_usage_sys: float
    cpu_cache_usage_sys: float
    #   Prefix caching block hit rate
    cpu_prefix_cache_hit_rate: float
    gpu_prefix_cache_hit_rate: float

    # Iteration stats (should have _iter suffix)
    num_prompt_tokens_iter: int
    num_generation_tokens_iter: int
    num_tokens_iter: int
    time_to_first_tokens_iter: List[float]
    time_per_output_tokens_iter: List[float]
    num_preemption_iter: int

    # Request stats (should have _requests suffix)
    #   Latency
    time_e2e_requests: List[float]
    time_queue_requests: List[float]
    time_inference_requests: List[float]
    time_prefill_requests: List[float]
    time_decode_requests: List[float]
    #   Metadata
    num_prompt_tokens_requests: List[int]
    num_generation_tokens_requests: List[int]
    n_requests: List[int]
    max_num_generation_tokens_requests: List[int]
    max_tokens_requests: List[int]
    finished_reason_requests: List[str]
    waiting_lora_adapters: List[str]
    running_lora_adapters: List[str]
    max_lora: str


class StatLoggerBase(ABC):
    """Base class for StatLogger."""

    def __init__(self, local_interval: float, vllm_config: VllmConfig) -> None:
        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []
        self.last_local_log = time.time()
        self.local_interval = local_interval

    @abstractmethod
    def log(self, stats: Stats) -> None:
        raise NotImplementedError

    @abstractmethod
    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        raise NotImplementedError
