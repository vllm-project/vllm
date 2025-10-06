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
    time_to_first_tokens_iter: list[float]
    inter_token_latencies_iter: list[float]
    num_preemption_iter: int

    # Request stats (should have _requests suffix)
    #   Latency
    time_e2e_requests: list[float]
    time_queue_requests: list[float]
    time_inference_requests: list[float]
    time_prefill_requests: list[float]
    time_decode_requests: list[float]
    #   Metadata
    num_prompt_tokens_requests: list[int]
    num_generation_tokens_requests: list[int]
    n_requests: list[int]
    max_num_generation_tokens_requests: list[int]
    max_tokens_requests: list[int]
    finished_reason_requests: list[str]
    waiting_lora_adapters: list[str]
    running_lora_adapters: list[str]
    max_lora: str


class StatLoggerBase(ABC):
    """Base class for StatLogger."""

    def __init__(self, local_interval: float, vllm_config: VllmConfig) -> None:
        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: list[int] = []
        self.num_generation_tokens: list[int] = []
        self.last_local_log = time.time()
        self.local_interval = local_interval

    @abstractmethod
    def log(self, stats: Stats) -> None:
        raise NotImplementedError

    @abstractmethod
    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        raise NotImplementedError
