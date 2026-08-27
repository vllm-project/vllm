# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

from prometheus_client import REGISTRY

from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.stats import IterationStats


def test_cached_prompt_tokens_are_exposed_by_physical_source():
    config = VllmConfig()
    config.model_config = cast(
        Any,
        SimpleNamespace(
            served_model_name="test-model",
            max_model_len=16,
            is_diffusion=False,
        ),
    )

    try:
        logger = PrometheusStatLogger(config)
        iteration_stats = IterationStats()
        iteration_stats.prompt_token_stats.cached_tokens = 7
        iteration_stats.prompt_token_stats.cached_tokens_by_source = {
            "device": 3,
            "cpu": 4,
        }

        logger.record(None, iteration_stats)

        samples = {}
        for metric in REGISTRY.collect():
            if metric.name != "vllm:prompt_tokens_cached_by_source":
                continue
            for sample in metric.samples:
                if sample.name == "vllm:prompt_tokens_cached_by_source_total":
                    samples[sample.labels["source"]] = sample.value
        assert samples == {"device": 3, "cpu": 4}
    finally:
        unregister_vllm_metrics()
