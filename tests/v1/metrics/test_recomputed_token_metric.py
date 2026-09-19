# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from prometheus_client import generate_latest

from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.stats import SchedulerStats

pytestmark = pytest.mark.cpu_test


class _NoopPrometheusAdapter:
    def __init__(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass


class _FakeCacheConfig:
    num_gpu_blocks = 1

    def metrics_info(self) -> dict[str, str]:
        return {"cache_dtype": "auto"}


class _TestPrometheusStatLogger(PrometheusStatLogger):
    _spec_decoding_cls = _NoopPrometheusAdapter
    _kv_connector_cls = _NoopPrometheusAdapter
    _perf_metrics_cls = _NoopPrometheusAdapter


def test_recomputed_token_counter_is_exposed_and_exact():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            served_model_name="fake",
            max_model_len=128,
            is_diffusion=False,
        ),
        observability_config=SimpleNamespace(
            show_hidden_metrics=False,
            kv_cache_metrics=False,
        ),
        speculative_config=None,
        lora_config=None,
        cache_config=_FakeCacheConfig(),
    )
    logger = _TestPrometheusStatLogger(config)

    logger.record(SchedulerStats(num_recomputed_tokens=8), None)
    logger.record(SchedulerStats(num_recomputed_tokens=9), None)

    exposition = generate_latest().decode()
    assert (
        'vllm:recomputed_token_executions_total{engine="0",model_name="fake"} 17.0'
        in exposition
    )
