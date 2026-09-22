# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

from prometheus_client import Counter, REGISTRY

from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics


def _vllm_config() -> MagicMock:
    config = MagicMock()
    config.observability_config.show_hidden_metrics = False
    config.observability_config.kv_cache_metrics = False
    config.model_config.served_model_name = "test-model"
    config.model_config.max_model_len = 2048
    config.model_config.is_diffusion = False
    config.speculative_config = None
    config.kv_transfer_config = None
    config.lora_config = None
    return config


def test_logger_rebuild_preserves_independently_owned_metrics() -> None:
    unregister_vllm_metrics()
    try:
        frontend_metric = Counter(
            "vllm:test_frontend_operation",
            "An independently-owned frontend metric.",
        )
        frontend_metric.inc(3)

        logger1 = PrometheusStatLogger(_vllm_config(), [0, 1])
        logger1.log_metrics_info("cache_config", MagicMock(metrics_info=lambda: {}))

        logger2 = PrometheusStatLogger(_vllm_config(), [0, 1, 2, 3])
        # This was created after __init__, so it must be owned too.
        logger2.log_metrics_info("cache_config", MagicMock(metrics_info=lambda: {}))

        assert frontend_metric in REGISTRY._collector_to_names
        assert frontend_metric._value.get() == 3
        assert set(logger2.gauge_scheduler_running) == {0, 1, 2, 3}
    finally:
        unregister_vllm_metrics()
