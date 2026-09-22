# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY, Counter

from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _vllm_config() -> MagicMock:
    config = MagicMock()
    config.observability_config.show_hidden_metrics = False
    config.observability_config.kv_cache_metrics = False
    config.model_config.served_model_name = "test-model"
    config.model_config.max_model_len = 2048
    config.model_config.is_diffusion = False
    config.speculative_config = None
    config.kv_transfer_config = None
    config.ec_transfer_config = None
    config.lora_config = None
    return config


class _TestECConnectorMetrics:
    def __init__(self, metric_types, labelnames) -> None:
        self.counter = metric_types[Counter](
            "vllm:test_ec_connector_operations",
            "Test EC connector operations.",
            labelnames=labelnames,
        )

    def observe(self, transfer_stats_data, engine_idx=0) -> None:
        pass


class _TestECConnector:
    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config,
        metric_types,
        labelnames,
        per_engine_labelvalues,
    ):
        del cls, vllm_config, per_engine_labelvalues
        return _TestECConnectorMetrics(metric_types, labelnames)


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


def test_logger_rebuild_recreates_ec_connector_metrics(monkeypatch) -> None:
    unregister_vllm_metrics()
    monkeypatch.setattr(
        ECConnectorFactory,
        "get_connector_class",
        lambda ec_transfer_config: _TestECConnector,
    )
    config = _vllm_config()
    config.ec_transfer_config = MagicMock(ec_connector="TestECConnector")
    try:
        logger1 = PrometheusStatLogger(config, [0, 1])
        first_counter = logger1.ec_connector_prom.prom_metrics.counter

        logger2 = PrometheusStatLogger(config, [0, 1, 2, 3])
        second_counter = logger2.ec_connector_prom.prom_metrics.counter

        assert first_counter not in REGISTRY._collector_to_names
        assert second_counter in REGISTRY._collector_to_names
        assert set(logger2.gauge_scheduler_running) == {0, 1, 2, 3}
    finally:
        unregister_vllm_metrics()
