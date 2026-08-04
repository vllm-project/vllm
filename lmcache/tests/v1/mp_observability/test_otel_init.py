# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``start_http_server`` flag in OTel metrics init."""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.config import (
    ObservabilityConfig,
    init_observability,
)
from lmcache.v1.mp_observability.otel_init import init_otel_metrics


@pytest.fixture(autouse=True)
def _mock_otel_provider(monkeypatch):
    """Avoid mutating the process-global OTel MeterProvider."""
    # Third Party
    from opentelemetry import metrics as otel_metrics

    monkeypatch.setattr(otel_metrics, "set_meter_provider", MagicMock())


def test_init_otel_metrics_starts_http_server_by_default():
    with patch("prometheus_client.start_http_server") as mock_start:
        init_otel_metrics(prometheus_port=19090)
    mock_start.assert_called_once_with(19090)


def test_init_otel_metrics_skips_http_server_when_disabled():
    with patch("prometheus_client.start_http_server") as mock_start:
        init_otel_metrics(prometheus_port=19091, start_http_server=False)
    mock_start.assert_not_called()


def test_init_otel_metrics_otlp_mode_never_starts_prom_server():
    """OTLP push mode must not spawn a Prometheus HTTP server,
    regardless of ``start_http_server``."""
    with (
        patch(
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter"
        ),
        patch("prometheus_client.start_http_server") as mock_start,
    ):
        init_otel_metrics(
            otlp_endpoint="http://localhost:4317",
            start_http_server=True,
        )
    mock_start.assert_not_called()


def test_init_observability_propagates_flag_false():
    """``init_observability`` must forward ``False`` to
    ``init_otel_metrics``."""
    cfg = ObservabilityConfig(enabled=True, metrics_enabled=True, logging_enabled=False)
    with patch("lmcache.v1.mp_observability.otel_init.init_otel_metrics") as mock_init:
        init_observability(cfg, start_prometheus_http_server=False)
    mock_init.assert_called_once()
    assert mock_init.call_args.kwargs["start_http_server"] is False


def test_init_observability_propagates_flag_true_by_default():
    cfg = ObservabilityConfig(enabled=True, metrics_enabled=True, logging_enabled=False)
    with patch("lmcache.v1.mp_observability.otel_init.init_otel_metrics") as mock_init:
        init_observability(cfg)
    mock_init.assert_called_once()
    assert mock_init.call_args.kwargs["start_http_server"] is True
