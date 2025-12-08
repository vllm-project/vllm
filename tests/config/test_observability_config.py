# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from unittest.mock import MagicMock, patch

import pytest

from vllm.config import ObservabilityConfig


@pytest.mark.cpu_test
class TestObservabilityConfig:
    """Test ObservabilityConfig validation."""

    def test_default_config(self):
        """Test that default config is valid."""
        config = ObservabilityConfig()
        assert config.disable_prometheus_metrics is False
        assert config.enable_otel_metrics is False
        assert config.otlp_traces_endpoint is None

    def test_enable_otel_without_package_raises_error(self):
        """Test that enabling otel_metrics without the package raises error."""
        # Mock the opentelemetry imports to fail
        with patch.dict(
            sys.modules,
            {
                "opentelemetry": None,
                "opentelemetry.metrics": None,
                "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": None,
            },
        ):
            # Force reload to pick up the mocked imports
            # Reload the module to simulate missing imports
            import importlib

            import vllm.v1.metrics.opentelemetry_metrics

            importlib.reload(vllm.v1.metrics.opentelemetry_metrics)

            with pytest.raises(ValueError) as exc_info:
                ObservabilityConfig(enable_otel_metrics=True)

            assert "OpenTelemetry Metrics SDK is not available" in str(exc_info.value)
            assert "pip install" in str(exc_info.value)

    def test_enable_otel_with_package_succeeds(self):
        """Test that enabling otel_metrics with the package succeeds."""
        # Mock the opentelemetry package as available
        mock_metrics = MagicMock()
        mock_exporter = MagicMock()
        mock_sdk = MagicMock()
        mock_resource = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "opentelemetry": MagicMock(),
                "opentelemetry.metrics": mock_metrics,
                "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": mock_exporter,
                "opentelemetry.sdk.metrics": mock_sdk,
                "opentelemetry.sdk.metrics.export": MagicMock(),
                "opentelemetry.sdk.resources": mock_resource,
            },
        ):
            # Force reload to pick up the mocked imports
            import importlib

            import vllm.v1.metrics.opentelemetry_metrics

            importlib.reload(vllm.v1.metrics.opentelemetry_metrics)

            # Should not raise
            config = ObservabilityConfig(enable_otel_metrics=True)
            assert config.enable_otel_metrics is True

    def test_disable_prometheus_without_otel_raises_error(self):
        """Test that disabling Prometheus without enabling OTel raises error."""
        with pytest.raises(ValueError) as exc_info:
            ObservabilityConfig(
                disable_prometheus_metrics=True, enable_otel_metrics=False
            )

        assert "Cannot disable Prometheus metrics" in str(exc_info.value)
        assert "--enable-otel-metrics" in str(exc_info.value)

    def test_disable_prometheus_with_otel_succeeds(self):
        """Test that disabling Prometheus with OTel enabled succeeds."""
        # Mock the opentelemetry package as available
        with patch.dict(
            sys.modules,
            {
                "opentelemetry": MagicMock(),
                "opentelemetry.metrics": MagicMock(),
                "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": MagicMock(),
                "opentelemetry.sdk.metrics": MagicMock(),
                "opentelemetry.sdk.metrics.export": MagicMock(),
                "opentelemetry.sdk.resources": MagicMock(),
            },
        ):
            # Force reload to pick up the mocked imports
            import importlib

            import vllm.v1.metrics.opentelemetry_metrics

            importlib.reload(vllm.v1.metrics.opentelemetry_metrics)

            # Should not raise
            config = ObservabilityConfig(
                disable_prometheus_metrics=True, enable_otel_metrics=True
            )
            assert config.disable_prometheus_metrics is True
            assert config.enable_otel_metrics is True

    def test_both_prometheus_and_otel_enabled_succeeds(self):
        """Test that having both Prometheus and OTel enabled succeeds."""
        # Mock the opentelemetry package as available
        with patch.dict(
            sys.modules,
            {
                "opentelemetry": MagicMock(),
                "opentelemetry.metrics": MagicMock(),
                "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": MagicMock(),
                "opentelemetry.sdk.metrics": MagicMock(),
                "opentelemetry.sdk.metrics.export": MagicMock(),
                "opentelemetry.sdk.resources": MagicMock(),
            },
        ):
            # Force reload to pick up the mocked imports
            import importlib

            import vllm.v1.metrics.opentelemetry_metrics

            importlib.reload(vllm.v1.metrics.opentelemetry_metrics)

            # Both enabled should work fine
            config = ObservabilityConfig(
                disable_prometheus_metrics=False, enable_otel_metrics=True
            )
            assert config.disable_prometheus_metrics is False
            assert config.enable_otel_metrics is True

    def test_tracing_validation_requires_endpoint(self):
        """Test that detailed tracing requires OTLP endpoint."""
        with pytest.raises(ValueError) as exc_info:
            ObservabilityConfig(
                collect_detailed_traces=["model"], otlp_traces_endpoint=None
            )

        assert "collect_detailed_traces requires" in str(exc_info.value)
        assert "--otlp-traces-endpoint" in str(exc_info.value)

    def test_tracing_with_endpoint_succeeds(self):
        """Test that detailed tracing with endpoint succeeds."""
        config = ObservabilityConfig(
            collect_detailed_traces=["model"],
            otlp_traces_endpoint="http://localhost:4317",
        )
        assert config.collect_detailed_traces == ["model"]
        assert config.otlp_traces_endpoint == "http://localhost:4317"


if __name__ == "__main__":
    pytest.main([__file__])
