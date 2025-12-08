# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import pytest

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ObservabilityConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.metrics.stats import (
    FinishedRequestStats,
    IterationStats,
    PrefixCacheStats,
    SchedulerStats,
)

pytestmark = pytest.mark.cpu_test


@dataclass
class MockFinishReason:
    """Mock finish reason for testing."""

    name: str = "completed"

    def __str__(self):
        return self.name


@pytest.fixture
def mock_vllm_config():
    """Create a mock VllmConfig for testing."""
    model_config = Mock(spec=ModelConfig)
    model_config.served_model_name = "test-model"

    cache_config = Mock(spec=CacheConfig)
    cache_config.num_gpu_blocks = 1000

    scheduler_config = Mock(spec=SchedulerConfig)

    observability_config = ObservabilityConfig()

    config = Mock(spec=VllmConfig)
    config.model_config = model_config
    config.cache_config = cache_config
    config.scheduler_config = scheduler_config
    config.observability_config = observability_config

    return config


@pytest.fixture(autouse=True)
def mock_otel_imports():
    """Mock OpenTelemetry imports for all tests."""
    # Mock the OpenTelemetry modules
    mock_meter = MagicMock()
    mock_meter_provider = MagicMock()
    mock_exporter = MagicMock()
    mock_reader = MagicMock()
    mock_resource = MagicMock()

    # Create mock instruments
    mock_counter = MagicMock()
    mock_gauge = MagicMock()
    mock_histogram = MagicMock()

    mock_meter.create_counter.return_value = mock_counter
    mock_meter.create_observable_gauge.return_value = mock_gauge
    mock_meter.create_histogram.return_value = mock_histogram

    mock_resource.create.return_value = mock_resource

    patches = [
        patch(
            "vllm.v1.metrics.opentelemetry_metrics.metrics.get_meter",
            return_value=mock_meter,
        ),
        patch("vllm.v1.metrics.opentelemetry_metrics.metrics.set_meter_provider"),
        patch(
            "vllm.v1.metrics.opentelemetry_metrics.OTLPMetricExporter",
            return_value=mock_exporter,
        ),
        patch(
            "vllm.v1.metrics.opentelemetry_metrics.PeriodicExportingMetricReader",
            return_value=mock_reader,
        ),
        patch(
            "vllm.v1.metrics.opentelemetry_metrics.MeterProvider",
            return_value=mock_meter_provider,
        ),
        patch(
            "vllm.v1.metrics.opentelemetry_metrics.Resource", create=mock_resource
        ),
        patch(
            "vllm.v1.metrics.opentelemetry_metrics.is_otel_metrics_available",
            return_value=True,
        ),
    ]

    for p in patches:
        p.start()

    yield {
        "meter": mock_meter,
        "meter_provider": mock_meter_provider,
        "exporter": mock_exporter,
        "reader": mock_reader,
        "resource": mock_resource,
        "counter": mock_counter,
        "gauge": mock_gauge,
        "histogram": mock_histogram,
    }

    for p in patches:
        p.stop()


class TestOpenTelemetryMetricsLogger:
    """Test OpenTelemetryMetricsLogger."""

    def test_initialization_default_config(self, mock_vllm_config, mock_otel_imports):
        """Test logger initialization with default configuration."""
        from vllm.v1.metrics.opentelemetry_metrics import OpenTelemetryMetricsLogger

        logger = OpenTelemetryMetricsLogger(
            vllm_config=mock_vllm_config, engine_indexes=[0]
        )

        # Verify meter was created
        assert logger.meter is not None
        assert logger.engine_indexes == [0]
        assert logger.common_attributes["model_name"] == "test-model"

        # Verify instruments were created
        assert mock_otel_imports["meter"].create_counter.called
        assert mock_otel_imports["meter"].create_observable_gauge.called
        assert mock_otel_imports["meter"].create_histogram.called

    def test_initialization_with_custom_endpoint(
        self, mock_vllm_config, mock_otel_imports
    ):
        """Test logger initialization with custom OTLP endpoint."""
        from vllm.v1.metrics.opentelemetry_metrics import OpenTelemetryMetricsLogger

        custom_endpoint = "http://custom-collector:4317"
        with patch.dict(
            os.environ,
            {"OTEL_EXPORTER_OTLP_METRICS_ENDPOINT": custom_endpoint},
        ):
            logger = OpenTelemetryMetricsLogger(
                vllm_config=mock_vllm_config, engine_indexes=[0, 1]
            )

            assert logger.engine_indexes == [0, 1]
            # Logger should be created successfully
            assert logger is not None

    def test_initialization_without_otel_raises_error(self, mock_vllm_config):
        """Test that initialization without OpenTelemetry raises error."""
        from vllm.v1.metrics.opentelemetry_metrics import OpenTelemetryMetricsLogger

        with patch(
            "vllm.v1.metrics.opentelemetry_metrics.is_otel_metrics_available",
            return_value=False,
        ):
            with pytest.raises(ValueError) as exc_info:
                OpenTelemetryMetricsLogger(
                    vllm_config=mock_vllm_config, engine_indexes=[0]
                )

            assert "OpenTelemetry Metrics SDK is not available" in str(exc_info.value)

    def test_record_scheduler_stats(self, mock_vllm_config, mock_otel_imports):
        """Test recording scheduler stats."""
        from vllm.v1.metrics.opentelemetry_metrics import OpenTelemetryMetricsLogger

        logger = OpenTelemetryMetricsLogger(
            vllm_config=mock_vllm_config, engine_indexes=[0]
        )

        # Create scheduler stats
        prefix_cache_stats = PrefixCacheStats(queries=100, hits=80)
        scheduler_stats = SchedulerStats(
            num_running_reqs=5,
            num_waiting_reqs=3,
            kv_cache_usage=0.75,
            prefix_cache_stats=prefix_cache_stats,
        )

        # Record stats
        logger.record(
            scheduler_stats=scheduler_stats,
            iteration_stats=None,
            mm_cache_stats=None,
            engine_idx=0,
        )

        # Verify prefix cache stats were recorded
        assert logger.counter_prefix_cache_queries.add.called
        assert logger.counter_prefix_cache_hits.add.called

        # Check that the scheduler stats were stored for gauge callbacks
        assert 0 in logger._latest_scheduler_stats
        assert logger._latest_scheduler_stats[0] == scheduler_stats

    def test_record_iteration_stats(self, mock_vllm_config, mock_otel_imports):
        """Test recording iteration stats."""
        from vllm.v1.metrics.opentelemetry_metrics import OpenTelemetryMetricsLogger

        logger = OpenTelemetryMetricsLogger(
            vllm_config=mock_vllm_config, engine_indexes=[0]
        )

        # Create iteration stats
        iteration_stats = IterationStats()
        iteration_stats.num_prompt_tokens = 50
        iteration_stats.num_generation_tokens = 100
        iteration_stats.num_preempted_reqs = 2
        iteration_stats.time_to_first_tokens_iter = [0.1, 0.2, 0.15]
        iteration_stats.inter_token_latencies_iter = [0.01, 0.02, 0.015]

        # Record stats
        logger.record(
            scheduler_stats=None,
            iteration_stats=iteration_stats,
            mm_cache_stats=None,
            engine_idx=0,
        )

        # Verify counters were called
        assert logger.counter_prompt_tokens.add.called
        assert logger.counter_generation_tokens.add.called
        assert logger.counter_num_preempted_reqs.add.called

        # Verify histograms were recorded (at least once each)
        assert logger.histogram_time_to_first_token.record.call_count >= 1
        assert logger.histogram_inter_token_latency.record.call_count >= 1

    def test_record_finished_requests(self, mock_vllm_config, mock_otel_imports):
        """Test recording finished request stats."""
        from vllm.v1.metrics.opentelemetry_metrics import OpenTelemetryMetricsLogger

        logger = OpenTelemetryMetricsLogger(
            vllm_config=mock_vllm_config, engine_indexes=[0]
        )

        # Create finished request stats
        finished_request = FinishedRequestStats(
            finish_reason=MockFinishReason("completed"),
            e2e_latency=1.5,
            num_prompt_tokens=50,
            num_generation_tokens=100,
            queued_time=0.1,
            inference_time=1.4,
        )

        iteration_stats = IterationStats()
        iteration_stats.finished_requests = [finished_request]

        # Record stats
        logger.record(
            scheduler_stats=None,
            iteration_stats=iteration_stats,
            mm_cache_stats=None,
            engine_idx=0,
        )

        # Verify request success counter was called
        assert logger.counter_request_success.add.called

        # Verify latency histograms were recorded
        assert logger.histogram_e2e_time_request.record.called
        assert logger.histogram_queue_time_request.record.called
        assert logger.histogram_inference_time_request.record.called

    def test_gauge_callbacks(self, mock_vllm_config, mock_otel_imports):
        """Test observable gauge callbacks."""
        from vllm.v1.metrics.opentelemetry_metrics import OpenTelemetryMetricsLogger

        logger = OpenTelemetryMetricsLogger(
            vllm_config=mock_vllm_config, engine_indexes=[0, 1]
        )

        # Create scheduler stats for multiple engines
        scheduler_stats_0 = SchedulerStats(
            num_running_reqs=5, num_waiting_reqs=3, kv_cache_usage=0.75
        )
        scheduler_stats_1 = SchedulerStats(
            num_running_reqs=8, num_waiting_reqs=2, kv_cache_usage=0.60
        )

        logger._latest_scheduler_stats[0] = scheduler_stats_0
        logger._latest_scheduler_stats[1] = scheduler_stats_1

        # Test running requests callback
        observations = logger._observe_running_requests(None)
        assert len(observations) == 2

        # Test waiting requests callback
        observations = logger._observe_waiting_requests(None)
        assert len(observations) == 2

        # Test KV cache usage callback
        observations = logger._observe_kv_cache_usage(None)
        assert len(observations) == 2

    def test_log_engine_initialized(self, mock_vllm_config, mock_otel_imports):
        """Test engine initialization logging."""
        from vllm.v1.metrics.opentelemetry_metrics import OpenTelemetryMetricsLogger

        logger = OpenTelemetryMetricsLogger(
            vllm_config=mock_vllm_config, engine_indexes=[0]
        )

        # Should not raise
        logger.log_engine_initialized()

    def test_multiple_engines(self, mock_vllm_config, mock_otel_imports):
        """Test recording metrics for multiple engines."""
        from vllm.v1.metrics.opentelemetry_metrics import OpenTelemetryMetricsLogger

        logger = OpenTelemetryMetricsLogger(
            vllm_config=mock_vllm_config, engine_indexes=[0, 1, 2]
        )

        # Create stats for different engines
        for engine_idx in [0, 1, 2]:
            scheduler_stats = SchedulerStats(
                num_running_reqs=engine_idx + 1,
                num_waiting_reqs=engine_idx,
                kv_cache_usage=0.5 + engine_idx * 0.1,
            )

            iteration_stats = IterationStats()
            iteration_stats.num_prompt_tokens = 10 * (engine_idx + 1)
            iteration_stats.num_generation_tokens = 20 * (engine_idx + 1)

            logger.record(
                scheduler_stats=scheduler_stats,
                iteration_stats=iteration_stats,
                mm_cache_stats=None,
                engine_idx=engine_idx,
            )

        # Verify all engines' stats were stored
        assert len(logger._latest_scheduler_stats) == 3
        assert all(idx in logger._latest_scheduler_stats for idx in [0, 1, 2])

    def test_custom_service_name(self, mock_vllm_config, mock_otel_imports):
        """Test custom service name configuration."""
        from vllm.v1.metrics.opentelemetry_metrics import OpenTelemetryMetricsLogger

        custom_service_name = "my-vllm-service"
        with patch.dict(
            os.environ,
            {"OTEL_SERVICE_NAME": custom_service_name},
        ):
            logger = OpenTelemetryMetricsLogger(
                vllm_config=mock_vllm_config, engine_indexes=[0]
            )

            # Logger should be created successfully
            assert logger is not None

    def test_custom_export_interval(self, mock_vllm_config, mock_otel_imports):
        """Test custom export interval configuration."""
        from vllm.v1.metrics.opentelemetry_metrics import OpenTelemetryMetricsLogger

        custom_interval = "30000"  # 30 seconds
        with patch.dict(
            os.environ,
            {"OTEL_METRIC_EXPORT_INTERVAL": custom_interval},
        ):
            logger = OpenTelemetryMetricsLogger(
                vllm_config=mock_vllm_config, engine_indexes=[0]
            )

            # Logger should be created successfully
            assert logger is not None


class TestIsOtelMetricsAvailable:
    """Test is_otel_metrics_available function."""

    def test_function_exists_and_returns_bool(self):
        """Test that is_otel_metrics_available exists and returns a boolean."""
        from vllm.v1.metrics.opentelemetry_metrics import is_otel_metrics_available

        result = is_otel_metrics_available()
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
