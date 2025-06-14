# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ObservabilityConfig, VllmConfig
from vllm.v1.core.kv_cache_utils import PrefixCachingMetrics
from vllm.v1.metrics.loggers import LoggingStatLogger


def test_observability_config_prefix_cache_metrics_validation():
    """Test that ObservabilityConfig validates
    prefix_cache_metrics_max_requests."""
    # Test valid values
    config = ObservabilityConfig(prefix_cache_metrics_max_requests=500)
    assert config.prefix_cache_metrics_max_requests == 500

    config = ObservabilityConfig(prefix_cache_metrics_max_requests=1000)
    assert config.prefix_cache_metrics_max_requests == 1000

    # Test default value
    config = ObservabilityConfig()
    assert config.prefix_cache_metrics_max_requests == 1000

    # Test invalid values
    with pytest.raises(ValueError, match="must be between 1 and 100000"):
        ObservabilityConfig(prefix_cache_metrics_max_requests=0)

    with pytest.raises(ValueError, match="must be between 1 and 100000"):
        ObservabilityConfig(prefix_cache_metrics_max_requests=-1)

    with pytest.raises(ValueError, match="must be between 1 and 100000"):
        ObservabilityConfig(prefix_cache_metrics_max_requests=100001)


def test_prefix_caching_metrics_configurable_interval():
    """Test that PrefixCachingMetrics respects the configurable interval."""
    # Test with custom interval
    metrics = PrefixCachingMetrics(max_recent_requests=500)
    assert metrics.max_recent_requests == 500

    # Test with default interval
    metrics = PrefixCachingMetrics()
    assert metrics.max_recent_requests == 1000


def test_logging_stat_logger_uses_config():
    """Test that LoggingStatLogger uses the configured prefix cache metrics
    interval."""
    # Create VllmConfig with custom observability config
    observability_config = ObservabilityConfig(
        prefix_cache_metrics_max_requests=2000)
    vllm_config = VllmConfig(observability_config=observability_config)

    # Create logger
    logger = LoggingStatLogger(vllm_config)

    # Verify it uses the configured value
    assert logger.prefix_caching_metrics.max_recent_requests == 2000

    # Test with None observability config (should use default)
    vllm_config_none = VllmConfig(observability_config=None)
    logger_none = LoggingStatLogger(vllm_config_none)
    assert logger_none.prefix_caching_metrics.max_recent_requests == 1000
