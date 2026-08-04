# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock

# Third Party
from prometheus_client import REGISTRY
import pytest

# First Party
from lmcache.observability import (
    LMCStatsMonitor,
    PrometheusLogger,
)


@pytest.fixture(scope="function")
def stats_monitor():
    LMCStatsMonitor.DestroyInstance()
    return LMCStatsMonitor.GetOrCreate()


def test_on_retrieve_request(stats_monitor):
    stats_monitor.on_retrieve_request(num_tokens=100)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_retrieve_requests == 1
    assert stats.retrieve_hit_rate == 1.0
    assert stats.local_cache_usage_bytes == 0
    assert stats.remote_cache_usage_bytes == 0
    assert len(stats.time_to_retrieve) == 0


def test_on_retrieve_finished(stats_monitor):
    stats_obj = stats_monitor.on_retrieve_request(num_tokens=100)
    stats_monitor.on_retrieve_finished(
        retrieve_stats=stats_obj,
        num_retrieved_tokens=100,
    )
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_retrieve_requests == 1
    assert stats.retrieve_hit_rate == 1.0
    assert len(stats.time_to_retrieve) == 1


def test_on_store_request_and_finished(stats_monitor):
    stats_obj = stats_monitor.on_store_request(num_tokens=50)
    stats_monitor.on_store_finished(store_stats=stats_obj)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_store_requests == 1
    assert len(stats.time_to_store) == 1


def test_update_local_cache_usage(stats_monitor):
    stats_monitor.update_local_cache_usage(usage=1024)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.local_cache_usage_bytes == 1024


def test_update_remote_cache_usage(stats_monitor):
    stats_monitor.update_remote_cache_usage(usage=2048)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.remote_cache_usage_bytes == 2048


def test_update_local_storage_usage(stats_monitor):
    stats_monitor.update_local_storage_usage(usage=4096)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.local_storage_usage_bytes == 4096


def test_on_lookup_request(stats_monitor):
    stats_monitor.on_lookup_request(num_tokens=50)
    assert len(stats_monitor.lookup_requests) == 1
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_lookup_requests == 1
    assert stats.interval_lookup_tokens == 50
    assert stats.lookup_hit_rate == 0
    assert len(stats.interval_lookup_hit_rates) == 0
    # on_lookup_finished is not called, lookup_requests is not clear
    assert len(stats_monitor.lookup_requests) == 1


def test_on_lookup_finished(stats_monitor):
    stats_obj = stats_monitor.on_lookup_request(num_tokens=100)
    assert len(stats_monitor.lookup_requests) == 1
    stats_monitor.on_lookup_finished(stats=stats_obj, num_hit_tokens=80)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_lookup_requests == 1
    assert stats.interval_lookup_tokens == 100
    assert stats.interval_lookup_hits == 80
    assert stats.lookup_hit_rate == 0.8
    assert len(stats.interval_lookup_hit_rates) == 1
    assert stats.interval_lookup_hit_rates[0] == 0.8
    assert len(stats_monitor.lookup_requests) == 0


def test_remote_read_metrics(stats_monitor):
    stats_monitor.update_interval_remote_read_metrics(read_bytes=1024)
    stats_monitor.update_interval_remote_read_metrics(read_bytes=2048)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_remote_read_requests == 2
    assert stats.interval_remote_read_bytes == 3072


def test_remote_write_metrics(stats_monitor):
    stats_monitor.update_interval_remote_write_metrics(write_bytes=512)
    stats_monitor.update_interval_remote_write_metrics(write_bytes=1024)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_remote_write_requests == 2
    assert stats.interval_remote_write_bytes == 1536


def test_remote_time_metrics(stats_monitor):
    stats_monitor.update_interval_remote_time_to_get(get_time=10.5)
    stats_monitor.update_interval_remote_time_to_put(put_time=15.2)
    stats_monitor.update_interval_remote_time_to_get_sync(get_time_sync=12.3)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_remote_time_to_get == [10.5]
    assert stats.interval_remote_time_to_put == [15.2]
    assert stats.interval_remote_time_to_get_sync == [12.3]


def test_remote_ping_metrics(stats_monitor):
    # Test successful ping
    stats_monitor.update_remote_ping_latency(latency=25.5)
    stats_monitor.update_remote_ping_error_code(error_code=0)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_remote_ping_latency == 25.5
    assert stats.interval_remote_ping_success == 1
    assert stats.interval_remote_ping_errors == 0
    assert stats.interval_remote_ping_error_code == 0


def test_remote_ping_errors(stats_monitor):
    # Test ping errors
    stats_monitor.update_remote_ping_error_code(error_code=404)
    stats_monitor.update_remote_ping_error_code(error_code=500)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_remote_ping_errors == 2
    assert stats.interval_remote_ping_success == 0
    assert stats.interval_remote_ping_error_code == 500


def test_retrieve_and_store_speed(stats_monitor):
    # Test retrieve speed calculation
    stats_obj_retrieve = stats_monitor.on_retrieve_request(num_tokens=1000)
    stats_monitor.on_retrieve_finished(
        retrieve_stats=stats_obj_retrieve, num_retrieved_tokens=1000
    )

    # Test store speed calculation
    stats_obj_store = stats_monitor.on_store_request(num_tokens=500)
    stats_monitor.on_store_finished(store_stats=stats_obj_store)

    stats = stats_monitor.get_stats_and_clear()
    assert len(stats.retrieve_speed) == 1
    assert len(stats.store_speed) == 1
    assert stats.retrieve_speed[0] > 0  # Should be tokens/second
    assert stats.store_speed[0] > 0  # Should be tokens/second


def test_multiple_lookup_operations(stats_monitor):
    # Test multiple lookup operations
    stats_obj_1 = stats_monitor.on_lookup_request(num_tokens=100)
    stats_monitor.on_lookup_finished(stats=stats_obj_1, num_hit_tokens=80)
    stats_obj_2 = stats_monitor.on_lookup_request(num_tokens=200)
    stats_monitor.on_lookup_finished(stats=stats_obj_2, num_hit_tokens=150)
    assert len(stats_monitor.lookup_requests) == 2
    assert stats_monitor.lookup_requests[stats_obj_1.request_id].hit_rate() == 0.8
    assert stats_monitor.lookup_requests[stats_obj_2.request_id].hit_rate() == 0.75

    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_lookup_requests == 2
    assert stats.interval_lookup_tokens == 300
    assert stats.interval_lookup_hits == 230
    assert stats.lookup_hit_rate == 230 / 300
    assert len(stats.interval_lookup_hit_rates) == 2
    assert len(stats_monitor.lookup_requests) == 0


def test_mixed_remote_operations(stats_monitor):
    # Test a mix of remote operations
    stats_monitor.update_interval_remote_read_metrics(read_bytes=1024)
    stats_monitor.update_interval_remote_write_metrics(write_bytes=512)
    stats_monitor.update_interval_remote_time_to_get(get_time=10.0)
    stats_monitor.update_interval_remote_time_to_put(put_time=20.0)
    stats_monitor.update_interval_remote_time_to_get_sync(get_time_sync=15.0)
    stats_monitor.update_remote_ping_latency(latency=30.0)
    stats_monitor.update_remote_ping_error_code(error_code=0)

    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_remote_read_requests == 1
    assert stats.interval_remote_read_bytes == 1024
    assert stats.interval_remote_write_requests == 1
    assert stats.interval_remote_write_bytes == 512
    assert stats.interval_remote_time_to_get == [10.0]
    assert stats.interval_remote_time_to_put == [20.0]
    assert stats.interval_remote_time_to_get_sync == [15.0]
    assert stats.interval_remote_ping_latency == 30.0
    assert stats.interval_remote_ping_success == 1
    assert stats.interval_remote_ping_errors == 0


def test_combined_operations(stats_monitor):
    stats_obj_retrieve = stats_monitor.on_retrieve_request(num_tokens=200)
    stats_monitor.on_retrieve_finished(
        retrieve_stats=stats_obj_retrieve,
        num_retrieved_tokens=200,
    )
    stats_obj_store = stats_monitor.on_store_request(num_tokens=100)
    stats_monitor.on_store_finished(store_stats=stats_obj_store)
    stats_monitor.update_local_cache_usage(usage=512)
    stats_monitor.update_remote_cache_usage(usage=1024)
    stats_monitor.update_local_storage_usage(usage=2048)

    stats_monitor2 = LMCStatsMonitor.GetOrCreate()
    stats = stats_monitor2.get_stats_and_clear()

    assert stats.interval_retrieve_requests == 1
    assert stats.interval_store_requests == 1
    assert stats.retrieve_hit_rate == 1.0
    assert stats.local_cache_usage_bytes == 512
    assert stats.remote_cache_usage_bytes == 1024
    assert stats.local_storage_usage_bytes == 2048
    assert len(stats.time_to_retrieve) == 1
    assert len(stats.time_to_store) == 1


def test_stats_clearing(stats_monitor):
    # Add some data
    stats_obj = stats_monitor.on_lookup_request(num_tokens=100)
    stats_monitor.update_interval_remote_read_metrics(read_bytes=1024)
    stats_monitor.update_remote_ping_latency(latency=25.0)

    assert len(stats_monitor.lookup_requests) == 1

    # Get stats (which should clear them)
    stats = stats_monitor.get_stats_and_clear()
    assert stats.interval_lookup_requests == 1
    assert stats.interval_lookup_tokens == 100
    assert stats.interval_lookup_hits == 0
    assert stats.interval_remote_read_requests == 1
    assert stats.interval_remote_read_bytes == 1024
    assert stats.interval_remote_ping_latency == 25.0
    assert len(stats.interval_lookup_hit_rates) == 0

    # Get stats again - should be cleared
    stats2 = stats_monitor.get_stats_and_clear()
    assert stats2.interval_lookup_requests == 0
    assert stats2.interval_lookup_tokens == 0
    assert stats2.interval_lookup_hits == 0
    assert stats2.interval_remote_read_requests == 0
    assert stats2.interval_remote_read_bytes == 0
    assert stats2.interval_remote_ping_latency == 0
    assert len(stats2.interval_lookup_hit_rates) == 0

    assert len(stats_monitor.lookup_requests) == 1

    # finish lookup request
    stats_monitor.on_lookup_finished(stats=stats_obj, num_hit_tokens=80)
    stats3 = stats_monitor.get_stats_and_clear()
    assert stats3.interval_lookup_requests == 0
    assert stats3.interval_lookup_tokens == 0
    assert stats3.interval_lookup_hits == 80
    assert stats3.interval_remote_read_requests == 0
    assert stats3.interval_remote_read_bytes == 0
    assert stats3.interval_remote_ping_latency == 0
    assert len(stats3.interval_lookup_hit_rates) == 1
    assert stats3.interval_lookup_hit_rates[0] == 0.8

    assert len(stats_monitor.lookup_requests) == 0


def test_zero_division_protection(stats_monitor):
    # Test that hit rates handle zero division gracefully
    stats = stats_monitor.get_stats_and_clear()
    assert stats.retrieve_hit_rate == 1.0
    assert stats.lookup_hit_rate == 0


# ---- PrometheusLogger custom histogram bucket tests ----


@pytest.fixture(scope="function")
def _cleanup_prometheus_logger():
    """Reset PrometheusLogger instances and metrics between tests."""
    LMCStatsMonitor.unregister_all_metrics()
    PrometheusLogger._instances = {}
    yield
    LMCStatsMonitor.unregister_all_metrics()
    PrometheusLogger._instances = {}


def _make_metadata():
    """Create a minimal mock LMCacheMetadata for testing."""
    meta = MagicMock()
    meta.model_name = "test_model"
    meta.worker_id = 0
    meta.role = "worker"
    meta.served_model_name = None
    return meta


def _make_config(extra_config=None):
    """Create a minimal mock config for testing."""
    cfg = MagicMock()
    cfg.extra_config = extra_config
    cfg.get_extra_config_value = lambda key, default=None: (
        extra_config.get(key, default) if extra_config is not None else default
    )
    return cfg


def _sample_labels(prom: PrometheusLogger) -> dict[str, str]:
    """Return labels as Prometheus exposes them in collected samples."""
    return {name: str(value) for name, value in prom.labels.items()}


def test_prometheus_logger_default_buckets(_cleanup_prometheus_logger):
    """Histogram should use default buckets when no config."""
    meta = _make_metadata()
    prom = PrometheusLogger(meta)
    # Verify histogram was created (basic sanity check)
    assert prom.histogram_time_to_retrieve is not None


def test_prometheus_logger_custom_buckets(_cleanup_prometheus_logger):
    """Histogram should use custom buckets from config."""
    meta = _make_metadata()
    custom_buckets = [0.1, 0.5, 1.0, 5.0]
    cfg = _make_config(
        {
            "histogram_bucket_time_to_retrieve": custom_buckets,
        }
    )
    prom = PrometheusLogger(meta, config=cfg)
    labels = prom.labels

    # Verify the histogram's upper_bounds match custom buckets + Inf
    hist = prom.histogram_time_to_retrieve.labels(**labels)
    # prometheus_client Histogram stores buckets as _upper_bounds
    upper_bounds = list(hist._upper_bounds)
    for bucket_val in custom_buckets:
        assert bucket_val in upper_bounds


def test_prometheus_logger_partial_custom_buckets(
    _cleanup_prometheus_logger,
):
    """Only specified histograms should get custom buckets."""
    meta = _make_metadata()
    custom_buckets = [1, 10, 100]
    cfg = _make_config(
        {
            "histogram_bucket_store_speed": custom_buckets,
        }
    )
    prom = PrometheusLogger(meta, config=cfg)
    labels = prom.labels

    # store_speed should have custom buckets
    hist = prom.histogram_store_speed.labels(**labels)
    upper_bounds = list(hist._upper_bounds)
    for bucket_val in custom_buckets:
        assert bucket_val in upper_bounds

    # time_to_retrieve should still have default buckets
    hist_default = prom.histogram_time_to_retrieve.labels(**labels)
    default_upper = list(hist_default._upper_bounds)
    # default first bucket is 0.001
    assert 0.001 in default_upper


def test_prometheus_logger_get_or_create_with_config(
    _cleanup_prometheus_logger,
):
    """GetOrCreate should pass config to the constructor."""
    meta = _make_metadata()
    custom_buckets = [0.01, 0.1, 1.0]
    cfg = _make_config(
        {
            "histogram_bucket_time_to_store": custom_buckets,
        }
    )
    prom = PrometheusLogger.GetOrCreate(meta, config=cfg)
    labels = prom.labels

    hist = prom.histogram_time_to_store.labels(**labels)
    upper_bounds = list(hist._upper_bounds)
    for bucket_val in custom_buckets:
        assert bucket_val in upper_bounds


def test_prometheus_logger_get_or_create_allows_multiple_roles(
    _cleanup_prometheus_logger,
):
    """Different roles should not be treated as global metadata conflicts."""
    worker_meta = _make_metadata()
    scheduler_meta = _make_metadata()
    scheduler_meta.role = "scheduler"

    worker_prom = PrometheusLogger.GetOrCreate(worker_meta)
    scheduler_prom = PrometheusLogger.GetOrCreate(scheduler_meta)

    assert worker_prom.labels["role"] == "worker"
    assert scheduler_prom.labels["role"] == "scheduler"


def test_prometheus_logger_label_view_rebinds_dynamic_metrics(
    _cleanup_prometheus_logger,
):
    """Dynamic gauge callbacks should use each label view's labels."""
    worker_meta = _make_metadata()
    scheduler_meta = _make_metadata()
    scheduler_meta.role = "scheduler"

    worker_prom = PrometheusLogger.GetOrCreate(worker_meta)
    scheduler_prom = PrometheusLogger.GetOrCreate(scheduler_meta)

    worker_prom.lmcache_is_healthy.set_function(lambda: 1)
    scheduler_prom.lmcache_is_healthy.set_function(lambda: 2)

    assert (
        REGISTRY.get_sample_value(
            "lmcache:lmcache_is_healthy",
            _sample_labels(worker_prom),
        )
        == 1
    )
    assert (
        REGISTRY.get_sample_value(
            "lmcache:lmcache_is_healthy",
            _sample_labels(scheduler_prom),
        )
        == 2
    )


def test_prometheus_logger_reset_reinitializes_all_label_views(
    _cleanup_prometheus_logger,
):
    """Resetting one label view should keep metric children visible for all views."""
    worker_meta = _make_metadata()
    scheduler_meta = _make_metadata()
    scheduler_meta.role = "scheduler"

    worker_prom = PrometheusLogger.GetOrCreate(worker_meta)
    scheduler_prom = PrometheusLogger.GetOrCreate(scheduler_meta)

    worker_prom.counter_num_retrieve_requests.labels(**worker_prom.labels).inc()
    scheduler_prom.counter_num_retrieve_requests.labels(**scheduler_prom.labels).inc()
    scheduler_prom.reset_counters()

    assert (
        REGISTRY.get_sample_value(
            "lmcache:num_retrieve_requests_total",
            _sample_labels(worker_prom),
        )
        == 0
    )
    assert (
        REGISTRY.get_sample_value(
            "lmcache:num_retrieve_requests_total",
            _sample_labels(scheduler_prom),
        )
        == 0
    )
