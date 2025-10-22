# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from unittest.mock import MagicMock, call, patch

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.metrics.stats import KVCacheLifetimeStats


class TestKVCacheLifetimeStats:
    """Test the KVCacheLifetimeStats class."""

    def test_initial_state(self):
        """Test that lifetime stats start with correct initial values."""
        stats = KVCacheLifetimeStats()
        assert stats.total_blocks_freed == 0
        assert stats.total_lifetime_seconds == 0.0
        assert stats.average_lifetime_seconds == 0.0

    def test_add_single_lifetime(self):
        """Test adding a single block lifetime."""
        stats = KVCacheLifetimeStats()
        stats.add_block_lifetime(5.0)

        assert stats.total_blocks_freed == 1
        assert stats.total_lifetime_seconds == 5.0
        assert stats.average_lifetime_seconds == 5.0

    def test_add_multiple_lifetimes(self):
        """Test adding multiple block lifetimes and average calculation."""
        stats = KVCacheLifetimeStats()
        stats.add_block_lifetime(2.0)
        stats.add_block_lifetime(4.0)
        stats.add_block_lifetime(6.0)

        assert stats.total_blocks_freed == 3
        assert stats.total_lifetime_seconds == 12.0
        assert stats.average_lifetime_seconds == 4.0

    def test_reset(self):
        """Test resetting lifetime statistics."""
        stats = KVCacheLifetimeStats()
        stats.add_block_lifetime(5.0)
        stats.add_block_lifetime(3.0)

        stats.reset()

        assert stats.total_blocks_freed == 0
        assert stats.total_lifetime_seconds == 0.0
        assert stats.average_lifetime_seconds == 0.0


class TestKVCacheBlockLifetime:
    """Test KVCacheBlock allocation time tracking."""

    def test_initial_allocation_time(self):
        """Test that blocks start with no allocation time."""
        block = KVCacheBlock(block_id=0)
        assert block.allocation_time is None

    def test_allocation_time_tracking(self):
        """Test that allocation time can be set and retrieved."""
        block = KVCacheBlock(block_id=0)
        current_time = time.time()
        block.allocation_time = current_time
        assert block.allocation_time == current_time


class TestBlockPoolLifetimeTracking:
    """Test BlockPool lifetime tracking functionality."""

    def test_block_pool_initialization(self):
        """Test that BlockPool initializes with lifetime stats."""
        pool = BlockPool(num_gpu_blocks=10,
                         enable_caching=True,
                         enable_kv_cache_events=False)
        assert hasattr(pool, 'lifetime_stats')
        assert isinstance(pool.lifetime_stats, KVCacheLifetimeStats)

    @patch('time.monotonic')
    def test_allocation_time_recording(self, mock_time):
        """Test that block allocation times are recorded."""
        mock_time.return_value = 100.0

        pool = BlockPool(num_gpu_blocks=10,
                         enable_caching=False,
                         enable_kv_cache_events=False)

        # Get blocks from the pool
        blocks = pool.get_new_blocks(2)

        # Verify allocation times were set
        for block in blocks:
            assert block.allocation_time == 100.0

    @patch('time.monotonic')
    def test_lifetime_calculation_on_free(self, mock_time):
        """Test that lifetimes are calculated when blocks are freed."""
        # Mock time progression
        allocation_time = 100.0
        free_time = 105.0
        expected_lifetime = free_time - allocation_time

        mock_time.side_effect = [allocation_time, free_time]

        pool = BlockPool(num_gpu_blocks=10,
                         enable_caching=False,
                         enable_kv_cache_events=False)

        # Allocate blocks
        blocks = pool.get_new_blocks(2)

        # Reset mock to return free time
        mock_time.return_value = free_time

        # Free the blocks
        pool.free_blocks(blocks)

        # Verify lifetime was recorded
        stats = pool.get_lifetime_stats()
        assert stats.total_blocks_freed == 2
        assert stats.average_lifetime_seconds == expected_lifetime

    @patch('time.monotonic')
    def test_null_block_lifetime_ignored(self, mock_time):
        """Test that null blocks don't contribute to lifetime stats."""
        mock_time.side_effect = [100.0, 105.0]

        pool = BlockPool(num_gpu_blocks=10,
                         enable_caching=False,
                         enable_kv_cache_events=False)

        # The null block is automatically allocated during initialization
        # Free the null block (should not affect stats)
        pool.free_blocks([pool.null_block])

        stats = pool.get_lifetime_stats()
        assert stats.total_blocks_freed == 0

    def test_lifetime_stats_retrieval(self):
        """Test that lifetime statistics can be retrieved."""
        pool = BlockPool(num_gpu_blocks=10,
                         enable_caching=True,
                         enable_kv_cache_events=False)

        stats = pool.get_lifetime_stats()
        assert isinstance(stats, KVCacheLifetimeStats)

    def test_lifetime_stats_reset(self):
        """Test that lifetime statistics can be reset."""
        pool = BlockPool(num_gpu_blocks=10,
                         enable_caching=True,
                         enable_kv_cache_events=False)

        # Add some fake lifetime data
        pool.lifetime_stats.add_block_lifetime(5.0)
        assert pool.lifetime_stats.total_blocks_freed == 1

        # Reset stats
        pool.reset_lifetime_stats()
        assert pool.lifetime_stats.total_blocks_freed == 0

    @patch('time.monotonic')
    def test_collect_recent_lifetimes(self, mock_time):
        """Collected lifetimes should reflect recent frees only once."""
        mock_time.side_effect = [100.0, 110.0]

        pool = BlockPool(num_gpu_blocks=4,
                         enable_caching=False,
                         enable_kv_cache_events=False)

        blocks = pool.get_new_blocks(1)
        mock_time.return_value = 110.0

        pool.free_blocks(blocks)

        lifetimes = pool.collect_recent_lifetimes()
        assert lifetimes == [10.0]
        assert pool.collect_recent_lifetimes() == []

    def test_reset_prefix_cache_resets_lifetime_stats(self):
        """Resetting the prefix cache should also clear lifetime stats."""

        pool = BlockPool(num_gpu_blocks=10,
                         enable_caching=True,
                         enable_kv_cache_events=False)

        pool.lifetime_stats.add_block_lifetime(7.5)
        assert pool.lifetime_stats.total_blocks_freed == 1

        assert pool.reset_prefix_cache() is True
        assert pool.lifetime_stats.total_blocks_freed == 0


class TestKVCacheManagerLifetimeIntegration:
    """Test KVCacheManager integration with lifetime tracking."""

    def test_kv_cache_manager_exposes_lifetime_stats(self):
        """Test that KVCacheManager can retrieve lifetime statistics."""
        with patch('vllm.v1.core.kv_cache_manager.get_kv_cache_coordinator'):
            from vllm.v1.core.kv_cache_manager import KVCacheManager

            # Create a minimal KVCacheConfig for testing
            kv_config = MagicMock()
            kv_config.kv_cache_groups = []

            with patch.object(KVCacheManager, '__init__', return_value=None):
                manager = KVCacheManager.__new__(KVCacheManager)

                # Mock the block pool
                mock_pool = MagicMock()
                mock_stats = KVCacheLifetimeStats()
                mock_pool.get_lifetime_stats.return_value = mock_stats
                manager.block_pool = mock_pool

                # Test stats retrieval
                stats = manager.get_kv_cache_lifetime_stats()
                assert stats == mock_stats
                mock_pool.get_lifetime_stats.assert_called_once()

    def test_kv_cache_manager_resets_lifetime_stats(self):
        """Test that KVCacheManager can reset lifetime statistics."""
        with patch('vllm.v1.core.kv_cache_manager.get_kv_cache_coordinator'):
            from vllm.v1.core.kv_cache_manager import KVCacheManager

            with patch.object(KVCacheManager, '__init__', return_value=None):
                manager = KVCacheManager.__new__(KVCacheManager)

                # Mock the block pool
                mock_pool = MagicMock()
                manager.block_pool = mock_pool

                # Test stats reset
                manager.reset_kv_cache_lifetime_stats()
                mock_pool.reset_lifetime_stats.assert_called_once()

    def test_collect_recent_kv_cache_lifetimes(self):
        """KVCacheManager should proxy recent lifetime collection."""
        with patch('vllm.v1.core.kv_cache_manager.get_kv_cache_coordinator'):
            from vllm.v1.core.kv_cache_manager import KVCacheManager

            with patch.object(KVCacheManager, '__init__', return_value=None):
                manager = KVCacheManager.__new__(KVCacheManager)

                mock_pool = MagicMock()
                mock_pool.collect_recent_lifetimes.return_value = [3.0]
                manager.block_pool = mock_pool

                assert manager.collect_recent_kv_cache_lifetimes() == [3.0]
                mock_pool.collect_recent_lifetimes.assert_called_once()

    def test_reset_prefix_cache_resets_lifetime_stats(self):
        """KVCacheManager.reset_prefix_cache should clear lifetime stats."""
        with patch('vllm.v1.core.kv_cache_manager.get_kv_cache_coordinator'):
            from vllm.v1.core.kv_cache_manager import KVCacheManager

            with patch.object(KVCacheManager, '__init__', return_value=None):
                manager = KVCacheManager.__new__(KVCacheManager)

                mock_pool = MagicMock()
                mock_pool.reset_prefix_cache.return_value = True

                manager.block_pool = mock_pool
                manager.log_stats = False
                manager.prefix_cache_stats = None

                assert manager.reset_prefix_cache() is True
                mock_pool.reset_prefix_cache.assert_called_once()
                mock_pool.reset_lifetime_stats.assert_called_once()


@pytest.mark.integration
class TestPrometheusMetricIntegration:
    """Test Prometheus metric integration for KV cache lifetime."""

    def test_prometheus_metric_definition(self):
        """Test that the Prometheus metric is properly defined."""
        from vllm.v1.metrics.loggers import PrometheusStatLogger

        with patch('vllm.v1.metrics.loggers.unregister_vllm_metrics'):
            # Mock VllmConfig
            mock_config = MagicMock()
            mock_config.model_config.served_model_name = "test_model"
            mock_config.model_config.max_model_len = 1000
            mock_config.speculative_config = None
            mock_config.lora_config = None
            mock_config.observability_config.show_hidden_metrics = False

            logger = PrometheusStatLogger(mock_config, [0])

            # Verify the lifetime histogram metric exists
            assert hasattr(logger, 'histogram_kv_cache_lifetime_seconds')
            assert 0 in logger.histogram_kv_cache_lifetime_seconds

    def test_prometheus_metric_recording(self):
        """Test that lifetime statistics are recorded to Prometheus."""
        from vllm.v1.metrics.loggers import PrometheusStatLogger
        from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats

        with patch('vllm.v1.metrics.loggers.unregister_vllm_metrics'):
            # Mock VllmConfig
            mock_config = MagicMock()
            mock_config.model_config.served_model_name = "test_model"
            mock_config.model_config.max_model_len = 1000
            mock_config.speculative_config = None
            mock_config.lora_config = None
            mock_config.observability_config.show_hidden_metrics = False

            logger = PrometheusStatLogger(mock_config, [0])

            # Mock the histogram observe method
            mock_histogram = MagicMock()
            logger.histogram_kv_cache_lifetime_seconds = {0: mock_histogram}

            scheduler_stats = SchedulerStats(
                num_running_reqs=1,
                num_waiting_reqs=0,
                kv_cache_usage=0.5,
                prefix_cache_stats=PrefixCacheStats(),
                kv_cache_block_lifetimes=[10.0, 20.0])

            # Record the stats
            logger.record(scheduler_stats, None, 0)

            # Verify histogram was updated with each lifetime sample
            mock_histogram.observe.assert_has_calls([
                call(10.0),
                call(20.0),
            ])
            assert mock_histogram.observe.call_count == 2

            # Record again to ensure only deltas are recorded
            scheduler_stats.kv_cache_block_lifetimes = []
            logger.record(scheduler_stats, None, 0)

            assert mock_histogram.observe.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
