"""
Tests for eviction policy monitoring and statistics.

Tests the get_stats() method and performance metrics tracking.
"""

import pytest
import time
from unittest.mock import Mock

from vllm.v1.core.eviction_policy import (
    PagedEvictionPolicy,
    LRUEvictionPolicy,
    BlockMetadata,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock


class TestGetStatsEmpty:
    """Test get_stats() with no metadata."""

    def test_get_stats_empty_metadata(self):
        """get_stats() should handle empty metadata gracefully."""
        policy = PagedEvictionPolicy()

        stats = policy.get_stats()

        assert stats["metadata_size"] == 0
        assert (
            stats["max_access_count"] == 1
        )  # Initialized to 1 to avoid division by zero
        assert "selection_count" in stats  # Performance metrics still present


class TestGetStatsWithMetadata:
    """Test get_stats() with actual metadata."""

    def test_get_stats_basic_metrics(self):
        """get_stats() should return basic metadata metrics."""
        policy = PagedEvictionPolicy()

        # Create some metadata
        policy.metadata[1] = BlockMetadata(
            last_access_time=100.0,
            access_count=5,
            is_cached=False,
            cache_hit_count=0,
        )
        policy.metadata[2] = BlockMetadata(
            last_access_time=200.0,
            access_count=10,
            is_cached=True,
            cache_hit_count=3,
        )
        policy.max_access_count = 10
        policy.current_time = 250.0

        stats = policy.get_stats()

        # Metadata size
        assert stats["metadata_size"] == 2
        assert stats["max_access_count"] == 10

        # Access counts
        assert stats["avg_access_count"] == 7.5  # (5 + 10) / 2
        assert stats["max_block_access_count"] == 10
        assert stats["min_block_access_count"] == 5
        assert stats["median_access_count"] in [5, 10]  # Either is valid for 2 elements

    def test_get_stats_recency_metrics(self):
        """get_stats() should track recency correctly."""
        policy = PagedEvictionPolicy()

        policy.metadata[1] = BlockMetadata(last_access_time=100.0, access_count=1)
        policy.metadata[2] = BlockMetadata(last_access_time=200.0, access_count=1)
        policy.current_time = 300.0

        stats = policy.get_stats()

        # Time since access: (300-100) + (300-200) = 200 + 100 = 300
        # Average: 300 / 2 = 150
        assert stats["avg_time_since_access"] == 150.0
        assert stats["max_time_since_access"] == 200.0

    def test_get_stats_cache_metrics(self):
        """get_stats() should track cached blocks correctly."""
        policy = PagedEvictionPolicy()

        # Cached block
        policy.metadata[1] = BlockMetadata(
            access_count=1,
            is_cached=True,
            cache_hit_count=5,
        )
        # Another cached block
        policy.metadata[2] = BlockMetadata(
            access_count=1,
            is_cached=True,
            cache_hit_count=10,
        )
        # Non-cached block
        policy.metadata[3] = BlockMetadata(
            access_count=1,
            is_cached=False,
            cache_hit_count=0,
        )

        stats = policy.get_stats()

        assert stats["cached_blocks_tracked"] == 2
        assert stats["avg_cache_hit_count"] == 7.5  # (5 + 10) / 2

    def test_get_stats_configuration(self):
        """get_stats() should include configuration parameters."""
        policy = PagedEvictionPolicy(
            recency_weight=0.5,
            frequency_weight=0.3,
            cache_weight=0.2,
            time_decay=0.9,
        )

        stats = policy.get_stats()

        assert stats["recency_weight"] == 0.5
        assert stats["frequency_weight"] == 0.3
        assert stats["cache_weight"] == 0.2
        assert stats["time_decay"] == 0.9
        assert stats["is_prefill_phase"] == True  # Default


class TestPerformanceMetrics:
    """Test performance timing metrics."""

    def test_selection_timing(self):
        """select_blocks_to_evict() should track timing."""
        policy = PagedEvictionPolicy()

        # Create mock blocks
        free_blocks = [
            Mock(
                spec=KVCacheBlock, block_id=i, is_null=False, ref_cnt=0, block_hash=None
            )
            for i in range(10)
        ]
        cached_blocks = {}

        # Perform selection
        policy.select_blocks_to_evict(5, free_blocks, cached_blocks)

        stats = policy.get_stats()

        assert stats["selection_count"] == 1
        assert stats["selection_avg_time"] > 0
        assert stats["selection_min_time"] > 0
        assert stats["selection_max_time"] > 0

    def test_update_timing(self):
        """update_access() should track timing."""
        policy = PagedEvictionPolicy()

        # Create mock blocks
        blocks = [
            Mock(spec=KVCacheBlock, block_id=i, block_hash=None) for i in range(5)
        ]

        # Perform update
        policy.update_access(blocks)

        stats = policy.get_stats()

        assert stats["update_count"] == 1
        assert stats["update_avg_time"] > 0

    def test_multiple_selections_timing(self):
        """Multiple selections should aggregate timing correctly."""
        policy = PagedEvictionPolicy()

        free_blocks = [
            Mock(
                spec=KVCacheBlock, block_id=i, is_null=False, ref_cnt=0, block_hash=None
            )
            for i in range(20)
        ]
        cached_blocks = {}

        # Perform multiple selections
        policy.select_blocks_to_evict(5, free_blocks[:10], cached_blocks)
        policy.select_blocks_to_evict(5, free_blocks[10:], cached_blocks)

        stats = policy.get_stats()

        assert stats["selection_count"] == 2
        assert stats["selection_avg_time"] > 0
        # Min should be <= avg <= max
        assert (
            stats["selection_min_time"]
            <= stats["selection_avg_time"]
            <= stats["selection_max_time"]
        )

    def test_timing_edge_case_no_operations(self):
        """Stats should handle case with no operations."""
        policy = PagedEvictionPolicy()

        stats = policy.get_stats()

        # No selections yet
        assert stats["selection_count"] == 0
        assert stats["selection_avg_time"] == 0.0
        assert stats["selection_min_time"] == 0.0  # Should handle inf correctly

        # No updates yet
        assert stats["update_count"] == 0
        assert stats["update_avg_time"] == 0.0


class TestBlockPoolStats:
    """Test BlockPool.get_eviction_policy_stats()."""

    def test_block_pool_stats_basic(self):
        """get_eviction_policy_stats() should return BlockPool metrics."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=True,
            hash_block_size=16,
        )

        stats = pool.get_eviction_policy_stats()

        # Policy type
        assert stats["policy_type"] == "LRUEvictionPolicy"  # Default

        # BlockPool metrics
        assert stats["num_gpu_blocks"] == 100
        assert stats["num_free_blocks"] > 0  # Should have free blocks
        assert stats["num_allocated_blocks"] >= 0
        assert "cache_utilization" in stats

    def test_block_pool_stats_with_paged_policy(self):
        """get_eviction_policy_stats() should work with PagedEviction."""
        policy = PagedEvictionPolicy()
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=True,
            hash_block_size=16,
            eviction_policy=policy,
        )

        stats = pool.get_eviction_policy_stats()

        assert stats["policy_type"] == "PagedEvictionPolicy"

        # Should include PagedEviction metrics
        assert "metadata_size" in stats
        assert "max_access_count" in stats
        assert "selection_count" in stats

    def test_block_pool_stats_after_allocation(self):
        """Stats should update after block allocation."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=True,
            hash_block_size=16,
            eviction_policy=PagedEvictionPolicy(),
        )

        initial_stats = pool.get_eviction_policy_stats()
        initial_free = initial_stats["num_free_blocks"]

        # Allocate some blocks
        blocks = pool.get_new_blocks(10)

        after_stats = pool.get_eviction_policy_stats()

        # Free blocks should decrease
        assert after_stats["num_free_blocks"] == initial_free - 10

        # Allocated blocks should increase
        assert (
            after_stats["num_allocated_blocks"]
            == initial_stats["num_allocated_blocks"] + 10
        )

        # Selection count should increment
        assert after_stats["selection_count"] > initial_stats["selection_count"]

    def test_block_pool_stats_cached_blocks(self):
        """Stats should track cached blocks correctly."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=True,
            hash_block_size=16,
            eviction_policy=PagedEvictionPolicy(),
        )

        stats = pool.get_eviction_policy_stats()

        assert "num_cached_blocks" in stats
        assert "cached_single_blocks" in stats
        assert "cached_dict_blocks" in stats
        assert "total_cached_block_instances" in stats


class TestIntegrationWithBlockPool:
    """Integration tests with real BlockPool usage."""

    def test_stats_after_touch(self):
        """Stats should update after touching blocks."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=True,
            hash_block_size=16,
            eviction_policy=PagedEvictionPolicy(),
        )

        # Allocate blocks
        blocks = pool.get_new_blocks(5)

        # Touch them
        pool.touch((blocks,))

        stats = pool.get_eviction_policy_stats()

        # Update count should increment
        assert stats["update_count"] >= 1

        # Metadata should track these blocks
        assert stats["metadata_size"] >= 5

    def test_stats_full_lifecycle(self):
        """Stats should track complete block lifecycle."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=True,
            hash_block_size=16,
            eviction_policy=PagedEvictionPolicy(),
        )

        # Allocate
        blocks = pool.get_new_blocks(10)
        stats1 = pool.get_eviction_policy_stats()

        # Touch
        pool.touch((blocks,))
        stats2 = pool.get_eviction_policy_stats()

        # Free
        pool.free_blocks(blocks)
        stats3 = pool.get_eviction_policy_stats()

        # Verify progression - stats should be collected at each step
        assert stats1["selection_count"] >= 1
        assert stats2["update_count"] > stats1["update_count"]
        # After freeing, blocks may be cached rather than returned to free pool
        # Just verify we can still get stats
        assert "num_free_blocks" in stats3
        assert stats3["metadata_size"] >= 0


class TestLRUEvictionStats:
    """Test that LRU policy works with stats (even if basic)."""

    def test_lru_get_stats(self):
        """LRU policy should not have get_stats() method."""
        policy = LRUEvictionPolicy()

        # LRU doesn't implement get_stats()
        assert not hasattr(policy, "get_stats")

    def test_lru_in_block_pool_stats(self):
        """BlockPool stats should work with LRU policy."""
        pool = BlockPool(
            num_gpu_blocks=100,
            enable_caching=True,
            hash_block_size=16,
            eviction_policy=LRUEvictionPolicy(),
        )

        stats = pool.get_eviction_policy_stats()

        # Should still have BlockPool-level metrics
        assert stats["policy_type"] == "LRUEvictionPolicy"
        assert "num_free_blocks" in stats
        assert "num_gpu_blocks" in stats

        # Should NOT have PagedEviction-specific metrics
        assert "metadata_size" not in stats


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_stats_with_single_block(self):
        """Stats should work with single metadata entry."""
        policy = PagedEvictionPolicy()

        policy.metadata[1] = BlockMetadata(
            last_access_time=100.0,
            access_count=5,
        )
        policy.current_time = 200.0

        stats = policy.get_stats()

        assert stats["metadata_size"] == 1
        assert stats["avg_access_count"] == 5.0
        assert stats["median_access_count"] == 5

    def test_stats_with_no_cached_blocks(self):
        """Stats should handle case with no cached blocks."""
        policy = PagedEvictionPolicy()

        # All blocks non-cached
        policy.metadata[1] = BlockMetadata(is_cached=False)
        policy.metadata[2] = BlockMetadata(is_cached=False)

        stats = policy.get_stats()

        assert stats["cached_blocks_tracked"] == 0
        assert stats["avg_cache_hit_count"] == 0.0
        assert stats["max_cache_hit_count"] == 0

    def test_timing_precision(self):
        """Timing should have reasonable precision."""
        policy = PagedEvictionPolicy()

        free_blocks = [
            Mock(
                spec=KVCacheBlock, block_id=i, is_null=False, ref_cnt=0, block_hash=None
            )
            for i in range(100)
        ]

        # Do a selection that takes non-trivial time
        policy.select_blocks_to_evict(50, free_blocks, {})

        stats = policy.get_stats()

        # Should be measured (not zero) but fast (< 1ms for 100 blocks)
        assert 0 < stats["selection_avg_time"] < 0.001
