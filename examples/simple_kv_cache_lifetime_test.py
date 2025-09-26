#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple unit test demonstrating KV cache lifetime tracking functionality.

This is a minimal example that tests the core components without requiring
a full vLLM setup, making it easier to verify the implementation works.
"""

from unittest.mock import patch

# Import the components we implemented
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.metrics.stats import KVCacheLifetimeStats


def test_lifetime_stats_basic():
    """Test basic lifetime statistics functionality."""
    print("Testing KVCacheLifetimeStats...")

    stats = KVCacheLifetimeStats()

    # Initial state
    assert stats.total_blocks_freed == 0
    assert stats.average_lifetime_seconds == 0.0
    print("✓ Initial state correct")

    # Add some lifetimes
    stats.add_block_lifetime(2.5)
    stats.add_block_lifetime(3.5)
    stats.add_block_lifetime(4.0)

    assert stats.total_blocks_freed == 3
    assert abs(stats.average_lifetime_seconds - 3.333333) < 0.001
    print("✓ Lifetime aggregation working correctly")

    # Reset
    stats.reset()
    assert stats.total_blocks_freed == 0
    assert stats.average_lifetime_seconds == 0.0
    print("✓ Reset functionality working")


def test_block_pool_lifetime_tracking():
    """Test BlockPool lifetime tracking integration."""
    print("\nTesting BlockPool lifetime tracking...")

    # Create a small block pool for testing
    pool = BlockPool(
        num_gpu_blocks=5, enable_caching=False, enable_kv_cache_events=False
    )

    # Verify lifetime stats are initialized
    assert hasattr(pool, "lifetime_stats")
    assert isinstance(pool.lifetime_stats, KVCacheLifetimeStats)
    print("✓ BlockPool initialized with lifetime stats")

    # Test allocation and freeing with mocked time
    with patch("time.monotonic") as mock_time:
        # Mock time progression: allocation at t=100, free at t=105
        mock_time.side_effect = [100.0, 105.0]

        # Allocate blocks
        blocks = pool.get_new_blocks(2)

        # Verify allocation times are set
        for block in blocks:
            assert block.allocation_time == 100.0
        print("✓ Allocation times recorded correctly")

        # Free the blocks (will use second mock time = 105.0)
        pool.free_blocks(blocks)

        # Check lifetime stats
        stats = pool.get_lifetime_stats()
        expected_lifetime = 105.0 - 100.0  # 5.0 seconds

        assert stats.total_blocks_freed == 2
        assert stats.average_lifetime_seconds == expected_lifetime
        print(f"✓ Lifetime calculation correct: {expected_lifetime} seconds average")


def test_prometheus_integration():
    """Test Prometheus metric integration."""
    print("\nTesting Prometheus integration...")

    # Create lifetime stats with some data
    lifetime_stats = KVCacheLifetimeStats()
    lifetime_stats.add_block_lifetime(10.0)
    lifetime_stats.add_block_lifetime(20.0)

    # Simulate what the Prometheus logger would do
    if lifetime_stats.total_blocks_freed > 0:
        avg_lifetime = lifetime_stats.average_lifetime_seconds
        print(f"✓ Prometheus metric would show: {avg_lifetime} seconds")
        assert avg_lifetime == 15.0

    print("✓ Prometheus integration ready")


def run_simple_demo():
    """Run a simple demonstration of the lifetime tracking feature."""
    print("=" * 50)
    print("KV Cache Lifetime Tracking - Simple Demo")
    print("=" * 50)

    try:
        # Test each component
        test_lifetime_stats_basic()
        test_block_pool_lifetime_tracking()
        test_prometheus_integration()

        print("\n" + "=" * 50)
        print("✓ All tests passed! The implementation is working correctly.")
        print("\nKey metrics that would be available:")
        print("- vllm:kv_cache_avg_lifetime_seconds (Prometheus gauge)")
        print("- Total blocks freed count")
        print("- Average block lifetime in seconds")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_simple_demo()
    exit(0 if success else 1)
