# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for token-level KV cache metrics in KVCacheManager.

These tests verify the new token-level properties (total_tokens, free_tokens)
added to KVCacheManager to complement the existing percentage-based usage metric.
"""

import pytest
import torch

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import KVCacheConfig, StandardAttentionSpec

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def basic_kv_config():
    """Create a basic KV cache configuration for testing."""
    return KVCacheConfig(
        num_blocks=100,
        kv_cache_groups=[
            StandardAttentionSpec(
                block_size=16,
                num_kv_heads=4,
                head_size=128,
                dtype=torch.float16,
            )
        ],
    )


@pytest.fixture
def multi_group_kv_config():
    """Create a multi-group KV cache configuration (hybrid model)."""
    return KVCacheConfig(
        num_blocks=120,
        kv_cache_groups=[
            StandardAttentionSpec(
                block_size=16,
                num_kv_heads=4,
                head_size=128,
                dtype=torch.float16,
            ),
            StandardAttentionSpec(
                block_size=8,  # Different block size
                num_kv_heads=2,
                head_size=64,
                dtype=torch.float16,
            ),
        ],
    )


def test_token_metrics_basic(basic_kv_config):
    """Test basic token metrics calculation with single group."""
    manager = KVCacheManager(
        kv_cache_config=basic_kv_config,
        max_model_len=1024,
        hash_block_size=16,
        enable_caching=True,
    )

    # Verify block_size
    assert manager.block_size == 16

    # Verify total_tokens
    # Formula: (num_blocks - 1) / num_groups × block_size
    #        = (100 - 1) / 1 × 16 = 1584
    assert manager.total_tokens == (100 - 1) * 16

    # Initially all tokens should be free
    assert manager.free_tokens == manager.total_tokens

    # Verify consistency
    assert manager.total_tokens >= manager.free_tokens


def test_token_metrics_multi_group(multi_group_kv_config):
    """Test token metrics with multi-group configuration (hybrid model)."""
    manager = KVCacheManager(
        kv_cache_config=multi_group_kv_config,
        max_model_len=1024,
        hash_block_size=8,
        enable_caching=True,
    )

    # Should use min block size across groups
    assert manager.block_size == min(16, 8)

    # Formula: (num_blocks - 1) / num_groups × block_size
    #        = (120 - 1) / 2 × 8 = 59 × 8 = 472
    expected_total = ((120 - 1) // 2) * 8
    assert manager.total_tokens == expected_total

    # free_tokens should also account for num_groups
    assert manager.free_tokens <= manager.total_tokens


def test_token_metrics_caching_disabled():
    """Test that metrics return 0 when caching is disabled."""
    kv_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_groups=[
            StandardAttentionSpec(
                block_size=16,
                num_kv_heads=4,
                head_size=128,
                dtype=torch.float16,
            )
        ],
    )

    manager = KVCacheManager(
        kv_cache_config=kv_config,
        max_model_len=1024,
        hash_block_size=16,
        enable_caching=False,
    )

    # All metrics should return 0 when caching is disabled
    assert manager.block_size == 0
    assert manager.total_tokens == 0
    assert manager.free_tokens == 0


def test_token_metrics_null_block_accounting(basic_kv_config):
    """Test that null_block is correctly excluded from total_tokens."""
    manager = KVCacheManager(
        kv_cache_config=basic_kv_config,
        max_model_len=1024,
        hash_block_size=16,
        enable_caching=True,
    )

    # total_tokens should be (num_blocks - 1) to exclude null_block
    # num_blocks = 100, so available blocks = 99
    expected_total = 99 * 16  # 1584
    assert manager.total_tokens == expected_total

    # Verify this matches the block_pool accounting
    # block_pool.get_usage() also subtracts 1 for null_block
    block_pool = manager.block_pool
    available_blocks = block_pool.num_gpu_blocks - 1
    assert manager.total_tokens == available_blocks * manager.block_size


def test_token_metrics_consistency_with_usage(basic_kv_config):
    """Test that token metrics are consistent with percentage-based usage."""
    manager = KVCacheManager(
        kv_cache_config=basic_kv_config,
        max_model_len=1024,
        hash_block_size=16,
        enable_caching=True,
    )

    # When cache is empty, usage should be 0
    assert manager.usage == 0.0
    assert manager.free_tokens == manager.total_tokens

    # The relationship: usage = 1.0 - (free_tokens / total_tokens)
    # should hold (within floating point precision)
    if manager.total_tokens > 0:
        calculated_usage = 1.0 - (manager.free_tokens / manager.total_tokens)
        assert abs(manager.usage - calculated_usage) < 0.01


def test_token_metrics_type_safety():
    """Test that metrics always return int, never None."""
    kv_config = KVCacheConfig(
        num_blocks=50,
        kv_cache_groups=[
            StandardAttentionSpec(
                block_size=8,
                num_kv_heads=2,
                head_size=64,
                dtype=torch.float16,
            )
        ],
    )

    manager = KVCacheManager(
        kv_cache_config=kv_config,
        max_model_len=1024,
        hash_block_size=8,
        enable_caching=True,
    )

    # All metrics should be int, not None or Optional[int]
    assert isinstance(manager.block_size, int)
    assert isinstance(manager.total_tokens, int)
    assert isinstance(manager.free_tokens, int)

    # Even when caching is disabled
    manager_disabled = KVCacheManager(
        kv_cache_config=kv_config,
        max_model_len=1024,
        hash_block_size=8,
        enable_caching=False,
    )

    assert isinstance(manager_disabled.block_size, int)
    assert isinstance(manager_disabled.total_tokens, int)
    assert isinstance(manager_disabled.free_tokens, int)
    assert manager_disabled.block_size == 0
    assert manager_disabled.total_tokens == 0
    assert manager_disabled.free_tokens == 0
