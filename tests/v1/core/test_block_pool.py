# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for BlockPool lazy null block allocation."""

import pytest

from vllm.v1.core.block_pool import BlockPool


class TestBlockPoolLazyNullBlock:
    """Test lazy null block allocation in BlockPool."""

    def test_null_block_not_allocated_initially(self):
        """Test that null block is not allocated during BlockPool initialization."""
        pool = BlockPool(num_gpu_blocks=4, enable_caching=True)

        # Initially, null block should not be allocated
        assert pool._null_block is None
        assert pool.get_num_free_blocks() == 4

        # Verify null_block is a property, not an instance attribute
        assert "null_block" not in pool.__dict__
        assert isinstance(type(pool).null_block, property)

    def test_null_block_lazy_allocation(self):
        """Test that null block is allocated only when first accessed."""
        pool = BlockPool(num_gpu_blocks=4, enable_caching=True)

        # Before accessing null_block
        assert pool._null_block is None
        assert pool.get_num_free_blocks() == 4

        # Access null_block - should trigger lazy allocation
        null_block = pool.null_block

        # After accessing null_block
        assert pool._null_block is not None
        assert pool.get_num_free_blocks() == 3  # One block consumed
        assert null_block.is_null is True
        assert null_block.block_id == 0

    def test_null_block_reuse(self):
        """Test that multiple accesses return the same null block instance."""
        pool = BlockPool(num_gpu_blocks=4, enable_caching=True)

        # First access
        null_block1 = pool.null_block
        free_blocks_after_first = pool.get_num_free_blocks()

        # Second access
        null_block2 = pool.null_block
        free_blocks_after_second = pool.get_num_free_blocks()

        # Should return same instance without additional allocation
        assert null_block1 is null_block2
        assert free_blocks_after_first == free_blocks_after_second == 3

    def test_null_block_allocation_when_no_blocks_available(self):
        """Test error handling when trying to allocate null block with
        no free blocks."""
        pool = BlockPool(num_gpu_blocks=2, enable_caching=True)

        # Consume all available blocks
        pool.get_new_blocks(2)
        assert pool.get_num_free_blocks() == 0

        # Trying to access null_block should raise RuntimeError
        with pytest.raises(RuntimeError, match="Cannot allocate null block"):
            _ = pool.null_block

    def test_get_usage_with_lazy_null_block(self):
        """Test that get_usage() correctly accounts for lazy null block allocation."""
        pool = BlockPool(num_gpu_blocks=4, enable_caching=True)

        # Before null block allocation
        usage_before = pool.get_usage()
        assert usage_before == 0.0  # No blocks used, no null block overhead

        # After null block allocation (but no actual workload blocks allocated)
        _ = pool.null_block
        usage_after = pool.get_usage()
        # Null block is overhead, not "usage" - so usage should still be 0
        assert usage_after == 0.0

        # Now allocate actual workload blocks
        _ = pool.get_new_blocks(2)
        usage_with_workload = pool.get_usage()
        # Formula: 1.0 - (1 / 3) = 2/3 (1 free block out of 3 available)
        assert usage_with_workload == pytest.approx(2.0 / 3.0)

    def test_reset_prefix_cache_with_lazy_null_block(self):
        """Test that reset_prefix_cache() works correctly with lazy null block."""
        pool = BlockPool(num_gpu_blocks=4, enable_caching=True)

        # Before null block allocation - should succeed
        assert pool.reset_prefix_cache() is True

        # After null block allocation - should still succeed
        _ = pool.null_block
        assert pool.reset_prefix_cache() is True


class TestManagerLazyNullBlock:
    """Test lazy null block allocation in KV cache managers."""
