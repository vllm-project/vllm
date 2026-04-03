"""
Unit tests for RoaringAttentionMask implementation.
"""

import pytest
import numpy as np
import torch

try:
    import pyroaring
    HAS_ROARING = True
except ImportError:
    HAS_ROARING = False

if HAS_ROARING:
    from vllm.attention.roaring_attention_mask import (
        RoaringAttentionMask,
        create_sliding_window_mask
    )


@pytest.mark.skipif(not HAS_ROARING, reason="pyroaring not installed")
class TestRoaringAttentionMask:
    """Test suite for RoaringAttentionMask."""
    
    def test_initialization(self):
        """Test basic initialization."""
        mask = RoaringAttentionMask(num_blocks=100)
        assert mask.num_blocks == 100
        assert len(mask.masks) == 100
        assert mask.get_density() == 0.0
    
    def test_add_attention(self):
        """Test adding attention connections."""
        mask = RoaringAttentionMask(num_blocks=10)
        
        # Add attention from block 0 to blocks 1, 2, 3
        mask.add_attention(0, [1, 2, 3])
        active = mask.get_active_blocks(0)
        assert np.array_equal(active, [1, 2, 3])
        
        # Add more connections
        mask.add_attention(0, [4, 5])
        active = mask.get_active_blocks(0)
        assert np.array_equal(active, [1, 2, 3, 4, 5])
    
    def test_remove_attention(self):
        """Test removing attention connections."""
        mask = RoaringAttentionMask(num_blocks=10)
        
        mask.add_attention(0, [1, 2, 3, 4, 5])
        mask.remove_attention(0, [2, 4])
        active = mask.get_active_blocks(0)
        assert np.array_equal(active, [1, 3, 5])
    
    def test_invalid_indices(self):
        """Test handling of invalid indices."""
        mask = RoaringAttentionMask(num_blocks=10)
        
        # Should raise for out-of-bounds query block
        with pytest.raises(ValueError):
            mask.add_attention(10, [1, 2])
        
        # Should filter out invalid key blocks silently
        mask.add_attention(0, [-1, 5, 15])
        active = mask.get_active_blocks(0)
        assert np.array_equal(active, [5])
    
    def test_intersection(self):
        """Test intersection operation."""
        mask = RoaringAttentionMask(num_blocks=10)
        
        mask.add_attention(0, [1, 2, 3, 4])
        mask.add_attention(1, [2, 3, 5, 6])
        
        intersection = mask.intersection(0, 1)
        assert np.array_equal(intersection, [2, 3])
    
    def test_union(self):
        """Test union operation."""
        mask = RoaringAttentionMask(num_blocks=10)
        
        mask.add_attention(0, [1, 2, 3])
        mask.add_attention(1, [3, 4, 5])
        
        union = mask.union(0, 1)
        assert np.array_equal(union, [1, 2, 3, 4, 5])
    
    def test_batch_intersection(self):
        """Test batch intersection operation."""
        mask = RoaringAttentionMask(num_blocks=10)
        
        mask.add_attention(0, [1, 2, 3, 4])
        mask.add_attention(1, [2, 3, 5])
        mask.add_attention(2, [3, 4, 5])
        
        # Intersection of all three
        intersection = mask.batch_intersection([0, 1, 2])
        assert np.array_equal(intersection, [3])
        
        # Empty list should return empty array
        intersection = mask.batch_intersection([])
        assert len(intersection) == 0
    
    def test_has_attention(self):
        """Test attention existence check."""
        mask = RoaringAttentionMask(num_blocks=10)
        
        mask.add_attention(0, [1, 3, 5])
        
        assert mask.has_attention(0, 1) == True
        assert mask.has_attention(0, 2) == False
        assert mask.has_attention(0, 3) == True
        assert mask.has_attention(10, 1) == False  # Invalid query
    
    def test_density_calculation(self):
        """Test density calculation."""
        mask = RoaringAttentionMask(num_blocks=10)
        
        # Initially empty
        assert mask.get_density() == 0.0
        
        # Add 10 connections out of 100 possible
        mask.add_attention(0, [1, 2, 3, 4, 5])
        mask.add_attention(1, [2, 3, 4, 5, 6])
        
        density = mask.get_density()
        assert density == 10 / 100  # 10 connections / 100 possible
    
    def test_to_dense_mask(self):
        """Test conversion to dense mask."""
        mask = RoaringAttentionMask(num_blocks=5)
        
        mask.add_attention(0, [1, 2])
        mask.add_attention(1, [0, 2, 3])
        mask.add_attention(2, [4])
        
        dense = mask.to_dense_mask()
        
        assert dense.shape == (5, 5)
        assert dense[0, 1] == True
        assert dense[0, 2] == True
        assert dense[1, 0] == True
        assert dense[1, 2] == True
        assert dense[1, 3] == True
        assert dense[2, 4] == True
        
        # Check all other positions are False
        assert dense.sum() == 6
    
    def test_sliding_window(self):
        """Test sliding window pattern."""
        mask = RoaringAttentionMask(num_blocks=10)
        mask.apply_sliding_window(window_size=3, global_tokens=2)
        
        # Check first block (global token)
        block0 = mask.get_active_blocks(0)
        assert len(block0) == 10  # Attends to all
        
        # Check non-global block
        block5 = mask.get_active_blocks(5)
        # Should attend to: global tokens (0,1) + sliding window (4,5,6)
        assert 0 in block5  # Global
        assert 1 in block5  # Global
        assert 4 in block5  # Window
        assert 5 in block5  # Self
        assert 6 in block5  # Window
    
    def test_serialization(self):
        """Test serialization and deserialization."""
        mask = RoaringAttentionMask(num_blocks=10)
        
        mask.add_attention(0, [1, 2, 3])
        mask.add_attention(5, [6, 7, 8, 9])
        
        # Serialize
        serialized = mask.serialize()
        
        # Deserialize
        mask2 = RoaringAttentionMask.deserialize(serialized)
        
        assert mask2.num_blocks == 10
        assert np.array_equal(mask2.get_active_blocks(0), [1, 2, 3])
        assert np.array_equal(mask2.get_active_blocks(5), [6, 7, 8, 9])
    
    def test_tensor_conversion(self):
        """Test conversion to PyTorch tensor."""
        mask = RoaringAttentionMask(num_blocks=10, device="cpu")
        
        mask.add_attention(0, [1, 3, 5, 7])
        tensor = mask.get_active_blocks_tensor(0)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.int32
        assert torch.equal(tensor, torch.tensor([1, 3, 5, 7], dtype=torch.int32))
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        mask = RoaringAttentionMask(num_blocks=100)
        
        # Empty mask should have minimal memory
        empty_memory = mask.get_memory_usage_bytes()
        assert empty_memory > 0
        assert empty_memory < 10000  # Should be < 10KB for 100 empty bitmaps
        
        # Add connections and check memory increases
        for i in range(10):
            mask.add_attention(i, list(range(50)))
        
        full_memory = mask.get_memory_usage_bytes()
        assert full_memory > empty_memory
    
    def test_create_sliding_window_helper(self):
        """Test the helper function for creating sliding window masks."""
        mask = create_sliding_window_mask(
            num_blocks=20,
            window_size=5,
            global_tokens=3,
            device="cpu"
        )
        
        assert mask.num_blocks == 20
        
        # Check global token behavior
        block1 = mask.get_active_blocks(1)
        assert len(block1) == 20  # Global token attends to all
        
        # Check regular block with sliding window
        block10 = mask.get_active_blocks(10)
        assert 0 in block10  # Global token
        assert 1 in block10  # Global token  
        assert 2 in block10  # Global token
        assert 8 in block10  # Window
        assert 10 in block10  # Self
        assert 12 in block10  # Window


@pytest.mark.skipif(not HAS_ROARING, reason="pyroaring not installed")
class TestRoaringAttentionPerformance:
    """Performance-related tests."""
    
    def test_large_scale_operations(self):
        """Test with larger number of blocks."""
        mask = RoaringAttentionMask(num_blocks=1000)
        
        # Add sliding window pattern
        for i in range(1000):
            start = max(0, i - 64)
            end = min(1000, i + 64)
            mask.add_attention(i, list(range(start, end)))
        
        # Test operations still work
        active = mask.get_active_blocks(500)
        assert len(active) == 128  # Window size
        assert min(active) == 436
        assert max(active) == 563
        
        # Test intersection
        intersection = mask.intersection(500, 510)
        assert len(intersection) > 0  # Should have overlap
    
    def test_sparse_vs_dense_patterns(self):
        """Compare behavior with sparse vs dense patterns."""
        # Very sparse pattern
        sparse_mask = RoaringAttentionMask(num_blocks=1000)
        for i in range(0, 1000, 100):
            sparse_mask.add_attention(i, [i+1, i+2])
        
        sparse_memory = sparse_mask.get_memory_usage_bytes()
        sparse_density = sparse_mask.get_density()
        
        # Dense pattern  
        dense_mask = RoaringAttentionMask(num_blocks=100)
        for i in range(100):
            dense_mask.add_attention(i, list(range(100)))
        
        dense_memory = dense_mask.get_memory_usage_bytes()
        dense_density = dense_mask.get_density()
        
        assert sparse_density < 0.01  # Very sparse
        assert dense_density == 1.0  # Fully dense
        
        # Sparse should use less memory per connection
        sparse_per_connection = sparse_memory / (20 if sparse_density > 0 else 1)
        dense_per_connection = dense_memory / 10000
        assert sparse_per_connection < dense_per_connection * 10  # Some overhead is OK


if __name__ == "__main__":
    pytest.main([__file__, "-v"])