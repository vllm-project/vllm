"""
Tests for INT4 unpacking utilities for LoRA compatibility.
"""

import pytest
import torch

from vllm.lora.int4_utils import INT4Unpacker, get_unpacker


class TestINT4Unpacker:
    """Test INT4 unpacking functionality."""

    def test_unpack_per_channel_quantization(self):
        """Test unpacking with per-channel quantization."""
        unpacker = INT4Unpacker()

        # Create mock packed weights: [4, 2] unpacks to [4, 4]
        packed = torch.tensor(
            [
                [0x12, 0x34],
                [0x56, 0x78],
                [0x9A, 0xBC],
                [0xDE, 0xF0],
            ],
            dtype=torch.uint8,
        )

        # Per-channel scales
        scales = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)

        unpacked = unpacker.unpack_int4_weights(packed, scales, zero_points=None)

        assert unpacked.shape == (4, 4)
        assert unpacked.dtype == torch.float16

    def test_unpack_grouped_quantization(self):
        """Test unpacking with grouped quantization."""
        unpacker = INT4Unpacker()

        # Create mock packed weights: [2, 4] unpacks to [2, 8]
        packed = torch.randint(0, 255, (2, 4), dtype=torch.uint8)

        # Grouped scales: [out_features, num_groups]
        # For in_features=8 and group_size=4, num_groups=2
        scales = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=torch.float16,
        )

        unpacked = unpacker.unpack_int4_weights(
            packed, scales, zero_points=None, group_size=4
        )

        assert unpacked.shape == (2, 8)
        assert unpacked.dtype == torch.float16

    def test_unpack_with_zero_points(self):
        """Test unpacking with asymmetric quantization."""
        unpacker = INT4Unpacker()

        packed = torch.randint(0, 255, (2, 2), dtype=torch.uint8)
        scales = torch.tensor([1.0, 2.0], dtype=torch.float16)
        zero_points = torch.tensor([0.0, 1.0], dtype=torch.float16)

        unpacked = unpacker.unpack_int4_weights(
            packed, scales, zero_points=zero_points
        )

        assert unpacked.shape == (2, 4)
        assert unpacked.dtype == torch.float16

    def test_unpack_module_with_cache(self):
        """Test module unpacking with caching."""
        unpacker = INT4Unpacker()

        class MockQuantizedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight_packed", torch.randint(0, 255, (4, 2), dtype=torch.uint8)
                )
                self.register_buffer("weight_scale", torch.ones(4, dtype=torch.float16))

        module = MockQuantizedModule()

        # First unpack - should miss cache
        unpacked1 = unpacker.unpack_module(module, "test_module")
        assert unpacked1 is not None
        assert unpacked1.shape == (4, 4)

        stats1 = unpacker.get_cache_stats()
        assert stats1["misses"] == 1
        assert stats1["hits"] == 0

        # Second unpack - should hit cache
        unpacked2 = unpacker.unpack_module(module, "test_module")
        assert unpacked2 is not None
        assert torch.equal(unpacked1, unpacked2)

        stats2 = unpacker.get_cache_stats()
        assert stats2["hits"] == 1
        assert stats2["misses"] == 1

    def test_is_int4_quantized(self):
        """Test detection of INT4 quantized modules."""
        unpacker = INT4Unpacker()

        class MockQuantizedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight_packed", torch.randint(0, 255, (4, 2), dtype=torch.uint8)
                )
                self.register_buffer("weight_scale", torch.ones(4, dtype=torch.float16))

        class MockRegularModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 4))

        quant_module = MockQuantizedModule()
        regular_module = MockRegularModule()

        assert unpacker.is_int4_quantized(quant_module)
        assert not unpacker.is_int4_quantized(regular_module)

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        unpacker = INT4Unpacker()

        class MockQuantizedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight_packed", torch.randint(0, 255, (4, 2), dtype=torch.uint8)
                )
                self.register_buffer("weight_scale", torch.ones(4, dtype=torch.float16))

        module = MockQuantizedModule()

        # Populate cache
        unpacker.unpack_module(module, "test_module")
        stats = unpacker.get_cache_stats()
        assert stats["size"] == 1

        # Clear cache
        unpacker.clear_cache()
        stats_after = unpacker.get_cache_stats()
        assert stats_after["size"] == 0
        assert stats_after["hits"] == 0
        assert stats_after["misses"] == 0

    def test_global_unpacker(self):
        """Test global unpacker instance."""
        unpacker1 = get_unpacker()
        unpacker2 = get_unpacker()

        # Should return the same instance
        assert unpacker1 is unpacker2

    def test_invalid_dtype(self):
        """Test that non-uint8 packed weights raise error."""
        unpacker = INT4Unpacker()

        packed = torch.randint(0, 127, (2, 2), dtype=torch.int8)
        scales = torch.ones(2, dtype=torch.float16)

        with pytest.raises(ValueError, match="must be uint8"):
            unpacker.unpack_int4_weights(packed, scales)

    def test_different_output_dtypes(self):
        """Test unpacking to different output dtypes."""
        unpacker = INT4Unpacker()

        packed = torch.randint(0, 255, (2, 2), dtype=torch.uint8)
        scales = torch.ones(2, dtype=torch.float16)

        # Test bfloat16
        unpacked_bf16 = unpacker.unpack_int4_weights(
            packed, scales, output_dtype=torch.bfloat16
        )
        assert unpacked_bf16.dtype == torch.bfloat16

        # Test float32
        unpacked_fp32 = unpacker.unpack_int4_weights(
            packed, scales, output_dtype=torch.float32
        )
        assert unpacked_fp32.dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
