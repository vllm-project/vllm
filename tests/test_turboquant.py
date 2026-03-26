"""
Tests for TurboQuant KV cache optimization.

This module contains comprehensive tests for the TurboQuant quantization
implementation, covering MSE-based and product-based quantization schemes.
"""
from __future__ import annotations

import math

import pytest
import torch

from vllm.turboquant import (
    TurboQuantKVCache,
    _TurboQuantMSECodec,
    _TurboQuantProdCodec,
    _build_codec,
    turboquant_enabled,
)


def _sample_unit_vectors(count: int, dim: int) -> torch.Tensor:
    """Generate random unit vectors."""
    vectors = torch.randn(count, dim)
    return vectors / torch.linalg.norm(vectors, dim=-1, keepdims=True)


class TestTurboQuantMSE:
    """Tests for MSE-based quantization."""

    def test_mse_codec_initialization(self):
        """Test codec initialization."""
        codec = _TurboQuantMSECodec(64, 2, seed=0)
        assert codec.dim == 64
        assert codec.bits == 2
        assert codec.rotation.shape == (64, 64)
        assert codec.rotation_t.shape == (64, 64)
        assert codec.codebook.shape[0] == 4  # 2^2

    def test_mse_quantize_dequantize_roundtrip(self):
        """Test quantization and dequantization roundtrip."""
        codec = _TurboQuantMSECodec(32, 2, seed=0)
        vectors = torch.randn(16, 32)

        state = codec.quantize(vectors)
        reconstructed = codec.dequantize(state)

        assert reconstructed.shape == vectors.shape
        # Check that norms are preserved
        assert torch.allclose(
            torch.linalg.norm(vectors, dim=-1),
            torch.linalg.norm(reconstructed, dim=-1),
            rtol=0.1,
        )

    def test_mse_higher_bits_lower_error(self):
        """Test that higher bit-widths result in lower quantization error."""
        vectors = _sample_unit_vectors(64, 32)

        mse_2bit = []
        mse_4bit = []

        for _ in range(3):
            codec_2 = _TurboQuantMSECodec(32, 2, seed=0)
            codec_4 = _TurboQuantMSECodec(32, 4, seed=0)

            state_2 = codec_2.quantize(vectors)
            state_4 = codec_4.quantize(vectors)

            error_2 = torch.mean((vectors - codec_2.dequantize(state_2)) ** 2)
            error_4 = torch.mean((vectors - codec_4.dequantize(state_4)) ** 2)

            mse_2bit.append(error_2.item())
            mse_4bit.append(error_4.item())

        assert sum(mse_4bit) / len(mse_4bit) < sum(mse_2bit) / len(mse_2bit)


class TestTurboQuantProd:
    """Tests for product quantization."""

    def test_prod_codec_initialization(self):
        """Test product codec initialization."""
        codec = _TurboQuantProdCodec(64, 3, seed=0)
        assert codec.dim == 64
        assert codec.bits == 3
        assert codec.mse_codec.bits == 2  # bits - 1
        assert codec.query_transform_t.shape == (64, 128)  # 64 for rotation + 64 for projection

    def test_prod_quantize_dequantize(self):
        """Test product quantization roundtrip."""
        codec = _TurboQuantProdCodec(32, 2, seed=0)
        vectors = torch.randn(16, 32)

        state = codec.quantize(vectors)
        reconstructed = codec.dequantize(state)

        assert reconstructed.shape == vectors.shape
        # Product quantization preserves norms approximately
        assert torch.allclose(
            torch.linalg.norm(vectors, dim=-1),
            torch.linalg.norm(reconstructed, dim=-1),
            rtol=0.2,
        )

    def test_prod_is_unbiased_across_seeds(self):
        """Test that product quantization is approximately unbiased."""
        keys = _sample_unit_vectors(32, 64)
        queries = torch.randn(32, 64)
        true_inner_products = torch.sum(keys * queries, dim=-1)

        estimates = []
        for seed in range(8):
            codec = _TurboQuantProdCodec(64, 2, seed=seed)
            state = codec.quantize(keys)
            reconstructed = codec.dequantize(state)
            estimates.append(torch.sum(reconstructed * queries, dim=-1))

        mean_estimate = torch.mean(torch.stack(estimates), dim=0)
        bias = torch.mean(mean_estimate - true_inner_products).item()
        assert abs(bias) < 0.1


class TestFractionalBits:
    """Tests for fractional bit-width support."""

    def test_turboquant_enabled_detection(self):
        """Test detection of TurboQuant-enabling fractional bits."""
        assert turboquant_enabled(3.5)
        assert turboquant_enabled(2.5)
        assert not turboquant_enabled(3.0)
        assert not turboquant_enabled(4.0)
        assert not turboquant_enabled(None)

    def test_fractional_improves_reconstruction(self):
        """Test that fractional bits improve reconstruction quality."""
        vectors = torch.randn(1, 2, 32, 64)

        codec_3bit = _build_codec(vectors, 3.0, mode="mse", seed=0)
        codec_35bit = _build_codec(vectors, 3.5, mode="mse", seed=0)

        state_3bit = codec_3bit.quantize(vectors)
        state_35bit = codec_35bit.quantize(vectors)

        error_3bit = torch.mean((vectors - codec_3bit.dequantize(state_3bit)) ** 2).item()
        error_35bit = torch.mean((vectors - codec_35bit.dequantize(state_35bit)) ** 2).item()

        assert error_35bit < error_3bit

    def test_bit_validation(self):
        """Test bit-width validation."""
        with pytest.raises(ValueError, match="requires a fractional bit-width"):
            _build_codec(torch.randn(1, 1, 1, 32), 3.7, mode="mse", seed=0)

        with pytest.raises(ValueError, match="kv_bits >= 1"):
            from vllm.turboquant import _validate_bits
            _validate_bits(0.5)


class TestTurboQuantKVCache:
    """Tests for TurboQuantKVCache."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = TurboQuantKVCache(bits=3.5)
        assert cache.bits == 3.5
        assert cache.offset == 0
        assert cache.keys is None
        assert cache.values is None

    def test_cache_update_and_fetch(self):
        """Test cache update and fetch."""
        cache = TurboQuantKVCache(bits=3.0)
        keys = torch.randn(1, 2, 8, 32)
        values = torch.randn(1, 2, 8, 32)

        k_state, v_state = cache.update_and_fetch(keys, values)

        assert k_state is not None
        assert v_state is not None
        assert cache.key_codec is not None
        assert cache.value_codec is not None

    def test_cache_dequantize(self):
        """Test cache dequantization."""
        cache = TurboQuantKVCache(bits=3.5)
        keys = torch.randn(1, 2, 8, 32)
        values = torch.randn(1, 2, 8, 32)

        cache.update_and_fetch(keys, values)
        k_deq, v_deq = cache.dequantize()

        assert k_deq.shape == keys.shape
        assert v_deq.shape == values.shape
        assert k_deq.dtype == torch.float32
        assert v_deq.dtype == torch.float32

    def test_cache_preserves_shapes(self):
        """Test that cache preserves tensor shapes."""
        cache = TurboQuantKVCache(bits=3.5)
        keys = torch.randn(2, 4, 16, 64)
        values = torch.randn(2, 4, 16, 64)

        cache.update_and_fetch(keys, values)
        k_deq, v_deq = cache.dequantize()

        assert k_deq.shape == keys.shape
        assert v_deq.shape == values.shape

    def test_cache_compression_ratio(self):
        """Test memory compression ratio."""
        cache = TurboQuantKVCache(bits=3.5)
        keys = torch.randn(1, 2, 256, 64)
        values = torch.randn(1, 2, 256, 64)

        cache.update_and_fetch(keys, values)

        # Rough estimate: 3.5 bits per value vs 32 bits for float32
        # Should achieve ~10x compression
        original_size = keys.nbytes + values.nbytes
        compressed_size = cache.nbytes

        # Allow some overhead but should still be significantly smaller
        compression_ratio = original_size / (compressed_size + 1)  # +1 to avoid division by zero
        assert compression_ratio > 5  # At least 5x compression


class TestModeSelection:
    """Tests for codec mode selection."""

    def test_mse_mode_for_values(self):
        """Test that MSE mode is selected for values."""
        vectors = torch.randn(1, 2, 16, 64)
        codec = _build_codec(vectors, 3.5, mode="mse", seed=0)
        
        # For integer bits, should be MSE codec
        codec_3bit = _build_codec(vectors, 3.0, mode="mse", seed=0)
        assert isinstance(codec_3bit, _TurboQuantMSECodec)

    def test_prod_mode_for_keys(self):
        """Test that prod mode is selected for keys."""
        vectors = torch.randn(1, 2, 16, 64)
        codec = _build_codec(vectors, 3.5, mode="prod", seed=0)
        
        # For integer bits, should be prod codec
        codec_3bit = _build_codec(vectors, 3.0, mode="prod", seed=0)
        assert isinstance(codec_3bit, _TurboQuantProdCodec)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_zero_dimension_codec(self):
        """Test codec with zero dimension."""
        codec = _TurboQuantMSECodec(0, 2, seed=0)
        assert codec.dim == 0
        assert codec.rotation.shape == (0, 0)

    def test_single_dimension_vectors(self):
        """Test quantization of single-dimension vectors."""
        codec = _TurboQuantMSECodec(1, 2, seed=0)
        vectors = torch.randn(16, 1)

        state = codec.quantize(vectors)
        reconstructed = codec.dequantize(state)

        assert reconstructed.shape == vectors.shape

    def test_single_bit_quantization(self):
        """Test quantization with single bit."""
        codec = _TurboQuantMSECodec(32, 1, seed=0)
        vectors = torch.randn(16, 32)

        state = codec.quantize(vectors)
        reconstructed = codec.dequantize(state)

        assert reconstructed.shape == vectors.shape
        # Single bit should have very high error
        error = torch.mean((vectors - reconstructed) ** 2)
        assert error > 0.1

    def test_batch_quantization(self):
        """Test quantization with different batch sizes."""
        codec = _TurboQuantMSECodec(32, 2, seed=0)

        for batch_size in [1, 8, 32, 128]:
            vectors = torch.randn(batch_size, 32)
            state = codec.quantize(vectors)
            reconstructed = codec.dequantize(state)
            assert reconstructed.shape == vectors.shape

    def test_device_compatibility(self):
        """Test codec works with different device placements (CPU)."""
        codec = _TurboQuantMSECodec(32, 2, seed=0)
        vectors = torch.randn(16, 32)

        state = codec.quantize(vectors)
        reconstructed = codec.dequantize(state)

        assert reconstructed.device == vectors.device


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_zero_vector_handling(self):
        """Test handling of zero vectors."""
        codec = _TurboQuantMSECodec(32, 2, seed=0)
        vectors = torch.zeros(16, 32)

        state = codec.quantize(vectors)
        reconstructed = codec.dequantize(state)

        assert torch.allclose(reconstructed, torch.zeros_like(reconstructed), atol=1e-6)

    def test_large_norm_vectors(self):
        """Test handling of very large norm vectors."""
        codec = _TurboQuantMSECodec(32, 2, seed=0)
        vectors = torch.randn(16, 32) * 1e6

        state = codec.quantize(vectors)
        reconstructed = codec.dequantize(state)

        # Norms should be preserved
        orig_norms = torch.linalg.norm(vectors, dim=-1)
        recon_norms = torch.linalg.norm(reconstructed, dim=-1)
        assert torch.allclose(orig_norms, recon_norms, rtol=0.1)

    def test_small_norm_vectors(self):
        """Test handling of very small norm vectors."""
        codec = _TurboQuantMSECodec(32, 2, seed=0)
        vectors = torch.randn(16, 32) * 1e-6

        state = codec.quantize(vectors)
        reconstructed = codec.dequantize(state)

        # Should handle gracefully without numerical issues
        assert not torch.isnan(reconstructed).any()
        assert not torch.isinf(reconstructed).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
