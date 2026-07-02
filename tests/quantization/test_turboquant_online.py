# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TurboQuant online weight compression.

Tests the codebook, pack/unpack, quantize/dequantize, and WHT rotation
from vllm.model_executor.layers.quantization.online.turboquant.

These are all CPU-only and can run without a GPU.

Run: python -m pytest tests/quantization/test_turboquant_online.py -v
"""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.quantization.online.turboquant import (
    TurboQuantW2OnlineLinearMethod,
    TurboQuantW3OnlineLinearMethod,
    TurboQuantW4OnlineLinearMethod,
    _fast_wht_batch,
    _get_quantizer,
    _pack_indices,
    _padded_size,
    _PolarQuant,
    _TurboQuantOnlineLinearMethodBase,
    _unpack_indices,
)
from vllm.utils.math_utils import next_power_of_2

# Codebook correctness is tested in tests/quantization/test_turboquant.py
# (covers the shared vllm.model_executor.layers.quantization.turboquant.centroids
# module this PR reuses).


# ============================================================================
# 3-bit packing — the hard part (cross-byte boundaries)
#
# The breakthrough fix: 8 values × 3 bits = 24 bits = 3 bytes.
# Values at positions 2 and 5 cross byte boundaries.
# ============================================================================


class TestPackUnpack:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_roundtrip(self, bits, dim):
        n_rows = 16
        max_val = 2**bits
        indices = torch.randint(0, max_val, (n_rows, dim), dtype=torch.int64)
        packed = _pack_indices(indices, bits)
        unpacked = _unpack_indices(packed, bits, dim)
        assert torch.equal(unpacked, indices), (
            f"bits={bits}, dim={dim}: pack/unpack roundtrip failed"
        )

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_packed_is_smaller(self, bits):
        dim = 128
        n_rows = 8
        indices = torch.randint(0, 2**bits, (n_rows, dim), dtype=torch.int64)
        packed = _pack_indices(indices, bits)
        assert (
            packed.numel() * packed.element_size()
            < indices.numel() * indices.element_size()
        )

    def test_unpack_truncates_to_dim(self):
        """Unpack should return exactly `dim` columns even with non-aligned dim."""
        dim = 100  # not aligned to 8 (for 3-bit)
        indices = torch.randint(0, 8, (4, dim), dtype=torch.int64)
        packed = _pack_indices(indices, bits=3)
        unpacked = _unpack_indices(packed, bits=3, dim=dim)
        assert unpacked.shape == (4, dim)
        assert torch.equal(unpacked, indices)

    def test_3bit_cross_byte_boundary(self):
        """Verify 3-bit packing handles cross-byte values at positions 2 and 5.

        Layout: 8 values → 3 bytes
        byte0: [v0:3][v1:3][v2_lo:2]
        byte1: [v2_hi:1][v3:3][v4:3][v5_lo:1]
        byte2: [v5_hi:2][v6:3][v7:3]

        Values at positions 2 and 5 split across two bytes.
        """
        # All max values (0b111 = 7) to stress every bit
        indices = torch.full((1, 8), 7, dtype=torch.int64)
        packed = _pack_indices(indices, bits=3)
        unpacked = _unpack_indices(packed, bits=3, dim=8)
        assert torch.equal(unpacked, indices)

        # Alternating pattern: catches swap/shift errors
        indices = torch.tensor([[0, 7, 0, 7, 0, 7, 0, 7]], dtype=torch.int64)
        packed = _pack_indices(indices, bits=3)
        unpacked = _unpack_indices(packed, bits=3, dim=8)
        assert torch.equal(unpacked, indices)

        # Specifically test cross-byte positions 2 and 5
        for pos in [2, 5]:
            for val in range(8):
                indices = torch.zeros(1, 8, dtype=torch.int64)
                indices[0, pos] = val
                packed = _pack_indices(indices, bits=3)
                unpacked = _unpack_indices(packed, bits=3, dim=8)
                assert unpacked[0, pos] == val, (
                    f"3-bit cross-byte fail: pos={pos}, val={val}, "
                    f"got={unpacked[0, pos]}"
                )

    @pytest.mark.parametrize("dim", [8, 16, 24, 48, 120, 128])
    def test_3bit_various_dims(self, dim):
        """3-bit packing at various dims including non-multiples of 8."""
        indices = torch.randint(0, 8, (4, dim), dtype=torch.int64)
        packed = _pack_indices(indices, bits=3)
        unpacked = _unpack_indices(packed, bits=3, dim=dim)
        assert torch.equal(unpacked, indices)

    def test_3bit_compression_ratio(self):
        """3-bit packing should achieve 3/8 bytes per element (62.5% savings)."""
        dim = 128
        indices = torch.randint(0, 8, (1, dim), dtype=torch.int64)
        packed = _pack_indices(indices, bits=3)
        # 128 values × 3 bits = 384 bits = 48 bytes
        assert packed.shape[1] == 48, (
            f"Expected 48 packed bytes for dim=128, got {packed.shape[1]}"
        )


# ============================================================================
# Padded size helper
# ============================================================================


class TestPaddedSize:
    def test_exact_multiple(self):
        padded, n_groups = _padded_size(256, 128)
        assert padded == 256
        assert n_groups == 2

    def test_needs_padding(self):
        padded, n_groups = _padded_size(200, 128)
        assert padded == 256
        assert n_groups == 2

    def test_small_dim(self):
        padded, n_groups = _padded_size(64, 128)
        assert padded == 128
        assert n_groups == 1


# ============================================================================
# WHT rotation
# ============================================================================


class TestWHT:
    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_wht_involution(self, n):
        """WHT applied twice (with normalization) should return the original."""
        x = torch.randn(4, n)
        y = _fast_wht_batch(x.clone())
        z = _fast_wht_batch(y.clone())
        assert torch.allclose(z, x, atol=1e-5), "WHT is not an involution"

    def test_wht_orthogonal(self):
        """WHT should preserve norms (up to float precision)."""
        x = torch.randn(8, 128)
        y = _fast_wht_batch(x.clone())
        x_norms = torch.linalg.norm(x, dim=1)
        y_norms = torch.linalg.norm(y, dim=1)
        assert torch.allclose(x_norms, y_norms, atol=1e-4)

    def test_wht_requires_power_of_2(self):
        """WHT only works with power-of-2 dimensions."""
        x = torch.randn(2, 128, device="cpu")
        y = _fast_wht_batch(x.clone())
        assert y.shape == (2, 128)


# ============================================================================
# PolarQuant quantize/dequantize
# ============================================================================


class TestPolarQuant:
    @pytest.mark.parametrize("bits", [3, 4])
    def test_roundtrip_error(self, bits):
        """Quantize→dequantize should have bounded relative error."""
        dim = 128
        pq = _PolarQuant(dim, bits, device="cpu")
        x = torch.randn(32, dim)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)

        row_err = (x - x_hat).norm(dim=1) / x.norm(dim=1).clamp(min=1e-8)
        mean_err = row_err.mean().item()

        # TQ3 ~18%, TQ4 ~10% relative error
        max_acceptable = 0.30 if bits == 3 else 0.20
        assert mean_err < max_acceptable, (
            f"bits={bits}: mean relative error {mean_err:.3f} > {max_acceptable}"
        )

    @pytest.mark.parametrize("bits", [3, 4])
    def test_indices_in_range(self, bits):
        dim = 128
        pq = _PolarQuant(dim, bits, device="cpu")
        x = torch.randn(16, dim)
        indices, _ = pq.quantize(x)
        assert indices.min() >= 0
        assert indices.max() < 2**bits

    def test_deterministic(self):
        pq = _PolarQuant(128, 3, device="cpu")
        x = torch.randn(8, 128)
        idx1, n1 = pq.quantize(x)
        idx2, n2 = pq.quantize(x)
        assert torch.equal(idx1, idx2)
        assert torch.equal(n1, n2)

    def test_zero_vector(self):
        """Zero input should not crash."""
        pq = _PolarQuant(128, 3, device="cpu")
        x = torch.zeros(2, 128)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)
        assert torch.allclose(x_hat, torch.zeros_like(x_hat), atol=1e-6)

    def test_norm_correction(self):
        """Stored norms should differ from raw L2 norms (norm correction ratio).

        PolarQuant stores original_norm / reconstruction_norm, not raw norms.
        This correction is critical for quality — without it, TQ3 error is
        ~2x worse.
        """
        dim = 128
        pq = _PolarQuant(dim, 3, device="cpu")
        x = torch.randn(16, dim)
        raw_norms = torch.linalg.norm(x, dim=1)
        _, corrected_norms = pq.quantize(x)

        # Corrected norms should be close to but not exactly equal to raw norms
        # (ratio = original_norm / recon_norm, which is ~1.0 but not 1.0)
        assert not torch.equal(raw_norms, corrected_norms), (
            "Norms should be corrected (original/recon ratio), not raw L2"
        )
        # But they should be in the same ballpark
        ratio = corrected_norms / raw_norms.clamp(min=1e-8)
        assert ratio.mean().item() == pytest.approx(1.0, abs=0.1)

    @pytest.mark.parametrize("dim", [64, 96, 200])
    def test_non_power_of_2_dim(self, dim):
        """PolarQuant should handle non-power-of-2 dims by padding internally."""
        pq = _PolarQuant(dim, 3, device="cpu")
        assert pq.padded_dim == next_power_of_2(dim)
        assert pq.padded_dim >= dim

        x = torch.randn(8, dim)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)
        assert x_hat.shape == (8, dim)

    def test_rotation_is_orthogonal(self):
        """Forward and inverse rotation should compose to identity."""
        pq = _PolarQuant(128, 3, device="cpu")
        x = torch.randn(4, 128)
        y = pq._rotate(x)
        x_back = pq._rotate_inverse(y)
        assert torch.allclose(x_back, x, atol=1e-5)

    def test_quantizer_cache(self):
        """_get_quantizer should return the same instance for same params."""
        q1 = _get_quantizer(128, 3, "cpu")
        q2 = _get_quantizer(128, 3, "cpu")
        assert q1 is q2

        q3 = _get_quantizer(128, 4, "cpu")
        assert q3 is not q1


# ============================================================================
# process_weights_after_loading: idempotency guard
#
# vLLM calls this twice (online processing + global sweep).
# Second call must be a no-op.
# ============================================================================


class TestProcessWeightsIdempotency:
    def test_double_call_is_noop(self):
        """Second call to process_weights_after_loading should be a no-op."""
        method = TurboQuantW3OnlineLinearMethod()
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(64, 128, device="cpu"))

        method.process_weights_after_loading(layer)

        # After first call: weight replaced with empty, TQ buffers present
        assert layer.weight.numel() == 0
        assert hasattr(layer, "tq_packed_weight")
        packed_id = id(layer.tq_packed_weight)

        # Second call should be a no-op (guard fires)
        method.process_weights_after_loading(layer)
        assert id(layer.tq_packed_weight) == packed_id

    def test_weight_kept_as_empty(self):
        """After compression, layer.weight should be empty(0) — not deleted.

        MLA post-processing in vLLM expects layer.weight to exist.
        """
        method = TurboQuantW3OnlineLinearMethod()
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(64, 128, device="cpu"))

        method.process_weights_after_loading(layer)
        assert hasattr(layer, "weight")
        assert layer.weight.numel() == 0

    def test_guard_attribute_set(self):
        """The _already_called flag should be set after processing."""
        method = TurboQuantW3OnlineLinearMethod()
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(64, 128, device="cpu"))

        assert not getattr(
            layer, "_already_called_process_weights_after_loading", False
        )
        method.process_weights_after_loading(layer)
        assert layer._already_called_process_weights_after_loading is True


# ============================================================================
# End-to-end: compress a Linear weight, check output
# ============================================================================


class TestLinearCompression:
    @pytest.mark.parametrize("bits", [3, 4])
    def test_compress_and_matmul(self, bits):
        """Compress a weight matrix, decompress, verify matmul is close."""
        in_dim, out_dim = 128, 64
        group_size = 128

        w = torch.randn(out_dim, in_dim)
        x = torch.randn(4, in_dim)
        ref = x @ w.t()

        pq = _PolarQuant(group_size, bits, device="cpu")
        grouped = w.reshape(-1, group_size)
        indices, norms_raw = pq.quantize(grouped)
        packed = _pack_indices(indices, bits)

        unpacked = _unpack_indices(packed, bits, group_size)
        w_hat_groups = pq.dequantize(unpacked, norms_raw)
        w_hat = w_hat_groups.reshape(out_dim, -1)[:, :in_dim]

        out = x @ w_hat.t()
        cos_sim = torch.nn.functional.cosine_similarity(ref, out, dim=1)
        threshold = 0.85 if bits == 3 else 0.92
        assert cos_sim.min().item() > threshold, (
            f"bits={bits}: min cosine similarity {cos_sim.min():.4f} < {threshold}"
        )

    def test_apply_fallback_path(self):
        """apply() PyTorch fallback (no Triton)."""
        method = TurboQuantW3OnlineLinearMethod()
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(64, 128, device="cpu"))

        method.process_weights_after_loading(layer)

        # Force PyTorch fallback by clearing Triton dispatch
        layer._tq_primary_fn = None

        x = torch.randn(2, 128, device="cpu")
        out = method.apply(layer, x, bias=None)
        assert out.shape == (2, 64)

        # With bias
        bias = torch.randn(64, device="cpu")
        out_bias = method.apply(layer, x, bias=bias)
        assert out_bias.shape == (2, 64)

    def test_apply_with_padding(self):
        """Compression should work when in_dim is not a multiple of group_size."""
        method = TurboQuantW3OnlineLinearMethod()
        layer = nn.Module()
        # in_dim=200 requires padding to 256 (2 groups of 128)
        layer.weight = nn.Parameter(torch.randn(64, 200, device="cpu"))

        method.process_weights_after_loading(layer)
        layer._tq_primary_fn = None  # force fallback

        x = torch.randn(2, 200, device="cpu")
        out = method.apply(layer, x, bias=None)
        assert out.shape == (2, 64)

    def test_apply_zero_tokens(self):
        """apply() should handle M=0 (zero-token batch in chunked prefill)."""
        method = TurboQuantW3OnlineLinearMethod()
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(64, 128, device="cpu"))

        method.process_weights_after_loading(layer)
        layer._tq_primary_fn = None

        x = torch.randn(0, 128, device="cpu")
        out = method.apply(layer, x, bias=None)
        assert out.shape == (0, 64)

    def test_3d_input(self):
        """apply() should handle 3D input (batch, seq, features)."""
        method = TurboQuantW3OnlineLinearMethod()
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(64, 128, device="cpu"))

        method.process_weights_after_loading(layer)
        layer._tq_primary_fn = None

        x = torch.randn(2, 8, 128, device="cpu")
        out = method.apply(layer, x, bias=None)
        assert out.shape == (2, 8, 64)


class TestTurboQuantClassRegistry:
    """The three concrete subclasses encode bits/group_size as class attrs."""

    @pytest.mark.parametrize(
        ("cls", "bits"),
        [
            (TurboQuantW2OnlineLinearMethod, 2),
            (TurboQuantW3OnlineLinearMethod, 3),
            (TurboQuantW4OnlineLinearMethod, 4),
        ],
    )
    def test_concrete_subclasses(self, cls, bits):
        method = cls()
        assert bits == method.BITS
        assert method.GROUP_SIZE == 128

    def test_base_rejects_unsupported_bits(self):
        class _Bad(_TurboQuantOnlineLinearMethodBase):
            BITS = 5

        with pytest.raises(ValueError, match=r"turboquant BITS must be 2, 3, or 4"):
            _Bad()

    def test_base_rejects_bad_group_size(self):
        class _Bad(_TurboQuantOnlineLinearMethodBase):
            BITS = 3
            GROUP_SIZE = 7

        with pytest.raises(
            ValueError, match=r"turboquant GROUP_SIZE must be a positive"
        ):
            _Bad()
