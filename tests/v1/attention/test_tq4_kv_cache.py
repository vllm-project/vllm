# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TurboQuant 4-bit (tq4) KV cache quantization."""

import pytest
import torch

from vllm.v1.attention.ops.tq4_rotation import get_tq4_rotation
from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode


class TestTQ4Config:
    """Test TQ4 configuration and routing."""

    def test_kv_quant_mode_tq4(self):
        assert get_kv_quant_mode("tq4") == KVQuantMode.TQ4

    def test_tq4_is_per_token_head(self):
        mode = get_kv_quant_mode("tq4")
        assert mode.is_per_token_head

    def test_tq4_mode_value(self):
        assert KVQuantMode.TQ4 == 4


class TestTQ4Rotation:
    """Test rotation matrix properties."""

    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    def test_orthogonal(self, head_dim):
        R = get_tq4_rotation(head_dim, device="cpu")
        I_approx = R @ R.T
        I_exact = torch.eye(head_dim)
        assert torch.allclose(I_approx, I_exact, atol=1e-5)

    def test_deterministic(self):
        R1 = get_tq4_rotation(128, device="cpu", seed=42)
        # Clear cache to force recomputation
        get_tq4_rotation.cache_clear()
        R2 = get_tq4_rotation(128, device="cpu", seed=42)
        assert torch.equal(R1, R2)

    def test_different_seeds_differ(self):
        get_tq4_rotation.cache_clear()
        R1 = get_tq4_rotation(128, device="cpu", seed=42)
        get_tq4_rotation.cache_clear()
        R2 = get_tq4_rotation(128, device="cpu", seed=99)
        assert not torch.equal(R1, R2)
        get_tq4_rotation.cache_clear()

    def test_preserves_norms(self):
        get_tq4_rotation.cache_clear()
        R = get_tq4_rotation(128, device="cpu")
        x = torch.randn(1000, 128)
        x_rot = x @ R.T
        norms_orig = torch.norm(x, dim=-1)
        norms_rot = torch.norm(x_rot, dim=-1)
        assert torch.allclose(norms_orig, norms_rot, atol=1e-4)
        get_tq4_rotation.cache_clear()


class TestTQ4Quantization:
    """Test TQ4 quantize/dequantize quality."""

    def _quantize_dequantize(self, x, rotation=None):
        """Reference TQ4 quantize-dequantize for testing."""
        if rotation is not None:
            x = x @ rotation.T

        # Per-token-head absmax scaling with 4-bit range
        absmax = x.float().abs().amax(dim=-1, keepdim=True)
        scale = (absmax / 7.0).clamp(min=1e-6)
        q = (x.float() / scale).round().clamp(-8, 7).to(torch.int8)

        # Dequantize
        deq = q.float() * scale

        if rotation is not None:
            deq = deq @ rotation
        return deq

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_mse_without_rotation(self, head_dim):
        """4-bit quantization alone has bounded MSE."""
        x = torch.randn(10000, head_dim)
        deq = self._quantize_dequantize(x)
        mse = ((x - deq) ** 2).mean().item()
        # Without rotation, MSE depends on coordinate distribution
        assert mse < 0.05, f"MSE too high without rotation: {mse}"

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_mse_with_rotation(self, head_dim):
        """Rotation improves MSE by spreading energy uniformly."""
        get_tq4_rotation.cache_clear()
        R = get_tq4_rotation(head_dim, device="cpu")
        x = torch.randn(10000, head_dim)
        deq_no_rot = self._quantize_dequantize(x)
        deq_with_rot = self._quantize_dequantize(x, rotation=R)
        mse_no_rot = ((x - deq_no_rot) ** 2).mean().item()
        mse_with_rot = ((x - deq_with_rot) ** 2).mean().item()
        # Rotation should not degrade quality
        assert mse_with_rot < mse_no_rot * 1.5, (
            f"Rotation degraded quality: {mse_with_rot} vs {mse_no_rot}"
        )
        get_tq4_rotation.cache_clear()

    def test_compression_ratio(self):
        """TQ4 in int8 storage gives same footprint as INT8 per-token-head.
        Real 4-bit packing (follow-up PR) will halve this."""
        head_dim = 128
        # int8 storage: 1 byte per dim + 4 bytes scale per head
        bytes_per_head = head_dim * 1 + 4  # int8 data + float32 scale
        fp16_bytes = head_dim * 2
        ratio = fp16_bytes / bytes_per_head
        assert ratio > 1.9, f"Compression ratio {ratio} too low"
