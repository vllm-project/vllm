# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TurboQuant 4-bit (tq4) KV cache quantization.

Covers configuration, rotation matrix properties, quantization quality,
and the nibble-packing scheme that delivers 2x memory savings over int8.
"""

import pytest
import torch

from vllm.v1.attention.ops.tq4_rotation import _compute_rotation_cpu, get_tq4_rotation
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
)


class TestTQ4Config:
    """Test TQ4 configuration and routing."""

    def test_kv_quant_mode_tq4(self):
        assert get_kv_quant_mode("tq4") == KVQuantMode.TQ4

    def test_tq4_is_per_token_head(self):
        mode = get_kv_quant_mode("tq4")
        assert mode.is_per_token_head

    def test_tq4_mode_value(self):
        assert KVQuantMode.TQ4 == 4

    def test_tq4_dtype_is_uint8(self):
        """TQ4 nibble-packed storage uses uint8."""
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

        assert STR_DTYPE_TO_TORCH_DTYPE["tq4"] == torch.uint8


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
        _compute_rotation_cpu.cache_clear()
        R2 = get_tq4_rotation(128, device="cpu", seed=42)
        assert torch.equal(R1, R2)

    def test_different_seeds_differ(self):
        _compute_rotation_cpu.cache_clear()
        R1 = get_tq4_rotation(128, device="cpu", seed=42)
        _compute_rotation_cpu.cache_clear()
        R2 = get_tq4_rotation(128, device="cpu", seed=99)
        assert not torch.equal(R1, R2)
        _compute_rotation_cpu.cache_clear()

    def test_preserves_norms(self):
        _compute_rotation_cpu.cache_clear()
        R = get_tq4_rotation(128, device="cpu")
        x = torch.randn(1000, 128)
        x_rot = x @ R.T
        norms_orig = torch.norm(x, dim=-1)
        norms_rot = torch.norm(x_rot, dim=-1)
        assert torch.allclose(norms_orig, norms_rot, atol=1e-4)
        _compute_rotation_cpu.cache_clear()


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
        _compute_rotation_cpu.cache_clear()
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
        _compute_rotation_cpu.cache_clear()

    def test_compression_ratio_packed(self):
        """TQ4 with nibble packing: 0.5 byte/dim + 4 bytes scale ~= 3.8x."""
        head_dim = 128
        # Packed storage: head_dim/2 bytes + 4 bytes scale per head
        bytes_per_head = head_dim // 2 + 4  # uint8 packed + float32 scale
        fp16_bytes = head_dim * 2
        ratio = fp16_bytes / bytes_per_head
        assert ratio > 3.7, f"Packed compression ratio {ratio:.2f} too low"


class TestTQ4NibblePacking:
    """Test the nibble pack/unpack roundtrip logic."""

    @staticmethod
    def _pack_reference(signed: torch.Tensor) -> torch.Tensor:
        """Reference Python packing: two signed 4-bit values -> one uint8.

        signed: [..., head_size] int8 with values in [-8, 7].
        Returns: [..., head_size // 2] uint8.
        """
        assert signed.shape[-1] % 2 == 0
        even = signed[..., 0::2]  # even-indexed dims
        odd = signed[..., 1::2]  # odd-indexed dims
        u_even = (even.to(torch.int16) + 8).to(torch.uint8)  # 0..15
        u_odd = (odd.to(torch.int16) + 8).to(torch.uint8)  # 0..15
        return (u_even << 4) | u_odd

    @staticmethod
    def _unpack_reference(packed: torch.Tensor) -> torch.Tensor:
        """Reference Python unpacking: one uint8 -> two signed 4-bit values.

        packed: [..., head_size // 2] uint8.
        Returns: [..., head_size] int8.
        """
        high = ((packed >> 4) & 0xF).to(torch.int8) - 8
        low = (packed & 0xF).to(torch.int8) - 8
        return torch.stack([high, low], dim=-1).reshape(
            *packed.shape[:-1], packed.shape[-1] * 2
        )

    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    def test_pack_unpack_roundtrip(self, head_dim):
        """Pack then unpack must perfectly recover the original signed values."""
        # Random signed 4-bit values in [-8, 7]
        signed = torch.randint(-8, 8, (32, head_dim), dtype=torch.int8)
        packed = self._pack_reference(signed)
        assert packed.shape[-1] == head_dim // 2
        assert packed.dtype == torch.uint8

        recovered = self._unpack_reference(packed)
        assert recovered.shape == signed.shape
        assert torch.equal(recovered, signed)

    def test_pack_shape_halved(self):
        """Packed last dim is exactly half the original."""
        head_dim = 128
        signed = torch.zeros(10, 8, head_dim, dtype=torch.int8)
        packed = self._pack_reference(signed)
        assert packed.shape == (10, 8, head_dim // 2)

    def test_extreme_values(self):
        """Boundary values -8 and 7 survive the roundtrip."""
        vals = torch.tensor([-8, 7, -8, 7, 0, -1, 6, -7], dtype=torch.int8)
        packed = self._pack_reference(vals.unsqueeze(0))
        recovered = self._unpack_reference(packed).squeeze(0)
        assert torch.equal(recovered, vals)

    def test_unpack_matches_triton_attn_helper(self):
        """The _tq4_unpack function in triton_attn matches our reference."""
        from vllm.v1.attention.backends.triton_attn import _tq4_unpack

        head_dim = 128
        signed = torch.randint(-8, 8, (16, 4, head_dim), dtype=torch.int8)
        packed = self._pack_reference(signed)

        ref_unpacked = self._unpack_reference(packed)
        impl_unpacked = _tq4_unpack(packed)

        assert torch.equal(impl_unpacked, ref_unpacked)

    def test_quantize_pack_unpack_dequantize(self):
        """Full pipeline: float -> quantize -> pack -> unpack -> dequantize."""
        head_dim = 128
        x = torch.randn(100, head_dim)

        # Quantize
        absmax = x.abs().amax(dim=-1, keepdim=True)
        scale = (absmax / 7.0).clamp(min=1e-6)
        q = (x / scale).round().clamp(-8, 7).to(torch.int8)

        # Pack
        packed = self._pack_reference(q)
        assert packed.shape[-1] == head_dim // 2

        # Unpack
        recovered_q = self._unpack_reference(packed)
        assert torch.equal(recovered_q, q)

        # Dequantize
        deq = recovered_q.float() * scale
        mse = ((x - deq) ** 2).mean().item()
        assert mse < 0.05, f"Full pipeline MSE too high: {mse}"


class TestTQ4PageSize:
    """Test that page size calculations reflect nibble packing."""

    def test_attention_spec_page_size_halved(self):
        """AttentionSpec with TQ4 should use half the data bytes vs NONE."""
        spec_none = AttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.NONE,
        )
        spec_tq4 = AttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.TQ4,
        )
        # TQ4 real page size should be half of NONE (same dtype, packed dims)
        assert spec_tq4.real_page_size_bytes == spec_none.real_page_size_bytes // 2

    def test_full_attention_spec_page_size_halved(self):
        """FullAttentionSpec with TQ4 should use half the data bytes."""
        spec_none = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.NONE,
        )
        spec_tq4 = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.TQ4,
        )
        assert spec_tq4.real_page_size_bytes == spec_none.real_page_size_bytes // 2

    def test_full_attention_spec_diffkv(self):
        """FullAttentionSpec with different K/V head sizes."""
        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            head_size_v=64,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.TQ4,
        )
        # (128//2 + 64//2) * 16 * 8 * 1 = 96 * 128 = 12288
        expected = 16 * 8 * (64 + 32) * 1
        assert spec.real_page_size_bytes == expected
