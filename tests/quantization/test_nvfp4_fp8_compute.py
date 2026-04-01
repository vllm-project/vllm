# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NVFP4-to-FP8 load-time conversion on Hopper GPUs.

Tests the FP8_COMPUTE backend which converts NVFP4 (FP4 E2M1) weights to
FP8 block-quantized format at model load time, enabling native FP8 tensor
core compute instead of the Marlin FP4->FP16 fallback.
"""

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    ref_nvfp4_quant,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    FP8_COMPUTE_BLOCK_SIZE,
    NvFp4LinearBackend,
    convert_nvfp4_weight_to_fp8_block,
)


def _make_nvfp4_weight(
    N: int, K: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a synthetic NVFP4 weight with realistic quantization.

    Returns (weight_fp4_packed, weight_scale, weight_global_scale).
    """
    group_size = 16
    assert K % group_size == 0

    w_bf16 = torch.randn(N, K, dtype=torch.float32, device=device) * 0.1
    global_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    w_fp4_f32, scale = ref_nvfp4_quant(w_bf16, global_scale, group_size)

    # Pack FP4 values into uint8 (2 values per byte)
    fp4_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    abs_w = w_fp4_f32.abs()
    signs = (w_fp4_f32 < 0).to(torch.uint8)

    indices = torch.zeros_like(abs_w, dtype=torch.uint8)
    for i, v in enumerate(fp4_values):
        mask = (abs_w - v).abs() < 0.01
        indices[mask] = i

    nibbles = (signs << 3) | indices
    nibbles_2d = nibbles.reshape(N, K // 2, 2)
    packed = (nibbles_2d[:, :, 1] << 4) | nibbles_2d[:, :, 0]
    weight_packed = packed.to(torch.uint8)

    scale_fp8 = scale.squeeze(-1).to(torch.float8_e4m3fn)

    return weight_packed, scale_fp8, global_scale


class TestNvfp4ToFp8Conversion:
    """Test the FP4->FP8 weight conversion utility."""

    def test_conversion_produces_correct_dtypes(self):
        """Converted weights should be FP8 with FP32 scales."""
        N, K = 128, 256
        w_packed, w_scale, g_scale = _make_nvfp4_weight(N, K)
        w_fp8, w_fp8_scale = convert_nvfp4_weight_to_fp8_block(
            w_packed, w_scale, g_scale
        )
        assert w_fp8.dtype == torch.float8_e4m3fn
        assert w_fp8_scale.dtype == torch.float32

    def test_conversion_preserves_shape(self):
        """FP8 weight should be (N, K) — same layout as dequantized BF16."""
        N, K = 256, 512
        w_packed, w_scale, g_scale = _make_nvfp4_weight(N, K)
        w_fp8, w_fp8_scale = convert_nvfp4_weight_to_fp8_block(
            w_packed, w_scale, g_scale
        )
        assert w_fp8.shape == (N, K)
        block_n, block_k = FP8_COMPUTE_BLOCK_SIZE
        expected_scale_shape = (
            (N + block_n - 1) // block_n,
            (K + block_k - 1) // block_k,
        )
        assert w_fp8_scale.shape == expected_scale_shape

    def test_conversion_accuracy(self):
        """FP8 dequantized values should closely match original FP4 dequant."""
        N, K = 256, 512
        w_packed, w_scale, g_scale = _make_nvfp4_weight(N, K)

        # Reference: FP4 -> BF16
        w_ref = dequantize_to_dtype(
            w_packed, w_scale, g_scale, torch.float32, w_packed.device
        )

        w_fp8, w_fp8_scale = convert_nvfp4_weight_to_fp8_block(
            w_packed, w_scale, g_scale
        )
        block_m, block_k = FP8_COMPUTE_BLOCK_SIZE
        w_fp8_f32 = w_fp8.to(torch.float32)
        scale_expanded = w_fp8_scale.repeat_interleave(
            block_m, dim=0
        ).repeat_interleave(block_k, dim=1)
        w_reconstructed = w_fp8_f32 * scale_expanded[:N, :K]

        cos_sim = torch.nn.functional.cosine_similarity(
            w_ref.flatten().unsqueeze(0),
            w_reconstructed.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.99, (
            f"FP4->FP8 conversion accuracy too low: cosine_sim={cos_sim:.4f}"
        )

    def test_conversion_non_aligned_dimensions(self):
        """Test with dimensions not perfectly aligned to 128."""
        N, K = 192, 384
        w_packed, w_scale, g_scale = _make_nvfp4_weight(N, K)
        w_fp8, w_fp8_scale = convert_nvfp4_weight_to_fp8_block(
            w_packed, w_scale, g_scale
        )
        assert w_fp8.shape == (N, K)

    def test_zero_weights(self):
        """Conversion should handle all-zero weights."""
        N, K = 128, 256
        w_packed = torch.zeros(N, K // 2, dtype=torch.uint8)
        w_scale = torch.ones(N, K // 16, dtype=torch.float8_e4m3fn)
        g_scale = torch.tensor(1.0, dtype=torch.float32)
        w_fp8, _ = convert_nvfp4_weight_to_fp8_block(w_packed, w_scale, g_scale)
        assert (w_fp8.to(torch.float32) == 0).all()


class TestBackendSelection:
    """Test that FP8_COMPUTE backend is correctly selected."""

    def test_fp8_compute_enum_exists(self):
        """FP8_COMPUTE should be a valid backend."""
        assert hasattr(NvFp4LinearBackend, "FP8_COMPUTE")
        assert NvFp4LinearBackend.FP8_COMPUTE.value == "fp8-compute"

    def test_fp8_compute_moe_enum_exists(self):
        """FP8_COMPUTE should be a valid MoE backend."""
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
            NvFp4MoeBackend,
        )

        assert hasattr(NvFp4MoeBackend, "FP8_COMPUTE")


class TestMoeConversion:
    """Test MoE-specific FP4->FP8 conversion."""

    def test_moe_conversion_shapes(self):
        """MoE expert weight conversion should produce correct shapes."""
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
            _convert_nvfp4_moe_to_fp8_compute,
        )

        E, N, K = 4, 256, 512

        w13_list, w13_scale_list, g_scales_13 = [], [], []
        w2_list, w2_scale_list, g_scales_2 = [], [], []

        for _ in range(E):
            wp, ws, gs = _make_nvfp4_weight(N * 2, K)  # w13 is gate+up
            w13_list.append(wp)
            w13_scale_list.append(ws)
            g_scales_13.append(gs)

            wp2, ws2, gs2 = _make_nvfp4_weight(K, N)  # w2 is down
            w2_list.append(wp2)
            w2_scale_list.append(ws2)
            g_scales_2.append(gs2)

        result = _convert_nvfp4_moe_to_fp8_compute(
            torch.stack(w13_list),
            torch.stack(w13_scale_list),
            torch.stack(g_scales_13),
            torch.stack(w2_list),
            torch.stack(w2_scale_list),
            torch.stack(g_scales_2),
        )

        (w13_fp8, w13_fp8_scale, w13_s2, a13_s, w2_fp8, w2_fp8_scale, w2_s2, a2_s) = (
            result
        )

        assert w13_fp8.dtype == torch.float8_e4m3fn
        assert w13_fp8.shape == (E, N * 2, K)
        assert w2_fp8.dtype == torch.float8_e4m3fn
        assert w2_fp8.shape == (E, K, N)
        # Activation and global scales should be None
        assert w13_s2 is None and a13_s is None
        assert w2_s2 is None and a2_s is None
