# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for GPTQ int4 -> int8 re-quantization pipeline.

Part 1 (G2 & G3): Pure Python tests
  - _dequant_gptq_to_float (with and without g_idx / desc_act)
  - _requantize_to_int8 (reused from AWQ A3)
  - Full GPTQ unpack → dequant → re-quantize roundtrip

Part 2 (G4 & G5): oneDNN integration tests — requires CPU C++ extensions.
  - oneDNN handler creation from GPTQ int4 weights
  - End-to-end: GPTQ int4 → int8 → oneDNN GEMM vs reference

Run:
    python -m pytest tests/kernels/test_gptq_int4_to_int8.py -v -s

"""

import torch
import pytest
import numpy as np


from vllm.model_executor.layers.quantization.kernels.mixed_precision.cpu import (
    _dequant_gptq_to_float,
)
from vllm.model_executor.layers.quantization.cpu_wna16 import (
    _requantize_to_int8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
    unpack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types

# oneDNN ops: may not be available outside Docker / CPU build
_has_onednn = False
try:
    from vllm import _custom_ops as ops
    from vllm.platforms import current_platform
    if current_platform.is_cpu():
        _has_onednn = True
except Exception:
    pass

requires_onednn = pytest.mark.skipif(not _has_onednn,reason="Requires vLLM CPU build with oneDNN C++ extensions")






# lyt_debug Helpers to create synthetic GPTQ-like data

def make_synthetic_gptq_data(
    K: int, N: int, group_size: int, seed: int = 42,
    use_g_idx: bool = False,
):
    """Create synthetic GPTQ-style quantized data.

    GPTQ uses uint4b8 format: raw values [0, 15], representing signed [-8, 7].No zero point.

    Returns:
        weight_int4: [K, N] int32, raw values in [0, 15]
        scales:      [num_groups, N] float32, per-group per-channel scale
        g_idx:       [K] int32 or None, group index for desc_act
        float_ref:   [K, N] float32, reference dequantized weights
    """
    rng = np.random.RandomState(seed)
    num_groups = K // group_size

    weight_int4_np = rng.randint(0, 16, size=(K, N)).astype(np.int32)
    scales_np = (rng.randn(num_groups, N) * 0.05).astype(np.float32)

    weight_int4 = torch.from_numpy(weight_int4_np)
    scales = torch.from_numpy(scales_np)

    # signed values: raw - 8
    signed_vals = weight_int4.float() - 8.0

    if use_g_idx:    # lyt_debug Simulate desc_act: shuffle group assignments
        g_idx_np = np.zeros(K, dtype=np.int32)
        for g in range(num_groups):
            g_idx_np[g * group_size:(g + 1) * group_size] = g
        rng.shuffle(g_idx_np)
        g_idx = torch.from_numpy(g_idx_np)
        scales_expanded = scales[g_idx.long(), :]
    else:
        g_idx = None
        scales_expanded = scales.repeat_interleave(group_size, dim=0)

    float_ref = signed_vals * scales_expanded.float()

    return weight_int4, scales, g_idx, float_ref


def _gptq_int4_to_int8_pipeline(
    K: int, N: int, group_size: int, seed: int = 42,
    use_g_idx: bool = False,
):
    """Full pipeline: synthetic GPTQ data → dequant → int8 requant.

    Returns:
        weight_int8:    [K, N] int8, column-major (stride(0)==1)
        channel_scale:  [N] float32
        float_weight:   [K, N] float32 (intermediate)
    """
    weight_int4, scales, g_idx, _ = make_synthetic_gptq_data(
        K, N, group_size, seed, use_g_idx
    )
    float_weight = _dequant_gptq_to_float(weight_int4, scales, group_size, g_idx)
    weight_int8, channel_scale = _requantize_to_int8(float_weight)
    weight_int8 = weight_int8.t().contiguous().t()
    return weight_int8, channel_scale, float_weight





# lyy_debug est G2: _dequant_gptq_to_float

class TestDequantGPTQToFloat:
    """Tests for _dequant_gptq_to_float (G2)."""

    @pytest.mark.parametrize("K,N,group_size", [
        (128, 64, 128),
        (256, 256, 128),
        (512, 128, 64),
        (1024, 512, 128),
    ])
    def test_dequant_matches_reference(self, K, N, group_size):
        """Dequant output should exactly match manual per-group computation."""
        weight_int4, scales, _, float_ref = make_synthetic_gptq_data(
            K, N, group_size
        )

        float_result = _dequant_gptq_to_float(weight_int4, scales, group_size)
        torch.testing.assert_close(float_result, float_ref, rtol=1e-5, atol=1e-5)
        print(f"  [PASS] dequant K={K}, N={N}, group_size={group_size}")

    def test_dequant_zero_weights(self):
        """When signed int4 value is 0 (raw=8), dequant should produce zero."""
        K, N, group_size = 128, 64, 128
        weight_int4 = torch.full((K, N), 8, dtype=torch.int32)  # raw 8 → signed 0
        scales = torch.ones(1, N, dtype=torch.float32) * 0.1

        result = _dequant_gptq_to_float(weight_int4, scales, group_size)

        torch.testing.assert_close(
            result, torch.zeros_like(result), rtol=0, atol=1e-7
        )
        print("  [PASS] dequant zero weights (raw=8 → signed=0)")

    def test_dequant_extreme_values(self):
        """Extreme int4 values: raw 0 → signed -8, raw 15 → signed +7."""
        K, N, group_size = 64, 32, 64
        scale_val = 0.5

        weight_min = torch.full((K, N), 0, dtype=torch.int32)
        weight_max = torch.full((K, N), 15, dtype=torch.int32)
        scales = torch.full((1, N), scale_val, dtype=torch.float32)

        result_min = _dequant_gptq_to_float(weight_min, scales, group_size)
        result_max = _dequant_gptq_to_float(weight_max, scales, group_size)

        expected_min = torch.full((K, N), -8.0 * scale_val, dtype=torch.float32)
        expected_max = torch.full((K, N), 7.0 * scale_val, dtype=torch.float32)

        torch.testing.assert_close(result_min, expected_min, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(result_max, expected_max, rtol=1e-5, atol=1e-5)
        print("  [PASS] dequant extreme values")


# lyt_debug Testing G6: desc_act (g_idx) handling

class TestDescActGIdx:
    """Tests for G6: desc_act handling with g_idx in dequant."""

    @pytest.mark.parametrize("K,N,group_size", [
        (128, 64, 64),
        (256, 128, 128),
        (512, 256, 64),
    ])
    def test_g_idx_dequant_matches_reference(self, K, N, group_size):
        """With g_idx, dequant should match element-wise reference."""
        weight_int4, scales, g_idx, float_ref = make_synthetic_gptq_data(
            K, N, group_size, use_g_idx=True
        )

        float_result = _dequant_gptq_to_float(
            weight_int4, scales, group_size, g_idx
        )

        torch.testing.assert_close(
            float_result, float_ref, rtol=1e-5, atol=1e-5
        )
        print(f"  [PASS] g_idx dequant K={K}, N={N}, gs={group_size}")

    def test_g_idx_vs_no_g_idx_uniform(self):
        """With sequential g_idx, result should match uniform grouping."""
        K, N, group_size = 256, 128, 128
        num_groups = K // group_size

        weight_int4, scales, _, _ = make_synthetic_gptq_data(K, N, group_size)
        g_idx_uniform = torch.arange(K, dtype=torch.int32) // group_size

        result_no_gidx = _dequant_gptq_to_float(
            weight_int4, scales, group_size, None
        )
        result_with_gidx = _dequant_gptq_to_float(
            weight_int4, scales, group_size, g_idx_uniform
        )

        torch.testing.assert_close(
            result_no_gidx, result_with_gidx, rtol=1e-5, atol=1e-5
        )
        print("  [PASS] uniform g_idx matches no-g_idx path")




# lyt_debug Test G3: _requantize_to_int8 (reused from AWQ A3)
class TestGPTQRequantizeToInt8:
    """G3: re-quantize float32 weights to int8 (reuses AWQ's _requantize_to_int8)."""

    @pytest.mark.parametrize("K,N,group_size", [
        (128, 128, 128),
        (256, 256, 128),
        (512, 128, 64),
    ])
    def test_requantize_gptq_pipeline(self, K, N, group_size):
        """Full GPTQ pipeline: dequant → int8 re-quantize shoud preserve signed int4 values within int8 range."""
        weight_int4, scales, _, float_ref = make_synthetic_gptq_data(
            K, N, group_size
        )
        float_weight = _dequant_gptq_to_float(weight_int4, scales, group_size)
        weight_int8, channel_scale = _requantize_to_int8(float_weight)

        assert weight_int8.dtype == torch.int8
        assert weight_int8.shape == (K, N)
        assert channel_scale.shape == (N,)
        assert weight_int8.min().item() >= -128
        assert weight_int8.max().item() <= 127

        # lyt_debugChecking roundtrip error
        reconstructed = weight_int8.float() * channel_scale.unsqueeze(0)
        rel_err_mask = float_weight.abs() > 1e-6
        rel_err = (float_weight - reconstructed).abs()
        mean_rel = (
            rel_err[rel_err_mask] / float_weight[rel_err_mask].abs()
        ).mean().item()

        assert mean_rel < 0.10, f"mean_rel_err {mean_rel:.4f} exceeds 0.10"
        print(f"  [PASS] GPTQ pipeline K={K}, N={N}, gs={group_size}: "
              f"mean_rel_err={mean_rel:.4f}")

    def test_requantize_with_g_idx(self):
        """Pipeline with desc_act should also produce valid int8 weights."""
        K, N, group_size = 256, 128, 64
        weight_int4, scales, g_idx, _ = make_synthetic_gptq_data(
            K, N, group_size, use_g_idx=True
        )
        float_weight = _dequant_gptq_to_float(
            weight_int4, scales, group_size, g_idx
        )
        weight_int8, channel_scale = _requantize_to_int8(float_weight)

        assert weight_int8.dtype == torch.int8
        assert weight_int8.shape == (K, N)
        assert weight_int8.max().item() == 127 or weight_int8.min().item() == -127
        print(f"  [PASS] GPTQ pipeline with g_idx: K={K}, N={N}")



# lyt_debug Test G4: oneDNN handler creation from GPTQ int4 weights
class TestGPTQOneDNNHandlerCreation:
    """G4: Create oneDNN int8 GEMM handler from GPTQ-derived int8 weights."""

    @requires_onednn
    @pytest.mark.parametrize("K,N,group_size", [
        (128, 128, 128),
        (256, 256, 128),
        (512, 256, 64),
    ])
    def test_handler_creation(self, K, N, group_size):
        """Handler should be created successfully with correct K, N."""
        weight_int8, channel_scale, _ = _gptq_int4_to_int8_pipeline(
            K, N, group_size
        )
        channel_scale_2d = channel_scale.unsqueeze(0)

        handler = ops.create_onednn_scaled_mm(
            weight_int8, channel_scale_2d, torch.bfloat16,
            True, False, 32,
        )

        
        assert handler.n == N
        assert handler.k == K
        print(f"  [PASS] handler creation K={K}, N={N}, gs={group_size}")

    @requires_onednn
    def test_handler_creation_with_g_idx(self):
        """Handler creation should work with desc_act (g_idx) weights."""
        K, N, group_size = 256, 128, 64
        weight_int8, channel_scale, _ = _gptq_int4_to_int8_pipeline(
            K, N, group_size, use_g_idx=True
        )
        channel_scale_2d = channel_scale.unsqueeze(0)

        handler = ops.create_onednn_scaled_mm(
            weight_int8, channel_scale_2d, torch.bfloat16,
            True, False, 32,
        )

        assert handler.k == K
        assert handler.n == N
        print(f"  [PASS] handler creation with g_idx K={K}, N={N}")


# lyt_debug Test G5: End-to-end GPTQ int4 → int8 → oneDNN GEMM
def _ref_onednn_dynamic_gemm(x_q, weight_int8, x_s, channel_scale_2d,
                              bias, out_dtype):
    """Reference matching oneDNN dynamic-quant int8 GEMM execution flow."""
    int_mm = torch.mm(x_q.to(torch.float64),
                      weight_int8.contiguous().to(torch.float64))
    tmp_f32 = int_mm.float() * channel_scale_2d
    out_f32 = x_s * tmp_f32
    if bias is not None:
        out_f32 = out_f32 + bias.float()
    return out_f32.to(out_dtype)


class TestGPTQOneDNNInt8GEMM:
    """G5: oneDNN int8 GEMM with GPTQ-derived int8 weights."""
    @requires_onednn
    @pytest.mark.parametrize("M,K,N,group_size", [
        (1, 128, 128, 128),
        (4, 256, 256, 128),
        (16, 512, 256, 64),
        (32, 256, 512, 128),
    ])
    def test_int8_gemm_kernel_correctness(self, M, K, N, group_size):
        """oneDNN int8 GEMM should match exact int8 reference."""
        weight_int8, channel_scale, _ = _gptq_int4_to_int8_pipeline(
            K, N, group_size
        )
        channel_scale_2d = channel_scale.unsqueeze(0)

        handler = ops.create_onednn_scaled_mm(
            weight_int8, channel_scale_2d, torch.bfloat16,
            True, False, 32,
        )

        x = torch.randn(M, K, dtype=torch.bfloat16)
        x_q, x_s, _ = ops.onednn_scaled_int8_quant(x, None, None, True)

        out_int8 = torch.zeros((M, N), dtype=torch.bfloat16)
        ops.onednn_scaled_mm(handler, x_q, out_int8, x_s, None, None, None)
        out_ref = _ref_onednn_dynamic_gemm(
            x_q, weight_int8, x_s, channel_scale_2d, None, torch.bfloat16
        )

        abs_diff = (out_int8.float() - out_ref.float()).abs()
        mean_abs = abs_diff.mean().item()
        pct95 = torch.quantile(abs_diff, 0.95).item()

        print(f"  lyt_debug_G5 kernel check M={M}, K={K}, N={N}, gs={group_size}: "
              f"max_abs={abs_diff.max().item():.4f}, mean_abs={mean_abs:.4f}, "
              f"pct95_abs={pct95:.4f}")

        assert mean_abs < 2.0, (
            f"mean_abs_diff {mean_abs:.4f} too large (threshold 2.0)")
        assert pct95 < 5.0, (
            f"95th-percentile abs_diff {pct95:.4f} too large (threshold 5.0)")
        print(f"  [PASS] GPTQ int8 GEMM: M={M}, K={K}, N={N}")

    @requires_onednn
    @pytest.mark.parametrize("M", [1, 8, 32])
    def test_int8_gemm_with_bias(self, M):
        """oneDNN int8 GEMM with bias should match reference."""
        K, N, group_size = 256, 128, 128
        weight_int8, channel_scale, _ = _gptq_int4_to_int8_pipeline(
            K, N, group_size
        )
        channel_scale_2d = channel_scale.unsqueeze(0)
        bias = torch.randn(N, dtype=torch.bfloat16)

        handler = ops.create_onednn_scaled_mm(
            weight_int8, channel_scale_2d, torch.bfloat16,
            True, False, 32,
        )

        x = torch.randn(M, K, dtype=torch.bfloat16)
        x_q, x_s, _ = ops.onednn_scaled_int8_quant(x, None, None, True)
        out_int8 = torch.zeros((M, N), dtype=torch.bfloat16)
        ops.onednn_scaled_mm(handler, x_q, out_int8, x_s, None, None, bias)

        out_ref = _ref_onednn_dynamic_gemm(
            x_q, weight_int8, x_s, channel_scale_2d, bias, torch.bfloat16
        )

        abs_diff = (out_int8.float() - out_ref.float()).abs()
        mean_abs = abs_diff.mean().item()
        pct95 = torch.quantile(abs_diff, 0.95).item()
        print(f"  lyt_debug_G5 bias M={M}: max_abs={abs_diff.max().item():.4f}, "
              f"mean_abs={mean_abs:.4f}, pct95_abs={pct95:.4f}")

        assert mean_abs < 2.0, (
            f"mean_abs_diff {mean_abs:.4f} with bias exceeds threshold")
        assert pct95 < 5.0, (
            f"95th-pctile abs_diff {pct95:.4f} with bias exceeds threshold")
        print(f"  [PASS] GPTQ int8 GEMM with bias: M={M}")

    @requires_onednn
    def test_int8_gemm_3d_input(self):
        """apply-fucntion reshapes 3D input [B, S, K] → [B*S, K] → back to 3D."""
        K, N, group_size = 256, 128, 128
        weight_int8, channel_scale, _ = _gptq_int4_to_int8_pipeline(
            K, N, group_size
        )
        channel_scale_2d = channel_scale.unsqueeze(0)

        handler = ops.create_onednn_scaled_mm(
            weight_int8, channel_scale_2d, torch.bfloat16,
            True, False, 32,
        )

        B, S = 2, 8
        x_3d = torch.randn(B, S, K, dtype=torch.bfloat16)
        x_2d = x_3d.reshape(-1, K)

        x_q, x_s, _ = ops.onednn_scaled_int8_quant(x_2d, None, None, True)
        out_2d = torch.zeros((B * S, N), dtype=torch.bfloat16)
        ops.onednn_scaled_mm(handler, x_q, out_2d, x_s, None, None, None)
        out_3d = out_2d.reshape(B, S, N)

        out_ref = _ref_onednn_dynamic_gemm(
            x_q, weight_int8, x_s, channel_scale_2d, None, torch.bfloat16
        ).reshape(B, S, N)

        assert out_3d.shape == (B, S, N)
        abs_diff = (out_3d.float() - out_ref.float()).abs()
        mean_abs = abs_diff.mean().item()
        pct95 = torch.quantile(abs_diff, 0.95).item()
        print(f"  lyt_debug_G5 3D: max_abs={abs_diff.max().item():.4f}, "
              f"mean_abs={mean_abs:.4f}, pct95_abs={pct95:.4f}")
        assert mean_abs < 2.0
        assert pct95 < 5.0
        print(f"  [PASS] 3D input [{B},{S},{K}] → output [{B},{S},{N}]")

    @requires_onednn
    def test_int8_gemm_with_g_idx(self):
        """End-to-end GEMM with desc_act (g_idx) weights."""
        M, K, N, group_size = 4, 256, 128, 64
        weight_int8, channel_scale, _ = _gptq_int4_to_int8_pipeline(
            K, N, group_size, use_g_idx=True
        )
        channel_scale_2d = channel_scale.unsqueeze(0)

        handler = ops.create_onednn_scaled_mm(
            weight_int8, channel_scale_2d, torch.bfloat16,
            True, False, 32,
        )

        x = torch.randn(M, K, dtype=torch.bfloat16)
        x_q, x_s, _ = ops.onednn_scaled_int8_quant(x, None, None, True)
        out_int8 = torch.zeros((M, N), dtype=torch.bfloat16)
        ops.onednn_scaled_mm(handler, x_q, out_int8, x_s, None, None, None)

        out_ref = _ref_onednn_dynamic_gemm(
            x_q, weight_int8, x_s, channel_scale_2d, None, torch.bfloat16
        )

        abs_diff = (out_int8.float() - out_ref.float()).abs()
        mean_abs = abs_diff.mean().item()
        pct95 = torch.quantile(abs_diff, 0.95).item()
        print(f"lyt_debug_G5 g_idx: max_abs={abs_diff.max().item():.4f}, "
              f"mean_abs={mean_abs:.4f}, pct95_abs={pct95:.4f}")
        assert mean_abs < 2.0
        assert pct95 < 5.0
        print(f"  [PASS] GPTQ int8 GEMM with g_idx: M={M}, K={K}, N={N}")

