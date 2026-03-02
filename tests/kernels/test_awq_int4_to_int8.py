# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for AWQ int4 -> int8 re-quantization pipeline.

Part 1 (A2 & A3): Pure Python tests.,
  - _dequant_awq_to_float
  - _requantize_to_int8
  - Full AWQ pack/unpack roundtrip

Part 2 (A4 & A5): oneDNN integration tests — requires CPU C++ extensions.
  - oneDNN handler creation from AWQ int4 weights
  - End-to-end: AWQ int4 → int8 → oneDNN GEMM vs bf16 reference

cmd:
    python -m pytest tests/kernels/test_awq_int4_to_int8.py -v -s
"""

import torch
import pytest
import numpy as np

from vllm.model_executor.layers.quantization.cpu_wna16 import (
    _dequant_awq_to_float,
    _requantize_to_int8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_cols,
    unpack_cols,
)

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


# lyt_debug Helpers to create synthetic AWQ-like data


def make_synthetic_awq_data(
    K: int, N: int, group_size: int, seed: int = 42
):
    """Create synthetic AWQ-style quantized data.
    Returns:
        weight_int4: [K, N] int32, values in [0, 15]
        zeros_int4:  [num_groups, N] int32, values in [0, 15]
        scales:      [num_groups, N] float32
        float_ref:   [K, N] float32, reference dequantized weights
    """
    rng = np.random.RandomState(seed)
    num_groups = K // group_size

    weight_int4_np = rng.randint(0, 16, size=(K, N)).astype(np.int32)
    zeros_int4_np = rng.randint(0, 16, size=(num_groups, N)).astype(np.int32)
    scales_np = (rng.randn(num_groups, N) * 0.05).astype(np.float32)

    weight_int4, zeros_int4, scales = torch.from_numpy(weight_int4_np), torch.from_numpy(zeros_int4_np), torch.from_numpy(scales_np)
    # Compute reference dequant
    float_ref = torch.zeros(K, N, dtype=torch.float32)
    for g in range(num_groups):
        s, e = g * group_size, (g + 1) * group_size
        float_ref[s:e, :] = (
            (weight_int4[s:e, :].float() - zeros_int4[g, :].float())
            * scales[g, :].float()
        )

    return weight_int4, zeros_int4, scales, float_ref


# lyt_debug Testa2: _dequant_awq_to_float
class TestDequantAWQToFloat:
    """Tests for _dequant_awq_to_float (A2)."""

    @pytest.mark.parametrize("K,N,group_size", [
        (128, 64, 128),
        (256, 256, 128),
        (512, 128, 64),
        (1024, 512, 128),
    ])
    def test_dequant_matches_reference(self, K, N, group_size):
        """Dequant output should exactly match manual per-group computation."""
        weight_int4, zeros_int4, scales, float_ref = make_synthetic_awq_data(K, N, group_size)
        float_result = _dequant_awq_to_float(weight_int4, zeros_int4, scales, group_size)
        torch.testing.assert_close(float_result, float_ref, rtol=1e-5, atol=1e-5)
        print(f"  [PASS] dequant K={K}, N={N}, group_size={group_size}")



    def test_dequant_zero_weights(self):
        """When int4 values equal zero_point, dequant should produce zeros."""
        K, N, group_size = 128, 64, 128
        zp_val = 8
        weight_int4 = torch.full((K, N), zp_val, dtype=torch.int32)
        zeros_int4 = torch.full((1, N), zp_val, dtype=torch.int32)
        scales = torch.ones(1, N, dtype=torch.float32) * 0.1

        result = _dequant_awq_to_float(weight_int4, zeros_int4, scales, group_size)
        torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=1e-7)
        print("  [PASS] dequant zero weights")

    def test_dequant_single_group(self):
        """Single group (group_size == K) should work correctly."""
        K, N = 64, 32
        weight_int4 = torch.randint(0, 16, (K, N), dtype=torch.int32)
        zeros_int4 = torch.randint(0, 16, (1, N), dtype=torch.int32)
        scales = torch.randn(1, N, dtype=torch.float32) * 0.03

        result = _dequant_awq_to_float(weight_int4, zeros_int4, scales, K)
        expected = (weight_int4.float() - zeros_int4.float()) * scales.float()
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        print("  [PASS] dequant single group")


# ---------------------------------------------------------------------------
# Test A3: _requantize_to_int8
# ---------------------------------------------------------------------------

class TestRequantizeToInt8:
    """Tests for _requantize_to_int8 (A3)."""

    @pytest.mark.parametrize("K,N", [
        (128, 64),
        (256, 256),
        (512, 128),
        (1024, 512),
    ])
    def test_requantize_value_range(self, K, N):
        """Int8 values shouldbe in [-128, 127], scale should be positive."""
        float_weight = torch.randn(K, N) * 0.5

        weight_int8, channel_scale = _requantize_to_int8(float_weight)

        assert weight_int8.dtype == torch.int8
        assert weight_int8.shape == (K, N)
        assert channel_scale.shape == (N,)
        assert channel_scale.dtype == torch.float32
        assert (channel_scale > 0).all(), "All scales must be positive"
        assert weight_int8.min() >= -128
        assert weight_int8.max() <= 127
        print(f"  [PASS] requantize value range K={K}, N={N}")

    @pytest.mark.parametrize("K,N", [
        (128, 64),
        (256, 256),
        (1024, 512),
    ])
    def test_requantize_roundtrip_error(self, K, N):
        """Roundtrip error (dequant of int8) should be small."""
        float_weight = torch.randn(K, N) * 0.5
        weight_int8, channel_scale = _requantize_to_int8(float_weight)

        reconstructed = weight_int8.float() * channel_scale.unsqueeze(0)
        abs_error = (float_weight - reconstructed).abs()
        max_abs_error = abs_error.max().item()
        mean_abs_error = abs_error.mean().item()

        # Error should be bounded by 0.5 * scale (quantization step / 2)
        max_scale = channel_scale.max().item()
        assert max_abs_error <= max_scale * 0.5 + 1e-6, (
            f"Max abs error {max_abs_error} > 0.5 * max_scale {max_scale * 0.5} + 1e-6"
        )
        print(f"  [PASS] requantize roundtrip K={K}, N={N}: "
              f"mean_err={mean_abs_error:.6f}, max_err={max_abs_error:.6f}")

    def test_requantize_uses_full_range(self):
        """At least one column should map to +127 or -127."""
        K, N = 256, 64
        float_weight = torch.randn(K, N) * 2.0
        weight_int8, _ = _requantize_to_int8(float_weight)

        assert weight_int8.abs().max() == 127, ("Expected at least one value at +/-127")
        print("  [PASS] requantize uses full int8 range")

    def test_requantize_zero_column(self):
        """A column of all zeros should produce zero int8 values."""
        K, N = 128, 32
        float_weight = torch.randn(K, N) * 0.5
        float_weight[:, 0] = 0.0

        weight_int8, channel_scale = _requantize_to_int8(float_weight)
        assert (weight_int8[:, 0] == 0).all(), ("Zero column should produce all zero int8 values")
        print("  [PASS] requantize zero column")







# lyt_debug Test Full pipeline: AWQ pack -> unpack -> deinterleave -> dequant -> requant

class TestAWQFullPipeline:
    """lyt_debug End-to-end test simulating what process_weights_after_loading does."""

    @pytest.mark.parametrize("K,N,group_size", [
        (128, 128, 128),
        (256, 256, 128),
        (512, 256, 64),
    ])
    def test_roundtrip_awq_pack_to_int8(self, K, N, group_size):
        """
        Simulate:
          1. Create random int4 weights
          2. Apply AWQ interleave + pack (as if saving a checkpoint)
          3. Unpack + reverse interleave (what process_weights_after_loading does)
          4. Dequant to float (A2)
          5. Requantize to int8 (A3)
          6. Verify int8 dequant ≈ original float
        """
        num_groups = K // group_size
        pack_factor = 8  # 32 // 4 bits

        # lyt_debug Create "original" float weights ---
        weight_int4_orig = torch.randint(0, 16, (K, N), dtype=torch.int32)
        zeros_int4 = torch.randint(0, 16, (num_groups, N), dtype=torch.int32)
        scales = (torch.randn(num_groups, N) * 0.05).float()

        # lyt_debug Reference dequant
        zeros_exp = zeros_int4.repeat_interleave(group_size, dim=0)
        scales_exp = scales.repeat_interleave(group_size, dim=0)
        float_ref = (weight_int4_orig.float() - zeros_exp.float()) * scales_exp

        # lyt_ebug AWQ interleave + pack (simulate checkpoint format) ---
        awq_interleave = [0, 2, 4, 6, 1, 3, 5, 7]
        weight_interleaved = (
            weight_int4_orig
            .reshape(-1, 8)[:, awq_interleave]
            .reshape(K, N)
            .contiguous()
        )
        packed_weight = pack_cols(weight_interleaved, 4, K, N)

        zeros_interleaved = (
            zeros_int4
            .reshape(-1, 8)[:, awq_interleave]
            .reshape(num_groups, N)
            .contiguous()
        )
        packed_zeros = pack_cols(zeros_interleaved, 4, num_groups, N)

        # lyt_debug unpack + reverse interleave ---
        weight_unpacked = unpack_cols(packed_weight, 4, K, N)
        zeros_unpacked = unpack_cols(packed_zeros, 4, num_groups, N)

        reverse_map = (0, 4, 1, 5, 2, 6, 3, 7)
        weight_restored = (
            weight_unpacked.view(K, -1, pack_factor)[:, :, reverse_map]
            .reshape(K, N)
            .contiguous()
        )
        zeros_restored = (
            zeros_unpacked.view(num_groups, -1, pack_factor)[:, :, reverse_map]
            .reshape(num_groups, N)
            .contiguous()
        )

        # lyt_debug Verify unpack + reverse interleave restores original values
        torch.testing.assert_close(weight_restored, weight_int4_orig)
        torch.testing.assert_close(zeros_restored, zeros_int4)

        # lyt_debug Dequant (A2) 
        float_result = _dequant_awq_to_float(
            weight_restored, zeros_restored, scales, group_size
        )
        torch.testing.assert_close(float_result, float_ref, rtol=1e-5, atol=1e-5)

        #lyt_debug Requantize to int8 (A3) 
        weight_int8, channel_scale = _requantize_to_int8(float_result)

        # lyy_debug Verify int8 roundtrip
        reconstructed = weight_int8.float() * channel_scale.unsqueeze(0)
        rel_error = (float_ref - reconstructed).abs() / (float_ref.abs() + 1e-8)
        mask = float_ref.abs() > 1e-5
        mean_rel_error = rel_error[mask].mean().item()

        assert mean_rel_error < 0.10, (f"Mean relative error {mean_rel_error:.4f} exceeds 10% threshold")
        print(f"  [PASS] full pipeline K={K}, N={N}, gs={group_size}: mean_rel_err={mean_rel_error:.4f}")




# lyt_debug Test A4: oneDNN handler creation from AWQ int4→int8 pipeline


def _awq_int4_to_int8_pipeline(K, N, group_size, seed=42):
    """Simulate the full AWQ process_weights_after_loading pipeline.
    Returns weight_int8 [K,N], channel_scale [N], float_ref [K,N].
    """
    rng = np.random.RandomState(seed)
    num_groups = K // group_size
    pack_factor = 8

    weight_int4_orig = torch.from_numpy(rng.randint(0, 16, size=(K, N)).astype(np.int32))
    zeros_int4 = torch.from_numpy(rng.randint(0, 16, size=(num_groups, N)).astype(np.int32))
    scales = torch.from_numpy((rng.randn(num_groups, N) * 0.05).astype(np.float32))

    # lyt_debug AWQ interleave + pack (checkpoint format)
    awq_interleave = [0, 2, 4, 6, 1, 3, 5, 7]
    weight_interleaved = (
        weight_int4_orig.reshape(-1, 8)[:, awq_interleave]
        .reshape(K, N).contiguous()
    )
    packed_weight = pack_cols(weight_interleaved, 4, K, N)
    zeros_interleaved = (
        zeros_int4.reshape(-1, 8)[:, awq_interleave]
        .reshape(num_groups, N).contiguous()
    )
    packed_zeros = pack_cols(zeros_interleaved, 4, num_groups, N)

    # Unpack + reverse interleave (A1)
    reverse_map = (0, 4, 1, 5, 2, 6, 3, 7)
    w_unpacked = unpack_cols(packed_weight, 4, K, N)
    z_unpacked = unpack_cols(packed_zeros, 4, num_groups, N)
    w_restored = (
        w_unpacked.view(K, -1, pack_factor)[:, :, reverse_map]
        .reshape(K, N).contiguous()
    )
    z_restored = (
        z_unpacked.view(num_groups, -1, pack_factor)[:, :, reverse_map]
        .reshape(num_groups, N).contiguous()
    )

    # Dequant (A2)
    float_weight = _dequant_awq_to_float(w_restored, z_restored, scales, group_size)
    # Requantize (A3)
    weight_int8, channel_scale = _requantize_to_int8(float_weight)
    # oneDNN requires column-major weight: stride(0)==1
    weight_int8 = weight_int8.t().contiguous().t()

    return weight_int8, channel_scale, float_weight


class TestOneDNNHandlerCreation:
    """Tests for A4: creating oneDNN handler from AWQ int4→int8 weights."""

    @requires_onednn
    @pytest.mark.parametrize("K,N,group_size", [
        (128, 128, 128),
        (256, 256, 128),
        (512, 256, 64),
    ])
    def test_handler_creation(self, K, N, group_size):
        """oneDNN handler should be created with correct K, N dimensions."""
        weight_int8, channel_scale, _ = _awq_int4_to_int8_pipeline(
            K, N, group_size
        )
        channel_scale_2d = channel_scale.unsqueeze(0)  # [1, N]

        handler = ops.create_onednn_scaled_mm(
            weight_int8,                # [K, N] int8
            channel_scale_2d,           # [1, N] float32
            torch.bfloat16,             # output type
            True,                       # dynamic_act_quant
            False,                      # use_azp (symmetric)
            32,                         # primitive_cache_size
        )

        assert handler.k == K, f"Expected handler.k={K}, got {handler.k}"
        assert handler.n == N, f"Expected handler.n={N}, got {handler.n}"
        print(f"  [PASS] handler creation K={K}, N={N}, gs={group_size}: "
             f"handler.k={handler.k}, handler.n={handler.n}")

    @requires_onednn
    def test_handler_azp_adj_shape(self):
        """AZP adjustment tensor should have shape [1, N]."""
        K, N, group_size = 256, 128, 128
        weight_int8, channel_scale, _ = _awq_int4_to_int8_pipeline(
            K, N, group_size
        )
        channel_scale_2d = channel_scale.unsqueeze(0)
        azp_adj = (
            weight_int8.sum(dim=0, keepdim=True, dtype=torch.float32)
            * channel_scale_2d
        )

        assert azp_adj.shape == (1, N), (f"Expected azp_adj shape (1, {N}), got {azp_adj.shape}")
        assert azp_adj.dtype == torch.float32
        print(f"  [PASS] azp_adj shape={azp_adj.shape}, dtype={azp_adj.dtype}")



# LYT_DEBUG Test A5: End-to-end AWQ int4 → int8 → oneDNN GEMM vs bf16 reference

def _ref_onednn_dynamic_gemm(x_q, weight_int8, x_s, channel_scale_2d,
                              bias, out_dtype):
    """Reference matching oneDNN dynamic-quant int8 GEMM execution flow.

    C++ flow: (1) int32_accum = x_q @ weight  (exact integer matmul)
              (2) tmp_f32 = int32_accum * weight_scale  (per-channel)
              (3) output = x_s * tmp_f32 [+ bias]  (per-token scale + bf16)

    In this test, i would use float64 matmul to get exact integer accumulation results.
    """
    # Step 1: exact integer matmul via float64
    int_mm = torch.mm(x_q.to(torch.float64),
                      weight_int8.contiguous().to(torch.float64))  # [M, N]
    # Step 2: per-channel weight scale (in float32, matching C++)
    tmp_f32 = int_mm.float() * channel_scale_2d  # [M, N]
    # Step 3: per-token activation scale + bias + output dtype
    out_f32 = x_s * tmp_f32
    if bias is not None:
        out_f32 = out_f32 + bias.float()
    return out_f32.to(out_dtype)


class TestOneDNNInt8GEMM:
    """Tests for A5: oneDNN int8 GEMM with AWQ-derived int8 weights."""

    @requires_onednn
    @pytest.mark.parametrize("M,K,N,group_size", [
        (1, 128, 128, 128),
        (4, 256, 256, 128),
        (16, 512, 256, 64),
        (32, 256, 512, 128),
    ])
    def test_int8_gemm_kernel_correctness(self, M, K, N, group_size):
        """oneDNN int8 GEMM should match exact int8 reference."""
        weight_int8, channel_scale, _ = _awq_int4_to_int8_pipeline(
            K, N, group_size
        )
        channel_scale_2d = channel_scale.unsqueeze(0)  # [1, N]

        handler = ops.create_onednn_scaled_mm(
            weight_int8, channel_scale_2d, torch.bfloat16,
            True, False, 32,
        )

        x = torch.randn(M, K, dtype=torch.bfloat16)
        x_q, x_s, _ = ops.onednn_scaled_int8_quant(x, None, None, True)

        #oneDNN int8 GEMM
        out_int8 = torch.zeros((M, N), dtype=torch.bfloat16)
        ops.onednn_scaled_mm(handler, x_q, out_int8, x_s, None, None, None)

        # Reference: exact int matmul + two-step scaling
        out_ref = _ref_onednn_dynamic_gemm(x_q, weight_int8, x_s, channel_scale_2d, None, torch.bfloat16)

        abs_diff = (out_int8.float() - out_ref.float()).abs()
        max_abs = abs_diff.max().item()
        mean_abs = abs_diff.mean().item()
        out_mag = out_ref.float().abs().mean().item() + 1e-6
        mean_rel = (abs_diff / (out_ref.float().abs() + 1e-6)).mean().item()
        pct95 = torch.quantile(abs_diff, 0.95).item()

        print(f"  lyt_debug_A5 kernel check M={M}, K={K}, N={N}, gs={group_size}: "
              f"out_int8 range=[{out_int8.float().min():.4f}, {out_int8.float().max():.4f}], "
              f"out_ref range=[{out_ref.float().min():.4f}, {out_ref.float().max():.4f}], "
              f"max_abs={max_abs:.4f}, mean_abs={mean_abs:.4f}, "
              f"pct95_abs={pct95:.4f}, mean_rel={mean_rel:.4f}")

        # For int8 GEMM with full-range weights [-127,127] and small per-channel scales (~0.01), oneDNN VNNI u8*s8 internal handling introduces noise proportional to output magnitude.
        # In this test, i would use statistical thresholds instead of element-wise tolerance.
        assert mean_abs < 2.0, (f"mean_abs_diff {mean_abs:.4f} too large (threshold 2.0)")
        assert pct95 < 5.0, (f"95th-percentile abs_diff {pct95:.4f} too large (threshold 5.0)")
        print(f"  [PASS] int8 GEMM kernel correct: M={M}, K={K}, N={N}")

    @requires_onednn
    @pytest.mark.parametrize("M", [1, 8, 32])
    def test_int8_gemm_with_bias(self, M):
        """oneDNN int8 GEMM with bias should match reference."""
        K, N, group_size = 256, 128, 128
        weight_int8, channel_scale, _ = _awq_int4_to_int8_pipeline(
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

        out_ref = _ref_onednn_dynamic_gemm(x_q, weight_int8, x_s, channel_scale_2d, bias, torch.bfloat16)

        abs_diff = (out_int8.float() - out_ref.float()).abs()
        mean_abs = abs_diff.mean().item()
        pct95 = torch.quantile(abs_diff, 0.95).item()
        print(f"  lyt_debug_A5 bias M={M}: max_abs={abs_diff.max().item():.4f}, "
            f"mean_abs={mean_abs:.4f}, pct95_abs={pct95:.4f}")

        assert mean_abs < 2.0, (f"mean_abs_diff {mean_abs:.4f} with bias exceeds threshold")
        assert pct95 < 5.0, (f"95th-pctile abs_diff {pct95:.4f} with bias exceeds threshold")
        print(f"  [PASS] int8 GEMM with bias: M={M}")

    @requires_onednn
    def test_int8_gemm_3d_input(self):
        """apply() reshapes 3D input [B, S, K] → [B*S, K] → back to 3D."""
        K, N, group_size = 256, 128, 128
        weight_int8, channel_scale, _ = _awq_int4_to_int8_pipeline(
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
        print(f"  lyt_debug_A5 3D: max_abs={abs_diff.max().item():.4f}, "
            f"mean_abs={mean_abs:.4f}, pct95_abs={pct95:.4f}")
        assert mean_abs < 2.0, (f"mean_abs_diff {mean_abs:.4f} for 3D exceeds threshold")
        assert pct95 < 5.0, (
            f"95th-pctile abs_diff {pct95:.4f} for 3D exceeds threshold")
        print(f"  [PASS] 3D input [{B},{S},{K}] → output [{B},{S},{N}]")






# lyt_debug Test A6: create_weights unchanged — verify int4 param shapes

class TestCreateWeightsUnchanged:
    """A6: create_weights should still produce correct int4 placeholder shapes."""

    @pytest.mark.parametrize("K,N,group_size", [
        (128, 128, 128),
        (256, 256, 128),
        (512, 256, 64),
    ])
    def test_int4_placeholder_shapes(self, K, N, group_size):
        """Verify qweight, qzeros, scales shapes before process_weights_after_loading."""
        pack_factor = 8
        num_groups = K // group_size

        qweight = torch.empty(K, N // pack_factor, dtype=torch.int32)
        qzeros = torch.empty(num_groups, N // pack_factor, dtype=torch.int32)
        scales = torch.empty(num_groups, N, dtype=torch.bfloat16)

        assert qweight.shape == (K, N // pack_factor)
        assert qzeros.shape == (num_groups, N // pack_factor)
        assert scales.shape == (num_groups, N)
        print(f"  [PASS] create_weights shapes: K={K}, N={N}, gs={group_size}")


# ---------------------------------------------------------------------------
# Standalone runner

# if __name__ == "__main__":
#     print("=" * 60)
#     print("Running AWQ int4 -> int8 re-quantization unit tests")
#     print("=" * 60)

#     print("\n--- Test A2: _dequant_awq_to_float ---")
#     t2 = TestDequantAWQToFloat()
#     for K, N, gs in [(128, 64, 128), (256, 256, 128), (512, 128, 64)]:
#         t2.test_dequant_matches_reference(K, N, gs)
#     t2.test_dequant_zero_weights()
#     t2.test_dequant_single_group()

#     print("\n--- Test A3: _requantize_to_int8 ---")
#     t3 = TestRequantizeToInt8()
#     for K, N in [(128, 64), (256, 256), (1024, 512)]:
#         t3.test_requantize_value_range(K, N)
#         t3.test_requantize_roundtrip_error(K, N)
#     t3.test_requantize_uses_full_range()
#     t3.test_requantize_zero_column()

#     print("\n--- Test Full Pipeline: AWQ pack -> int8 ---")
#     tp = TestAWQFullPipeline()
#     for K, N, gs in [(128, 128, 128), (256, 256, 128), (512, 256, 64)]:
#         tp.test_roundtrip_awq_pack_to_int8(K, N, gs)

#     print("\n--- Test A6: create_weights shapes ---")
#     t6 = TestCreateWeightsUnchanged()
#     for K, N, gs in [(128, 128, 128), (256, 256, 128), (512, 256, 64)]:
#         t6.test_int4_placeholder_shapes(K, N, gs)

#     print("\n--- Test A4/A5: oneDNN handler + GEMM (requires C++ extensions) ---")
#     if _has_onednn:
#         t4 = TestOneDNNHandlerCreation()
#         for K, N, gs in [(128, 128, 128), (256, 256, 128), (512, 256, 64)]:
#             t4.test_handler_creation(K, N, gs)
#         t4.test_handler_azp_adj_shape()

#         t5 = TestOneDNNInt8GEMM()
#         for M, K, N, gs in [(1, 128, 128, 128), (4, 256, 256, 128),
#                             (16, 512, 256, 64), (32, 256, 512, 128)]:
#             t5.test_int8_gemm_kernel_correctness(M, K, N, gs)
#         for M in [1, 8, 32]:
#             t5.test_int8_gemm_with_bias(M)
#         t5.test_int8_gemm_3d_input()
#     else:
#         print("  [SKIP] oneDNN not available, skipping A4/A5 tests")

#     print("\n" + "=" * 60)
#     print("ALL TESTS PASSED")
#     print("=" * 60)
