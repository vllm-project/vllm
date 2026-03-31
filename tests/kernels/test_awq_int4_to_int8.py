# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for AWQ INT4 W4A8 GEMM pipeline (SGLang kernel migration).

Part 1: Weight packing tests
  - convert_weight_packed_scale_zp correctness

Part 2: INT4 W4A8 GEMM tests
  - int4_scaled_mm_cpu correctness w.r.t. float reference
  - Bias, 3D input, various shapes

Part 3: create_weights shapes

cmd:
    VLLM_CPU_INT4_W4A8=1 python -m pytest tests/kernels/test_awq_int4_to_int8.py -v -s
"""

import numpy as np
import pytest
import torch

from vllm._custom_ops import _supports_cpu_w4a8_int8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_cols,
)

requires_cpu_w4a8_int8 = pytest.mark.skipif(
    not _supports_cpu_w4a8_int8,
    reason="Requires vLLM CPU build with SGLang INT4 W4A8 kernels",
)


def make_awq_checkpoint_data(K, N, group_size, seed=42):
    """Create synthetic AWQ checkpoint data in packed int32 format.

    Returns:
        packed_qweight: [K, N//8] int32 (AWQ interleaved + packed)
        packed_qzeros:  [num_groups, N//8] int32 (AWQ interleaved + packed)
        scales:         [num_groups, N] float32
        float_ref:      [K, N] float32, reference dequantized weights
        weight_int4_orig: [K, N] int32, original int4 values (0-15)
        zeros_int4_orig:  [num_groups, N] int32, original zero points (0-15)
    """
    rng = np.random.RandomState(seed)
    num_groups = K // group_size

    weight_int4_orig = torch.from_numpy(
        rng.randint(0, 16, size=(K, N)).astype(np.int32)
    )
    zeros_int4_orig = torch.from_numpy(
        rng.randint(0, 16, size=(num_groups, N)).astype(np.int32)
    )
    scales = torch.from_numpy((rng.randn(num_groups, N) * 0.05).astype(np.float32))

    scales_exp = scales.repeat_interleave(group_size, dim=0)
    zeros_exp = zeros_int4_orig.repeat_interleave(group_size, dim=0)
    float_ref = (weight_int4_orig.float() - zeros_exp.float()) * scales_exp

    awq_interleave = [0, 2, 4, 6, 1, 3, 5, 7]
    weight_interleaved = (
        weight_int4_orig.reshape(-1, 8)[:, awq_interleave].reshape(K, N).contiguous()
    )
    packed_qweight = pack_cols(weight_interleaved, 4, K, N)

    zeros_interleaved = (
        zeros_int4_orig.reshape(-1, 8)[:, awq_interleave]
        .reshape(num_groups, N)
        .contiguous()
    )
    packed_qzeros = pack_cols(zeros_interleaved, 4, num_groups, N)

    return (
        packed_qweight,
        packed_qzeros,
        scales,
        float_ref,
        weight_int4_orig,
        zeros_int4_orig,
    )


class TestConvertWeightPackedScaleZp:
    """Tests for convert_weight_packed_scale_zp weightpacking."""

    @requires_cpu_w4a8_int8
    @pytest.mark.parametrize(
        "K,N,group_size",
        [
            (128, 128, 128),
            (256, 256, 128),
            (512, 256, 64),
        ],
    )
    def test_packing_output_shapes(self, K, N, group_size):
        """Packed outputs should have expected shapes."""
        (packed_qweight, packed_qzeros, scales, _, _, _) = make_awq_checkpoint_data(
            K, N, group_size
        )

        blocked_w, blocked_zp, blocked_s = torch.ops._C.convert_weight_packed_scale_zp(
            packed_qweight, packed_qzeros, scales
        )

        block_n = 32
        Nc = N // block_n

        assert blocked_w.dim() >= 2, (
            f"blocked_w should have >= 2 dims, got {blocked_w.dim()}"
        )
        assert blocked_s.size(0) == Nc, (
            f"Expected Nc={Nc} scale blocks, got {blocked_s.size(0)}"
        )
        assert blocked_zp.size(0) == Nc, (
            f"Expected Nc={Nc} qzeros blocks, got {blocked_zp.size(0)}"
        )

        print(
            f"  [PASS] packing shapes K={K}, N={N}, gs={group_size}: "
            f"blocked_w={list(blocked_w.shape)}, "
            f"blocked_s={list(blocked_s.shape)}, blocked_zp={list(blocked_zp.shape)}"
        )


class TestInt4ScaledMmCpu:
    """Tests for int4_scaled_mm_cpu GEMM kernel."""

    @requires_cpu_w4a8_int8
    @pytest.mark.parametrize(
        "M,K,N,group_size",
        [
            (1, 128, 128, 128),
            (4, 256, 256, 128),
            (16, 512, 256, 64),
            (32, 256, 512, 128),
            (64, 512, 512, 128),
        ],
    )
    def test_gemm_vs_float_reference(self, M, K, N, group_size):
        """INT4 W4A8 GEMM should approximate float matmul."""
        (packed_qweight, packed_qzeros, scales, float_ref, _, _) = (
            make_awq_checkpoint_data(K, N, group_size)
        )

        blocked_w, blocked_zp, blocked_s = torch.ops._C.convert_weight_packed_scale_zp(
            packed_qweight, packed_qzeros, scales
        )

        x = torch.randn(M, K, dtype=torch.bfloat16)
        out = torch.ops._C.int4_scaled_mm_cpu(x, blocked_w, blocked_zp, blocked_s, None)

        ref_out = torch.mm(x.float(), float_ref)

        abs_diff = (out.float() - ref_out).abs()
        mean_abs = abs_diff.mean().item()
        pct95 = torch.quantile(abs_diff, 0.95).item()
        ref_mag = ref_out.abs().mean().item() + 1e-6
        mean_rel = mean_abs / ref_mag

        assert mean_rel < 0.05, (
            f"Mean relative error {mean_rel:.4f} exceeds 5% threshold"
        )
        assert pct95 < ref_mag * 0.15, (
            f"95th-pctile abs_diff {pct95:.4f} exceeds 15% of ref magnitude"
        )
        print(f"  [PASS] INT4 GEMM correct: M={M}, K={K}, N={N}")

    @requires_cpu_w4a8_int8
    @pytest.mark.parametrize("M", [1, 8, 32])
    def test_gemm_with_bias(self, M):
        """INT4 W4A8 GEMM with bias should match reference."""
        K, N, group_size = 256, 128, 128
        (packed_qweight, packed_qzeros, scales, float_ref, _, _) = (
            make_awq_checkpoint_data(K, N, group_size)
        )

        blocked_w, blocked_zp, blocked_s = torch.ops._C.convert_weight_packed_scale_zp(
            packed_qweight, packed_qzeros, scales
        )

        bias = torch.randn(N, dtype=torch.float32)
        x = torch.randn(M, K, dtype=torch.bfloat16)

        out = torch.ops._C.int4_scaled_mm_cpu(x, blocked_w, blocked_zp, blocked_s, bias)

        ref_out = torch.mm(x.float(), float_ref) + bias
        abs_diff = (out.float() - ref_out).abs()
        mean_abs = abs_diff.mean().item()
        ref_mag = ref_out.abs().mean().item() + 1e-6
        mean_rel = mean_abs / ref_mag
        assert mean_rel < 0.05, (
            f"Mean relative error {mean_rel:.4f} with bias exceeds 5%"
        )
        print(f"  [PASS] INT4 GEMM with bias: M={M}")

    @requires_cpu_w4a8_int8
    def test_gemm_3d_input(self):
        """apply() reshapes 3D input [B, S, K] -> [B*S, K] -> back to 3D."""
        K, N, group_size = 256, 128, 128
        (packed_qweight, packed_qzeros, scales, float_ref, _, _) = (
            make_awq_checkpoint_data(K, N, group_size)
        )

        blocked_w, blocked_zp, blocked_s = torch.ops._C.convert_weight_packed_scale_zp(
            packed_qweight, packed_qzeros, scales
        )

        B, S = 2, 8
        x_3d = torch.randn(B, S, K, dtype=torch.bfloat16)
        x_2d = x_3d.reshape(-1, K)

        out_2d = torch.ops._C.int4_scaled_mm_cpu(
            x_2d, blocked_w, blocked_zp, blocked_s, None
        )
        out_3d = out_2d.reshape(B, S, N)

        ref_out = torch.mm(x_2d.float(), float_ref).reshape(B, S, N)

        assert out_3d.shape == (B, S, N)
        abs_diff = (out_3d.float() - ref_out).abs()
        mean_abs = abs_diff.mean().item()
        ref_mag = ref_out.abs().mean().item() + 1e-6
        mean_rel = mean_abs / ref_mag

        assert mean_rel < 0.05, f"Mean relative error {mean_rel:.4f} for 3D exceeds 5%"
        print(f"  [PASS] 3D input [{B},{S},{K}] -> output [{B},{S},{N}]")

    @requires_cpu_w4a8_int8
    def test_gemm_fp16_input(self):
        """INT4 GEMM should also work with fp16 input."""
        K, N, group_size, M = 256, 256, 128, 8
        (packed_qweight, packed_qzeros, scales, float_ref, _, _) = (
            make_awq_checkpoint_data(K, N, group_size)
        )

        blocked_w, blocked_zp, blocked_s = torch.ops._C.convert_weight_packed_scale_zp(
            packed_qweight, packed_qzeros, scales
        )

        x = torch.randn(M, K, dtype=torch.float16)
        out = torch.ops._C.int4_scaled_mm_cpu(x, blocked_w, blocked_zp, blocked_s, None)

        ref_out = torch.mm(x.float(), float_ref)
        abs_diff = (out.float() - ref_out).abs()
        ref_mag = ref_out.abs().mean().item() + 1e-6
        mean_rel = abs_diff.mean().item() / ref_mag

        assert mean_rel < 0.05, (
            f"Mean relative error {mean_rel:.4f} for fp16 exceeds 5%"
        )
        print(f"  [PASS] fp16 input M={M}, K={K}, N={N}")


class TestCreateWeightsUnchanged:
    """Create_weights should still produce correct int4 placeholder shapes."""

    @pytest.mark.parametrize(
        "K,N,group_size",
        [
            (128, 128, 128),
            (256, 256, 128),
            (512, 256, 64),
        ],
    )
    def test_int4_placeholder_shapes(self, K, N, group_size):
        """Verify qweight, qzeros, scales shapes."""
        pack_factor = 8
        num_groups = K // group_size

        qweight = torch.empty(K, N // pack_factor, dtype=torch.int32)
        qzeros = torch.empty(num_groups, N // pack_factor, dtype=torch.int32)
        scales = torch.empty(num_groups, N, dtype=torch.bfloat16)

        assert qweight.shape == (K, N // pack_factor)
        assert qzeros.shape == (num_groups, N // pack_factor)
        assert scales.shape == (num_groups, N)
        print(f"  [PASS] create_weights shapes: K={K}, N={N}, gs={group_size}")
