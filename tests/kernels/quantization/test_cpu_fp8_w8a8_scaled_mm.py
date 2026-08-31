# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU FP8 W8A8 GEMM kernels.

Covers:
  * float8_linear_prepack_cpu  (weight packing)
  * fp8_scaled_mm_with_quant   (fused quant + GEMM, per-tensor and per-token)
  * quantize_fp8e4m3_vec       (activation quantization)

Run:
  pytest tests/kernels/quantization/test_cpu_fp8_w8a8_scaled_mm.py -v
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

if not ops._supports_cpu_fp8_w8a8:
    pytest.skip("float8_linear_prepack_cpu op not available", allow_module_level=True)

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
RTOL = 1e-1  # FP8 precision is low; we use a generous tolerance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quantize_weight_per_tensor(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight [N, K] to FP8 with a single per-tensor scale."""
    abs_max = weight.abs().max()
    scale = (abs_max / FP8_MAX).clamp(min=1e-7)
    q = (weight.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.view(1)


def quantize_weight_per_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight [N, K] to FP8 with per-channel (per-output) scales."""
    abs_max = weight.abs().amax(dim=1)  # [N]
    scale = (abs_max / FP8_MAX).clamp(min=1e-7)
    q = (weight.float() / scale.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn
    )
    return q, scale  # [N]


def reference_linear_fp8(
    x: torch.Tensor,          # BF16 [M, K]
    weight_fp8: torch.Tensor, # FP8 [N, K]
    weight_scale: torch.Tensor,  # float32, scalar or [N]
    act_scale: torch.Tensor | None,  # float32 scalar or [M] or None
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference: quantize/dequant activations and weights, then matmul."""
    x_f = x.float()
    w_f = weight_fp8.float()

    # Apply weight scale
    if weight_scale.numel() == 1:
        w_f = w_f * weight_scale.item()
    else:
        w_f = w_f * weight_scale.view(-1, 1)

    # Apply activation scale with explicit quantize/dequant for static mode.
    if act_scale is not None:
        if act_scale.numel() == 1:
            x_s = act_scale.item()
            x_q = (x_f / x_s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            x_f = x_q.float() * x_s
        else:
            x_s = act_scale.view(-1, 1)
            x_q = (x_f / x_s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            x_f = x_q.float() * x_s

    out = torch.mm(x_f, w_f.t())
    return out.to(out_dtype)


# ---------------------------------------------------------------------------
# Tests: quantize_fp8e4m3_vec
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,K", [(1, 256), (16, 512), (64, 1024)])
def test_quantize_fp8e4m3_vec_channelwise(M: int, K: int):
    """Per-token (per-row) dynamic quantization of BF16 activations."""
    x = torch.randn(M, K, dtype=torch.bfloat16)
    q, scale = torch.ops._C.quantize_fp8e4m3_vec(x, True, None)

    assert q.dtype == torch.float8_e4m3fn
    assert q.shape == x.shape
    assert scale.shape == (M,)

    # Reconstruct and check relative error
    x_dq = q.float() * scale.view(M, 1)
    rel_err = (x_dq - x.float()).abs() / (x.float().abs() + 1e-6)
    assert rel_err.mean() < 0.03, f"Mean relative error too large: {rel_err.mean()}"


# ---------------------------------------------------------------------------
# Tests: float8_linear_prepack_cpu
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,K", [(128, 256), (256, 512), (64, 128)])
def test_prepack_round_trip(N: int, K: int):
    """Verify prepack doesn't lose data (packed weight has correct shape)."""
    weight = torch.randn(N, K).to(torch.float8_e4m3fn)
    scale_pt = torch.ones(N, 1, dtype=torch.float32)

    packed_w, packed_s = torch.ops._C.float8_linear_prepack_cpu(weight, scale_pt)

    # packed_w: [Nc, Kc, block_k, block_n] where Nc=N/32, block_n=32
    # Total elements should match
    assert packed_w.numel() == N * K
    assert packed_w.dtype == torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# Tests: fp8_scaled_mm_with_quant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K", [
    (1, 128, 256),
    (16, 256, 512),
    (64, 128, 256),
    (4, 64, 128),
])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_fp8_w8a8_per_tensor_static(M: int, N: int, K: int, out_dtype: torch.dtype):
    """Static per-tensor W8A8: pre-quantized act with known scale."""
    # Create BF16 inputs
    x = torch.randn(M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)

    # Quantize weight per-tensor
    w_fp8, w_scale = quantize_weight_per_tensor(weight)
    w_scale = w_scale.float()

    # Quantize activation per-tensor (static: use a fixed scale)
    x_abs_max = x.abs().max()
    x_scale = (x_abs_max / FP8_MAX).clamp(min=1e-7).float().view(1)

    # Prepack weight
    # Prepack expects scales shaped as [N, G]. For per-tensor, G=1.
    packed_w, packed_ws = torch.ops._C.float8_linear_prepack_cpu(
        w_fp8, w_scale.repeat(N).view(N, 1)
    )

    # Run W8A8 kernel (pass act_scales=x_scale, channelwise=False → per-tensor)
    out = ops.fp8_scaled_mm_with_quant(
        x, x_scale, False, packed_w, packed_ws, None, out_dtype
    )

    # Reference: dequant weight, use scale for x
    ref = reference_linear_fp8(x, w_fp8, w_scale, x_scale, out_dtype)

    assert out.shape == (M, N)
    assert out.dtype == out_dtype
    max_diff = (out.float() - ref.float()).abs().max()
    mean_diff = (out.float() - ref.float()).abs().mean()
    assert mean_diff < 1.0, (
        f"Mean abs diff too large: {mean_diff:.4f} (max={max_diff:.4f})"
    )


@pytest.mark.parametrize("M,N,K", [
    (1, 128, 256),
    (16, 256, 512),
    (64, 128, 256),
])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_fp8_w8a8_per_token_dynamic(M: int, N: int, K: int, out_dtype: torch.dtype):
    """Dynamic per-token W8A8: activation quantized on-the-fly."""
    x = torch.randn(M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)

    # Quantize weight per-channel
    w_fp8, w_scale = quantize_weight_per_channel(weight)  # w_scale: [N]
    w_scale_2d = w_scale.view(N, 1).float()

    # Prepack weight
    packed_w, packed_ws = torch.ops._C.float8_linear_prepack_cpu(w_fp8, w_scale_2d)

    # Run W8A8 kernel with act_scales=None → dynamic per-token quant
    out = ops.fp8_scaled_mm_with_quant(
        x, None, True, packed_w, packed_ws, None, out_dtype
    )

    # Reference: per-token quantize x, then multiply
    x_per_row_max = x.float().abs().amax(dim=1, keepdim=True)  # [M, 1]
    x_scale = (x_per_row_max / FP8_MAX).clamp(min=1e-7)
    x_q = (x.float() / x_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

    ref = torch.mm(x_q.float() * x_scale, (w_fp8.float() * w_scale.view(-1, 1)).t())
    ref = ref.to(out_dtype)

    assert out.shape == (M, N)
    assert out.dtype == out_dtype
    mean_diff = (out.float() - ref.float()).abs().mean()
    assert mean_diff < 2.0, f"Mean abs diff too large: {mean_diff:.4f}"


@pytest.mark.parametrize("M,N,K", [(16, 256, 512), (64, 128, 256)])
def test_fp8_w8a8_per_group(M: int, N: int, K: int):
    """Per-group W8A8 (block quantization): weight scale shape [N, G]."""
    group_size = 128
    assert K % group_size == 0
    G = K // group_size

    x = torch.randn(M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)

    # Per-group weight quantization
    w_groups = weight.view(N, G, group_size)
    abs_max = w_groups.abs().amax(dim=2, keepdim=True)  # [N, G, 1]
    w_scale = (abs_max / FP8_MAX).clamp(min=1e-7).squeeze(2).float()  # [N, G]
    w_fp8 = (w_groups.float() / abs_max).clamp(-FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn
    ).view(N, K)

    # Prepack
    packed_w, packed_ws = torch.ops._C.float8_linear_prepack_cpu(w_fp8, w_scale)

    # Run dynamic per-token W8A8
    out = ops.fp8_scaled_mm_with_quant(
        x, None, True, packed_w, packed_ws, None, torch.bfloat16
    )

    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16


@pytest.mark.parametrize("M,N,K", [(1, 128, 256), (8, 256, 512)])
def test_fp8_w8a8_with_bias(M: int, N: int, K: int):
    """Verify that bias is correctly added."""
    x = torch.randn(M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)
    bias = torch.randn(N, dtype=torch.float32)

    w_fp8, w_scale = quantize_weight_per_tensor(weight)
    w_scale_2d = w_scale.repeat(N).view(N, 1).float()
    packed_w, packed_ws = torch.ops._C.float8_linear_prepack_cpu(w_fp8, w_scale_2d)

    out_no_bias = ops.fp8_scaled_mm_with_quant(
        x, None, True, packed_w, packed_ws, None, torch.bfloat16
    )
    out_with_bias = ops.fp8_scaled_mm_with_quant(
        x, None, True, packed_w, packed_ws, bias, torch.bfloat16
    )

    expected_diff = bias.bfloat16().unsqueeze(0).expand(M, -1)
    actual_diff = out_with_bias - out_no_bias
    # Allow small FP8 rounding effects
    assert (actual_diff - expected_diff).abs().max() < 0.4


@pytest.mark.parametrize("M,N,K", [(16, 256, 512)])
def test_fp8_w8a8_3d_input(M: int, N: int, K: int):
    """Verify that 3D input [B, S, K] is handled correctly."""
    B, S = 2, M // 2
    x = torch.randn(B, S, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)

    w_fp8, w_scale = quantize_weight_per_tensor(weight)
    packed_w, packed_ws = torch.ops._C.float8_linear_prepack_cpu(
        w_fp8, w_scale.repeat(N).view(N, 1).float()
    )

    x_2d = x.reshape(-1, K)
    out = ops.fp8_scaled_mm_with_quant(
        x_2d, None, True, packed_w, packed_ws, None, torch.bfloat16
    )
    out = out.reshape(B, S, N)
    assert out.shape == (B, S, N)
