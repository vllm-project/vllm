# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU FP8 scaled-mm GEMM kernels.

Covers:
  * fp8_scaled_mm_cpu (W8A16)
  * fp8_scaled_mm_with_quant (W8A8)

Run `pytest tests/kernels/quantization/test_cpu_fp8_scaled_mm.py -v`.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

requires_cpu_fp8_w8a16 = pytest.mark.skipif(
    not ops._supports_cpu_fp8_w8a16, reason="fp8_scaled_mm_cpu op not available"
)
requires_cpu_fp8_w8a8 = pytest.mark.skipif(
    not ops._supports_cpu_fp8_w8a8,
    reason="float8_linear_prepack_cpu op not available",
)

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
BLOCK_SIZE = [128, 128]
GROUP_SIZE = 128


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


# ---------------------------------------------------------------------------
# W8A16 helpers
# ---------------------------------------------------------------------------


def quantize_weight_block_fp8(
    weight: torch.Tensor,
    block_size: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight [N, K] to FP8 with block scales.

    Returns:
        fp8_weight: [N, K] float8_e4m3fn
        scales: [n_tiles, k_tiles] float32
    """
    N, K = weight.shape
    block_n, block_k = block_size
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    n_tiles = cdiv(N, block_n)
    k_tiles = cdiv(K, block_k)

    # Pad for even blocking
    pad_N = (block_n - (N % block_n)) % block_n
    pad_K = (block_k - (K % block_k)) % block_k
    if pad_N > 0 or pad_K > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_K, 0, pad_N))

    # Reshape into blocks
    w_blocks = weight.view(n_tiles, block_n, k_tiles, block_k)
    w_blocks = w_blocks.permute(0, 2, 1, 3).contiguous()

    # Per-block scale
    abs_max = w_blocks.abs().amax(dim=(-2, -1), keepdim=True)
    scales = abs_max / fp8_max
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)

    # Quantize
    q_fp8 = (w_blocks / scales).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    # Reshape back
    fp8_weight = (
        q_fp8.permute(0, 2, 1, 3)
        .contiguous()
        .view(N + pad_N, K + pad_K)[:N, :K]
        .contiguous()
    )

    scales = scales.view(n_tiles, k_tiles)
    return fp8_weight, scales


def dequant_weight_block_fp8(
    fp8_weight: torch.Tensor,
    scales: torch.Tensor,
    block_size: list[int],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize FP8 weight back to float for reference computation."""
    N, K = fp8_weight.shape
    block_n, block_k = block_size
    n_tiles, k_tiles = scales.shape

    pad_N = (block_n - (N % block_n)) % block_n
    pad_K = (block_k - (K % block_k)) % block_k
    if pad_N > 0 or pad_K > 0:
        fp8_padded = torch.nn.functional.pad(fp8_weight.float(), (0, pad_K, 0, pad_N))
    else:
        fp8_padded = fp8_weight.float()

    w_blocks = fp8_padded.view(n_tiles, block_n, k_tiles, block_k)
    w_blocks = w_blocks.permute(0, 2, 1, 3).contiguous()
    dq = w_blocks * scales.view(n_tiles, k_tiles, 1, 1)
    dq = dq.permute(0, 2, 1, 3).contiguous().view(N + pad_N, K + pad_K)
    return dq[:N, :K].to(out_dtype)


def ref_fp8_block_scaled_mm(
    x: torch.Tensor,
    fp8_weight: torch.Tensor,
    scales: torch.Tensor,
    block_size: list[int],
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference: dequant FP8→float32, matmul in float32, cast to out_dtype."""
    w_dq = dequant_weight_block_fp8(fp8_weight, scales, block_size, torch.float32)
    out = torch.mm(x.float(), w_dq.t())
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


# ---------------------------------------------------------------------------
# W8A16 test: fp8_scaled_mm_cpu
# ---------------------------------------------------------------------------
M_SIZES = [1, 4, 16, 64, 128]
# (N, K) — weight shape is [N, K], output has N columns.
NK_SIZES = [
    (128, 256),
    (256, 512),
    (512, 1024),
    (1024, 2048),
    (5120, 5120),
    (17408, 5120),
    (5120, 17408),
]


@requires_cpu_fp8_w8a16
@pytest.mark.parametrize("M", M_SIZES)
@pytest.mark.parametrize("N,K", NK_SIZES)
@pytest.mark.parametrize("use_bias", [False, True])
def test_cpu_fp8_scaled_mm(M: int, N: int, K: int, use_bias: bool):
    """fp8_scaled_mm_cpu correctness against float reference."""
    torch.manual_seed(42)
    out_dtype = torch.bfloat16
    block_size = BLOCK_SIZE

    x = torch.randn(M, K, dtype=out_dtype) / (K**0.5)
    w_f32 = torch.randn(N, K, dtype=torch.float32) / (K**0.5)
    fp8_weight, scales = quantize_weight_block_fp8(w_f32, block_size)

    bias = torch.randn(N, dtype=torch.float32) * 0.1 if use_bias else None

    ref_out = ref_fp8_block_scaled_mm(
        x, fp8_weight, scales, block_size, bias, out_dtype
    )

    packed_weight = torch.ops._C.convert_weight_packed(fp8_weight)
    kernel_out = ops.fp8_scaled_mm_cpu(
        x,
        packed_weight,
        scales,
        block_size,
        bias,
        out_dtype,
        True,
    )

    assert kernel_out.dtype == out_dtype
    torch.testing.assert_close(kernel_out, ref_out, rtol=0.02, atol=0.01)


# ---------------------------------------------------------------------------
# W8A8 helpers
# ---------------------------------------------------------------------------


def quantize_weight_per_tensor(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight [N, K] to FP8 with a single per-tensor scale, broadcast
    to [N, 1] (the shape float8_linear_prepack_cpu expects)."""
    N = weight.shape[0]
    abs_max = weight.abs().max()
    scale = (abs_max / FP8_MAX).clamp(min=1e-7)
    q = (weight.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.expand(N).contiguous().view(N, 1)


def quantize_weight_per_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight [N, K] to FP8 with per-channel (per-row) scales [N, 1]."""
    abs_max = weight.abs().amax(dim=1, keepdim=True)
    scale = (abs_max / FP8_MAX).clamp(min=1e-7)
    q = (weight.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.float()


def quantize_weight_per_group(
    weight: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight [N, K] to FP8 with per-group (128-block) scales [N, G]."""
    N, K = weight.shape
    G = K // group_size
    w_groups = weight.view(N, G, group_size)
    abs_max = w_groups.abs().amax(dim=2, keepdim=True)
    scale = (abs_max / FP8_MAX).clamp(min=1e-7)
    q = (w_groups.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(N, K), scale.squeeze(2).float()


def quantize_act_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-token (row) dynamic FP8 quantization of activations."""
    x_f = x.float()
    scale = (x_f.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-7)
    q = (x_f / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def reference_static_fp8_linear(
    x: torch.Tensor,  # BF16 [M, K]
    weight_fp8: torch.Tensor,  # FP8 [N, K]
    weight_scale: torch.Tensor,  # float32 [N, 1]
    act_scale: torch.Tensor,  # float32 scalar
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Static per-tensor: quantize/dequant x with act_scale, dequant weight, matmul."""
    x_q = (
        (x.float() / act_scale.item()).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    )
    x_dq = x_q.float() * act_scale.item()
    w_dq = weight_fp8.float() * weight_scale.view(-1, 1)
    out = torch.mm(x_dq, w_dq.t())
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


def reference_dynamic_fp8_linear(
    x: torch.Tensor,  # BF16 [M, K]
    weight_fp8: torch.Tensor,  # FP8 [N, K]
    weight_scale: torch.Tensor,  # float32 [N, G]
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dynamic per-token act x per-group weight: quantize x per-token,
    dequant weight per-group, matmul."""
    N, K = weight_fp8.shape
    G = weight_scale.shape[1]
    group_size = K // G
    x_q, x_scale = quantize_act_per_token(x)
    x_dq = x_q.float() * x_scale
    w_dq = (
        weight_fp8.float().view(N, G, group_size) * weight_scale.view(N, G, 1)
    ).view(N, K)
    out = torch.mm(x_dq, w_dq.t())
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


# ---------------------------------------------------------------------------
# W8A8 tests: fp8_scaled_mm_with_quant
# ---------------------------------------------------------------------------
W8A8_M_SIZES = [1, 4, 16, 64]
W8A8_NK_SIZES = [(128, 256), (256, 512), (64, 128)]


@requires_cpu_fp8_w8a8
@pytest.mark.parametrize("static_activation", [True, False])
@pytest.mark.parametrize("M", W8A8_M_SIZES)
@pytest.mark.parametrize("N,K", W8A8_NK_SIZES)
@pytest.mark.parametrize("use_bias", [False, True])
def test_cpu_fp8_w8a8_per_tensor(
    M: int, N: int, K: int, use_bias: bool, static_activation: bool
):
    """CPUFP8W8A8ScaledMMLinearKernel with per-tensor weight scale."""
    torch.manual_seed(42)
    out_dtype = torch.bfloat16

    x = torch.randn(M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)
    bias = torch.randn(N, dtype=torch.float32) if use_bias else None

    w_fp8, w_scale = quantize_weight_per_tensor(weight)
    packed_weight, packed_scale = torch.ops._C.float8_linear_prepack_cpu(w_fp8, w_scale)

    if static_activation:
        act_scale = (x.float().abs().max() / FP8_MAX).clamp(min=1e-7).view(1)
        kernel_out = ops.fp8_scaled_mm_with_quant(
            x, act_scale, False, packed_weight, packed_scale, bias, out_dtype
        )
        ref_out = reference_static_fp8_linear(
            x, w_fp8, w_scale, act_scale, bias, out_dtype
        )
        atol = 0.05
    else:
        kernel_out = ops.fp8_scaled_mm_with_quant(
            x, None, True, packed_weight, packed_scale, bias, out_dtype
        )
        ref_out = reference_dynamic_fp8_linear(x, w_fp8, w_scale, bias, out_dtype)
        atol = 0.8

    assert kernel_out.dtype == out_dtype
    torch.testing.assert_close(kernel_out, ref_out, rtol=0.05, atol=atol)


@requires_cpu_fp8_w8a8
@pytest.mark.parametrize("static_activation", [True, False])
@pytest.mark.parametrize("M", W8A8_M_SIZES)
@pytest.mark.parametrize("N,K", W8A8_NK_SIZES)
@pytest.mark.parametrize("use_bias", [False, True])
def test_cpu_fp8_w8a8_per_channel(
    M: int, N: int, K: int, use_bias: bool, static_activation: bool
):
    """CPUFP8W8A8ScaledMMLinearKernel with per-channel weight scale."""
    torch.manual_seed(42)
    out_dtype = torch.bfloat16

    x = torch.randn(M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)
    bias = torch.randn(N, dtype=torch.float32) if use_bias else None

    w_fp8, w_scale = quantize_weight_per_channel(weight)
    packed_weight, packed_scale = torch.ops._C.float8_linear_prepack_cpu(w_fp8, w_scale)

    if static_activation:
        act_scale = (x.float().abs().max() / FP8_MAX).clamp(min=1e-7).view(1)
        kernel_out = ops.fp8_scaled_mm_with_quant(
            x, act_scale, False, packed_weight, packed_scale, bias, out_dtype
        )
        ref_out = reference_static_fp8_linear(
            x, w_fp8, w_scale, act_scale, bias, out_dtype
        )
        atol = 0.05
    else:
        kernel_out = ops.fp8_scaled_mm_with_quant(
            x, None, True, packed_weight, packed_scale, bias, out_dtype
        )
        ref_out = reference_dynamic_fp8_linear(x, w_fp8, w_scale, bias, out_dtype)
        atol = 0.8

    assert kernel_out.dtype == out_dtype
    torch.testing.assert_close(kernel_out, ref_out, rtol=0.05, atol=atol)


@requires_cpu_fp8_w8a8
@pytest.mark.parametrize("M", W8A8_M_SIZES)
@pytest.mark.parametrize("N,K", W8A8_NK_SIZES)
@pytest.mark.parametrize("use_bias", [False, True])
def test_cpu_fp8_w8a8_per_group(M: int, N: int, K: int, use_bias: bool):
    """CPUFp8W8A8BlockScaledMMKernel: per-group (128-block) weight scale +
    dynamic per-token activation scale."""
    torch.manual_seed(42)
    out_dtype = torch.bfloat16

    x = torch.randn(M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)
    bias = torch.randn(N, dtype=torch.float32) if use_bias else None

    w_fp8, w_scale = quantize_weight_per_group(weight)
    packed_weight, packed_scale = torch.ops._C.float8_linear_prepack_cpu(w_fp8, w_scale)
    kernel_out = ops.fp8_scaled_mm_with_quant(
        x, None, True, packed_weight, packed_scale, bias, out_dtype
    )
    ref_out = reference_dynamic_fp8_linear(x, w_fp8, w_scale, bias, out_dtype)

    assert kernel_out.dtype == out_dtype
    torch.testing.assert_close(kernel_out, ref_out, rtol=0.05, atol=0.8)
