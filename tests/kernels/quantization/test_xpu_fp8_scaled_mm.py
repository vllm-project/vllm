# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for XPU FP8 W8A8 block-scaled GEMM kernel (fp8_gemm).

Run `pytest tests/kernels/quantization/test_xpu_fp8_scaled_mm.py -v`.
"""

import itertools

import pytest
import torch

from tests.kernels.quant_utils import (
    native_per_token_group_quant_fp8,
    native_w8a8_block_matmul,
)
from vllm.platforms import current_platform

if not current_platform.is_xpu():
    pytest.skip("skipping XPU-only tests", allow_module_level=True)

if not hasattr(torch.ops, "_xpu_C") or not hasattr(torch.ops._xpu_C, "fp8_gemm"):
    pytest.skip("_xpu_C.fp8_gemm op not available", allow_module_level=True)

BLOCK_SIZE = [128, 128]

# Test parameters
M_SIZES = [1, 4, 16, 64, 128]
NK_SIZES = [
    (128, 256),
    (256, 512),
    (512, 1024),
    (1024, 2048),
    (5120, 5120),
]
GROUP_SIZE = [128]
OUT_DTYPES = [torch.bfloat16]
SEEDS = [0]


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


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


@pytest.fixture(autouse=True)
def setup_xpu():
    torch.set_default_device("xpu")


@pytest.mark.parametrize(
    "M,N_K,out_dtype,seed",
    itertools.product(M_SIZES, NK_SIZES, OUT_DTYPES, SEEDS),
)
@torch.inference_mode()
def test_xpu_fp8_block_scaled_gemm(
    M: int, N_K: tuple[int, int], out_dtype: torch.dtype, seed: int
):
    """fp8_gemm correctness against native block-scaled matmul reference."""
    N, K = N_K
    torch.manual_seed(seed)
    block_size = BLOCK_SIZE
    block_n, block_k = block_size

    # Generate random input and weight in float
    x = torch.randn(M, K, dtype=out_dtype) / (K**0.5)
    w_f32 = torch.randn(N, K, dtype=torch.float32) / (K**0.5)

    # Quantize weight to FP8 with block scales
    B_fp8, Bs = quantize_weight_block_fp8(w_f32, block_size)

    # Quantize activation per-token-group
    A_fp8, As = native_per_token_group_quant_fp8(x, block_k)

    # Reference: native block matmul
    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    # XPU kernel expects:
    #   q_input [M, K], q_weight [K, N] (transposed), out_dtype,
    #   input_scales, weight_scales, bias
    # Weight is stored [N, K]; Bs is [n_tiles, k_tiles].
    # The kernel (via XPUFp8BlockScaledMMKernel.apply_block_scaled_mm)
    # transposes weight and weight_scale before calling fp8_gemm.
    kernel_out = torch.ops._xpu_C.fp8_gemm(
        A_fp8,
        B_fp8.t(),
        out_dtype,
        As,
        Bs.t().contiguous(),
        torch.Tensor(),
    )

    assert kernel_out.dtype == out_dtype
    rel_diff = torch.mean(torch.abs(kernel_out.float() - ref_out.float())) / torch.mean(
        torch.abs(ref_out.float())
    )
    assert rel_diff < 0.001, f"relative diff {rel_diff} >= 0.001"
