# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU FP8 W8A16 block-scaled GEMM kernel (fp8_scaled_mm_cpu).

Run `pytest tests/kernels/quantization/test_cpu_fp8_scaled_mm.py -v`.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
    CPUFp8PerTensorScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

if not ops._supports_cpu_fp8_w8a16:
    pytest.skip("fp8_scaled_mm_cpu op not available", allow_module_level=True)

if not torch.cpu._is_amx_tile_supported():
    pytest.skip("requires AMX tile support", allow_module_level=True)

BLOCK_SIZE = [128, 128]


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
# Test parameters
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


def quantize_weight_per_tensor_fp8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight [N, K] to FP8 with a single per-tensor scale."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = weight.abs().amax() / fp8_max
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = (weight / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return q, scale


def ref_fp8_per_tensor_scaled_mm(
    x: torch.Tensor,
    fp8_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    w_dq = fp8_weight.float() * scale
    out = torch.mm(x.float(), w_dq.t())
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


NK_SIZES_PER_TENSOR = [
    (32, 64),
    (5120, 5120),
]


@pytest.mark.parametrize("M", [1, 64])
@pytest.mark.parametrize("N,K", NK_SIZES_PER_TENSOR)
@pytest.mark.parametrize("use_bias", [False, True])
def test_cpu_fp8_per_tensor_scaled_mm_kernel(
    M: int, N: int, K: int, use_bias: bool, default_vllm_config
):
    """CPUFp8PerTensorScaledMMLinearKernel correctness against float reference.

    Exercises the full kernel class (not just the raw op), including the
    weight-orientation fixup: Fp8LinearMethod stores `layer.weight` as
    [K, N] (torch._scaled_mm convention) before calling
    process_weights_after_loading, so the kernel must transpose back to
    [N, K] before VNNI-packing.
    """
    torch.manual_seed(0)
    out_dtype = torch.bfloat16

    x = torch.randn(M, K, dtype=out_dtype) / (K**0.5)
    w_f32 = torch.randn(N, K, dtype=torch.float32) / (K**0.5)
    fp8_weight, scale = quantize_weight_per_tensor_fp8(w_f32)
    bias = torch.randn(N, dtype=torch.float32) * 0.1 if use_bias else None

    ref_out = ref_fp8_per_tensor_scaled_mm(x, fp8_weight, scale, bias, out_dtype)

    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticTensorSym,
        activation_quant_key=kFp8StaticTensorSym,
        weight_shape=(N, K),
        input_dtype=out_dtype,
        out_dtype=out_dtype,
    )
    kernel = CPUFp8PerTensorScaledMMLinearKernel(
        config,
        layer_param_names=["weight", "weight_scale", "input_scale", "input_scale_ub"],
    )

    layer = torch.nn.Module()
    # Fp8LinearMethod stores the weight transposed to [K, N] before calling
    # process_weights_after_loading.
    layer.register_parameter(
        "weight", torch.nn.Parameter(fp8_weight.t().contiguous(), requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale", torch.nn.Parameter(scale.clone(), requires_grad=False)
    )
    kernel.process_weights_after_loading(layer)

    kernel_out = kernel.apply_weights(layer, x, bias)

    assert kernel_out.dtype == out_dtype
    torch.testing.assert_close(kernel_out, ref_out, rtol=0.02, atol=0.01)


@pytest.mark.parametrize("n", [16, 48])
def test_cpu_fp8_per_tensor_kernel_rejects_non_multiple_of_32_n(n: int):
    """The AMX tinygemm kernel tiles N in chunks of 32 and cannot handle a
    remainder tile smaller than that, so can_implement must reject any N
    that isn't a multiple of 32 rather than let it crash the process.
    """
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticTensorSym,
        activation_quant_key=kFp8StaticTensorSym,
        weight_shape=(n, 64),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    supported, _ = CPUFp8PerTensorScaledMMLinearKernel.can_implement(config)
    assert not supported
