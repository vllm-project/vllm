# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Triton NVFP4 GEMM kernel.

Compares the Triton kernel output against a PyTorch reference that
dequantizes FP4 values with linear (non-swizzled) block scales and
performs a standard matmul.
"""

import sys
from pathlib import Path

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="NVFP4 requires compute capability of 10.0 or above (Blackwell).",
        allow_module_level=True,
    )

# Add the benchmarks/kernels directory so we can import the Triton kernel
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "kernels"))
from nvfp4_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, break_fp4_bytes
from triton_nvfp4_gemm import triton_scaled_fp4_mm

DTYPES = [torch.bfloat16]
# (M, N, K) where K is the logical (unpacked) dimension
SHAPES = [
    (128, 128, 128),
    (256, 128, 128),
    (128, 256, 256),
    (256, 256, 256),
    (512, 512, 256),
    (128, 128, 256),
]
SEEDS = [42]


def dequantize_linear_scales(
    fp4: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize FP4 tensor with linear (non-swizzled) block scales."""
    m, k_packed = fp4.shape
    k = k_packed * 2
    block_size = 16

    # Unpack FP4 nibbles to float values
    values = break_fp4_bytes(fp4, torch.float32).reshape(m, k // block_size, block_size)
    # Apply block scales: dequant = fp4_val * (block_scale / global_scale)
    scale_f32 = scale.to(torch.float32) / global_scale
    out = (values * scale_f32.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype)


def get_ref_results(
    a_fp4: torch.Tensor,
    b_fp4: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    a_global_scale: torch.Tensor,
    b_global_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute reference output by dequantizing then doing matmul."""
    a_deq = dequantize_linear_scales(a_fp4, a_scale, a_global_scale, dtype)
    b_deq = dequantize_linear_scales(b_fp4, b_scale, b_global_scale, dtype)
    return torch.matmul(a_deq, b_deq.t())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_triton_nvfp4_gemm_correctness(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    seed: int,
) -> None:
    """Test Triton NVFP4 GEMM against dequantized reference."""
    set_random_seed(seed)
    m, n, k = shape
    device = "cuda"

    a_bf16 = torch.randn((m, k), dtype=dtype, device=device)
    b_bf16 = torch.randn((n, k), dtype=dtype, device=device)

    # Compute global scales
    a_global_scale = (
        FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (torch.abs(a_bf16).max().float() + 1e-12)
    )
    b_global_scale = (
        FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (torch.abs(b_bf16).max().float() + 1e-12)
    )
    alpha = 1.0 / (a_global_scale * b_global_scale)

    # Quantize with linear (non-swizzled) scales for Triton
    a_fp4, a_scale = ops.scaled_fp4_quant(
        a_bf16, a_global_scale, is_sf_swizzled_layout=False
    )
    b_fp4, b_scale = ops.scaled_fp4_quant(
        b_bf16, b_global_scale, is_sf_swizzled_layout=False
    )

    # Triton kernel
    out = triton_scaled_fp4_mm(a_fp4, b_fp4, a_scale, b_scale, alpha, dtype)

    # Reference
    expected = get_ref_results(
        a_fp4,
        b_fp4,
        a_scale,
        b_scale,
        a_global_scale,
        b_global_scale,
        dtype,
    )

    assert out.shape == expected.shape
    assert out.dtype == dtype
    # FP4 quantization introduces significant rounding error;
    # use tolerances consistent with existing NVFP4 tests.
    torch.testing.assert_close(
        out,
        expected.to(dtype),
        atol=1e-1,
        rtol=1e-1,
        msg=f"Mismatch at M={m}, N={n}, K={k}, dtype={dtype}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_triton_nvfp4_gemm_output_shape(dtype: torch.dtype) -> None:
    """Verify output has correct shape and dtype."""
    m, n, k = 256, 128, 256
    device = "cuda"

    a_bf16 = torch.randn((m, k), dtype=dtype, device=device)
    b_bf16 = torch.randn((n, k), dtype=dtype, device=device)

    a_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (torch.abs(a_bf16).max().float() + 1e-12)
    b_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (torch.abs(b_bf16).max().float() + 1e-12)
    alpha = 1.0 / (a_gs * b_gs)

    a_fp4, a_scale = ops.scaled_fp4_quant(a_bf16, a_gs, is_sf_swizzled_layout=False)
    b_fp4, b_scale = ops.scaled_fp4_quant(b_bf16, b_gs, is_sf_swizzled_layout=False)

    out = triton_scaled_fp4_mm(a_fp4, b_fp4, a_scale, b_scale, alpha, dtype)

    assert out.shape == (m, n)
    assert out.dtype == dtype
    assert out.device.type == "cuda"


@pytest.mark.parametrize(
    "m,n,k",
    [(128, 128, 128), (256, 256, 256), (512, 256, 128)],
)
@torch.inference_mode()
def test_triton_vs_cutlass_nvfp4(m: int, n: int, k: int) -> None:
    """Cross-check Triton output against CUTLASS kernel output.

    Both kernels quantize the same input data (with their respective
    scale layouts) and should agree to within FP4 quantization error.
    """
    device = "cuda"
    dtype = torch.bfloat16

    a_bf16 = torch.randn((m, k), dtype=dtype, device=device)
    b_bf16 = torch.randn((n, k), dtype=dtype, device=device)

    a_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (torch.abs(a_bf16).max().float() + 1e-12)
    b_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (torch.abs(b_bf16).max().float() + 1e-12)
    alpha = 1.0 / (a_gs * b_gs)

    # CUTLASS: swizzled scales
    a_fp4_sw, a_scale_sw = ops.scaled_fp4_quant(a_bf16, a_gs)
    b_fp4_sw, b_scale_sw = ops.scaled_fp4_quant(b_bf16, b_gs)
    cutlass_out = ops.cutlass_scaled_fp4_mm(
        a_fp4_sw, b_fp4_sw, a_scale_sw, b_scale_sw, alpha, dtype
    )

    # Triton: linear scales
    a_fp4_lin, a_scale_lin = ops.scaled_fp4_quant(
        a_bf16, a_gs, is_sf_swizzled_layout=False
    )
    b_fp4_lin, b_scale_lin = ops.scaled_fp4_quant(
        b_bf16, b_gs, is_sf_swizzled_layout=False
    )
    triton_out = triton_scaled_fp4_mm(
        a_fp4_lin, b_fp4_lin, a_scale_lin, b_scale_lin, alpha, dtype
    )

    # Both should produce similar results despite different scale layouts
    torch.testing.assert_close(
        triton_out,
        cutlass_out,
        atol=1e-1,
        rtol=1e-1,
        msg=f"Triton vs CUTLASS mismatch at M={m}, N={n}, K={k}",
    )
