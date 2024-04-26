"""Tests for cutlass kernels

Run `pytest tests/kernels/test_cutlass.py`.
"""
import pytest
import torch

from vllm import _custom_ops as ops

def to_fp8(tensor):
    # Assuming input tensor is float32
    # Scale tensor to range of FP8 E4M3 by clamping exponent and truncating mantissa
    max_exp = 2**4 - 1  # Maximum exponent for E4M3
    max_mantissa = 2**3 - 1  # Maximum mantissa for E4M3
    base = 2**max_exp
    # Scale the mantissa
    scaled = torch.clamp(tensor, -base, base)
    # Quantize the mantissa
    quantized = torch.round(scaled * max_mantissa) / max_mantissa
    return quantized.to(dtype=torch.float8_e4m3fn)

def to_int8(tensor):
    return torch.round(torch.clamp(tensor, -128, 127)).to(dtype=torch.int8)

def cutlass_fp8_gemm_per_row_and_col_scales(
    m: int,
    n: int,
    k: int,
):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    a = to_fp8(torch.randn((m, k), device='cuda'))
    b = to_fp8(torch.randn((n, k), device='cuda').t())

    scale_a = torch.randn((m,1), device='cuda', dtype=torch.float32) / 10
    scale_b = torch.randn((1,n), device='cuda', dtype=torch.float32) / 10

    out = ops.cutlass_scaled_mm_dq(a, b, scale_a, scale_b)
    baseline = torch.mm(scale_a * a.to(dtype=torch.float32), 
                        scale_b * b.to(dtype=torch.float32)).to(dtype=torch.bfloat16)

    assert torch.allclose(out, baseline, rtol=1e-2, atol=1e-1)

def cutlass_int8_gemm_per_row_and_col_scales(
    m: int,
    n: int,
    k: int,
):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    a = to_int8(torch.randn((m, k), device='cuda') * 5)
    b = to_int8(torch.randn((n, k), device='cuda').t() * 5)

    scale_a = torch.randn((m,1), device='cuda', dtype=torch.float32) / 10
    scale_b = torch.randn((1,n), device='cuda', dtype=torch.float32) / 10

    out = ops.cutlass_scaled_mm_dq(a, b, scale_a, scale_b)
    baseline = torch.mm(scale_a * a.to(dtype=torch.float32), 
                        scale_b * b.to(dtype=torch.float32)).to(dtype=torch.bfloat16)

    assert torch.allclose(out, baseline, rtol=1e-2, atol=1e-1)

@pytest.mark.parametrize("m", [512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 1024])
def test_cutlass_fp8_gemm(
    m: int, n: int, k: int,
):
    cutlass_fp8_gemm_per_row_and_col_scales(m,n,k)
    

@pytest.mark.parametrize("m", [512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [511])
@pytest.mark.skip(reason="Illegal instruction at k=511")
def test_cutlass_bad_size(
    m: int, n: int, k: int,
):
    cutlass_int8_gemm_per_row_and_col_scales(m,n,k)
    cutlass_fp8_gemm_per_row_and_col_scales(m,n,k)

@pytest.mark.parametrize("m", [512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 1024])
def test_cutlass_int8_gemm(
    m: int,
    n: int,
    k: int,
):
    cutlass_int8_gemm_per_row_and_col_scales(m,n,k)

@pytest.mark.parametrize("m", [512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 1024])
def test_cutlass_int8_gemm_per_tensor_scales(
    m: int,
    n: int,
    k: int,
):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    a = to_int8(torch.randn((m, k), device='cuda') * 5)
    b = to_int8(torch.randn((n, k), device='cuda').t() * 5)

    scale_a = torch.randn((1,1), dtype=torch.float32) / 10
    scale_b = torch.randn((1,1), dtype=torch.float32) / 10

    out = ops.cutlass_scaled_mm_dq(a, b, scale_a, scale_b)
    baseline = torch.mm(scale_a.to(device='cuda') * a.to(dtype=torch.float32), 
                        scale_b.to(device='cuda') * b.to(dtype=torch.float32)).to(dtype=torch.bfloat16)

    assert torch.allclose(out, baseline, rtol=1e-4, atol=1e-1)
