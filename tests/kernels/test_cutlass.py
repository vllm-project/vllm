"""Tests for cutlass kernels

Run `pytest tests/kernels/test_cutlass.py`.
"""
import pytest
import torch

from vllm import _custom_ops as ops

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]


def to_fp8(tensor):
    # Assuming input tensor is float32
    # Scale tensor to range of FP8 E4M3
    # by clamping exponent and truncating mantissa
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


def cutlass_fp8_gemm_helper(
    m: int,
    n: int,
    k: int,
    per_token_act_quant: bool,
    per_out_channel_weight_quant: bool,
):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    a = to_fp8(torch.randn((m, k), device="cuda"))
    b = to_fp8(torch.randn((n, k), device="cuda").t())

    m_a_scales = m if per_token_act_quant else 1
    n_b_scales = n if per_out_channel_weight_quant else 1

    scale_a = (torch.randn(
        (m_a_scales, 1), device="cuda", dtype=torch.float32) / 10)
    scale_b = (torch.randn(
        (1, n_b_scales), device="cuda", dtype=torch.float32) / 10)

    out = ops.cutlass_scaled_mm_dq(a, b, scale_a, scale_b)
    baseline = torch.mm(scale_a * a.to(dtype=torch.float32),
                        scale_b *
                        b.to(dtype=torch.float32)).to(dtype=torch.bfloat16)

    assert torch.allclose(out, baseline, rtol=1e-2, atol=1e-1)


def cutlass_int8_gemm_helper(
    m: int,
    n: int,
    k: int,
    per_token_act_quant: bool,
    per_out_channel_weight_quant: bool,
):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    a = to_int8(torch.randn((m, k), device="cuda") * 5)
    b = to_int8(torch.randn((n, k), device="cuda").t() * 5)

    m_a_scales = m if per_token_act_quant else 1
    n_b_scales = n if per_out_channel_weight_quant else 1

    scale_a = (torch.randn(
        (m_a_scales, 1), device="cuda", dtype=torch.float32) / 10)
    scale_b = (torch.randn(
        (1, n_b_scales), device="cuda", dtype=torch.float32) / 10)

    out = ops.cutlass_scaled_mm_dq(a, b, scale_a, scale_b)
    baseline = torch.mm(scale_a * a.to(dtype=torch.float32),
                        scale_b *
                        b.to(dtype=torch.float32)).to(dtype=torch.bfloat16)

    assert torch.allclose(out, baseline, rtol=1e-1, atol=1e0)


@pytest.mark.parametrize("m", [512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.skipif(capability < 89,
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm(m: int, n: int, k: int, per_act_token: bool,
                          per_out_ch: bool):
    cutlass_fp8_gemm_helper(m, n, k, per_act_token, per_out_ch)


@pytest.mark.parametrize("m", [512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
def test_cutlass_int8_gemm(m: int, n: int, k: int, per_act_token: bool,
                           per_out_ch: bool):
    cutlass_int8_gemm_helper(m, n, k, per_act_token, per_out_ch)


# For the following two tests:
# N and K correspond to the size of the weight matrix and likely to be multiples
# of a large power of two. In any case, the kernel will have a naive fallback
# when N and K are not divisible by 16. But M is the number of tokens and the
# kernel must handle any M thrown at it.
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.skipif(capability < 89,
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm_m_sweep(per_act_token: bool, per_out_ch: bool):
    for nk in range(32, 128, 32):
        for m in range(1, 128):
            cutlass_fp8_gemm_helper(m, nk, nk, per_act_token, per_out_ch)


@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
def test_cutlass_int8_gemm_m_sweep(per_act_token: bool, per_out_ch: bool):
    for nk in range(32, 128, 32):
        for m in range(1, 128):
            cutlass_int8_gemm_helper(m, nk, nk, per_act_token, per_out_ch)


# Test working with a subset of A and B
def test_cutlass_subset():
    big_m, big_n, big_k = 1024, 1024, 1024
    m, n, k = 512, 512, 512

    whole_a = to_int8(torch.randn((big_m, big_k), device="cuda") * 5)
    whole_b = to_int8(torch.randn((big_n, big_k), device="cuda").t() * 5)
    a = whole_a[0:m, 0:k]
    b = whole_b[0:k, 0:n]

    scale_a = torch.randn((1, 1), device="cuda", dtype=torch.float32) / 10
    scale_b = torch.randn((1, 1), device="cuda", dtype=torch.float32) / 10

    out = ops.cutlass_scaled_mm_dq(a, b, scale_a, scale_b)
    baseline = torch.mm(scale_a * a.to(dtype=torch.float32),
                        scale_b *
                        b.to(dtype=torch.float32)).to(dtype=torch.bfloat16)

    assert torch.allclose(out, baseline, rtol=1e-1, atol=1e0)
