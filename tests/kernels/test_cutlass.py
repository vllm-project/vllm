"""Tests for cutlass kernels

Run `pytest tests/kernels/test_cutlass.py`.
"""
from typing import Type

import pytest
import torch

from vllm import _custom_ops as ops

CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]


def to_fp8(tensor: torch.tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)


def to_int8(tensor: torch.tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def cutlass_fp8_gemm_helper(m: int,
                            n: int,
                            k: int,
                            per_token_act_quant: bool,
                            per_out_channel_weight_quant: bool,
                            out_dtype: Type[torch.dtype] = torch.bfloat16,
                            device: str = "cuda"):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    a = to_fp8(torch.randn((m, k), device=device))
    b = to_fp8(torch.randn((n, k), device=device).t())

    m_a_scales = m if per_token_act_quant else 1
    n_b_scales = n if per_out_channel_weight_quant else 1

    scale_a = (torch.randn(
        (m_a_scales, 1), device=device, dtype=torch.float32) / 10)
    scale_b = (torch.randn(
        (1, n_b_scales), device=device, dtype=torch.float32) / 10)

    out = ops.cutlass_scaled_mm_dq(a, b, scale_a, scale_b, out_dtype)
    baseline = torch.mm(scale_a * a.to(dtype=torch.float32),
                        scale_b * b.to(dtype=torch.float32)).to(out_dtype)

    assert torch.allclose(out, baseline, rtol=1e-2, atol=1e-1)


def cutlass_int8_gemm_helper(m: int,
                             n: int,
                             k: int,
                             per_token_act_quant: bool,
                             per_out_channel_weight_quant: bool,
                             out_dtype: Type[torch.dtype] = torch.bfloat16,
                             device: str = "cuda"):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    a = to_int8(torch.randn((m, k), device=device) * 5)
    b = to_int8(torch.randn((n, k), device=device).t() * 5)

    m_a_scales = m if per_token_act_quant else 1
    n_b_scales = n if per_out_channel_weight_quant else 1

    scale_a = (torch.randn(
        (m_a_scales, 1), device=device, dtype=torch.float32) / 10)
    scale_b = (torch.randn(
        (1, n_b_scales), device=device, dtype=torch.float32) / 10)

    out = ops.cutlass_scaled_mm_dq(a, b, scale_a, scale_b, out_dtype)
    baseline = torch.mm(scale_a * a.to(dtype=torch.float32),
                        scale_b *
                        b.to(dtype=torch.float32)).to(dtype=out_dtype)

    assert torch.allclose(out, baseline, rtol=1e-1, atol=1e0)


@pytest.mark.parametrize("m", [512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 496, 1024])
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.skipif(capability < 89,
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm(m: int, n: int, k: int, per_act_token: bool,
                          per_out_ch: bool):
    cutlass_fp8_gemm_helper(m, n, k, per_act_token, per_out_ch)


@pytest.mark.parametrize("m", [512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 496, 1024])
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
def test_cutlass_int8_gemm(m: int, n: int, k: int, per_act_token: bool,
                           per_out_ch: bool):
    cutlass_int8_gemm_helper(m, n, k, per_act_token, per_out_ch)


@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_cutlass_int8_gemm_output_dtype(per_act_token: bool, per_out_ch: bool,
                                        out_dtype: Type[torch.dtype]):
    cutlass_int8_gemm_helper(512, 512, 512, per_act_token, per_out_ch,
                             out_dtype)


@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(capability < 89,
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm_output_dtype(per_act_token: bool, per_out_ch: bool,
                                       out_dtype: Type[torch.dtype]):
    cutlass_fp8_gemm_helper(512, 512, 512, per_act_token, per_out_ch,
                            out_dtype)


@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.skipif(capability < 89,
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm_devices(per_act_token: bool, per_out_ch: bool,
                                  device: str):
    cutlass_fp8_gemm_helper(512, 512, 512, per_act_token, per_out_ch,
                            torch.bfloat16, device)


@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_cutlass_int8_gemm_devices(per_act_token: bool, per_out_ch: bool,
                                   device: str):
    cutlass_int8_gemm_helper(512, 512, 512, per_act_token, per_out_ch,
                             torch.bfloat16, device)


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

    out = ops.cutlass_scaled_mm_dq(a,
                                   b,
                                   scale_a,
                                   scale_b,
                                   out_dtype=torch.bfloat16)
    baseline = torch.mm(scale_a * a.to(dtype=torch.float32),
                        scale_b *
                        b.to(dtype=torch.float32)).to(dtype=torch.bfloat16)

    assert torch.allclose(out, baseline, rtol=1e-1, atol=1e0)
