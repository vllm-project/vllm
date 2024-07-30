"""Tests for cutlass kernels

Run `pytest tests/kernels/test_cutlass.py`.
"""
from typing import Optional, Type

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]

capability = current_platform.get_device_capability()
capability = capability[0] * 10 + capability[1]


def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def baseline_scaled_mm(a: torch.Tensor,
                       b: torch.Tensor,
                       scale_a: torch.Tensor,
                       scale_b: torch.Tensor,
                       out_dtype: Type[torch.dtype],
                       bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    output = (scale_a * (scale_b * (torch.mm(
        a.to(dtype=torch.float32), b.to(dtype=torch.float32))))).to(out_dtype)
    if bias is not None:
        output = output + bias

    return output


def cutlass_fp8_gemm_helper(m: int,
                            n: int,
                            k: int,
                            per_token_act_quant: bool,
                            per_out_channel_weight_quant: bool,
                            use_bias: bool,
                            out_dtype: Type[torch.dtype] = torch.bfloat16,
                            device: str = "cuda"):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    a = to_fp8(torch.randn((m, k), device=device))
    b = to_fp8(torch.randn((n, k), device=device).t())

    m_a_scales = m if per_token_act_quant else 1
    n_b_scales = n if per_out_channel_weight_quant else 1

    scale_a = (torch.randn((m_a_scales, 1), device=device,
                           dtype=torch.float32))
    scale_b = (torch.randn((1, n_b_scales), device=device,
                           dtype=torch.float32))
    if use_bias:
        bias = torch.rand((n, ), device=device, dtype=out_dtype) * 10
    else:
        bias = None

    out = ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    baseline = baseline_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    assert torch.allclose(out, baseline, rtol=1e-2, atol=5e-2)


def cutlass_int8_gemm_helper(m: int,
                             n: int,
                             k: int,
                             per_token_act_quant: bool,
                             per_out_channel_weight_quant: bool,
                             use_bias: bool,
                             out_dtype: Type[torch.dtype] = torch.bfloat16,
                             device: str = "cuda"):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    a = to_int8(torch.randn((m, k), device=device) * 5)
    b = to_int8(torch.randn((n, k), device=device).t() * 5)

    m_a_scales = m if per_token_act_quant else 1
    n_b_scales = n if per_out_channel_weight_quant else 1

    scale_a = (torch.randn((m_a_scales, 1), device=device,
                           dtype=torch.float32))
    scale_b = (torch.randn((1, n_b_scales), device=device,
                           dtype=torch.float32))

    if use_bias:
        bias = torch.rand((n, ), device=device, dtype=out_dtype) * 10
    else:
        bias = None

    out = ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    baseline = baseline_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    assert torch.allclose(out, baseline, rtol=1e-1, atol=1e0)


@pytest.mark.parametrize("m", [1, 16, 32, 64, 128, 256, 512, 222, 100, 33])
@pytest.mark.parametrize("n", [2048, 4096, 8192, 16384, 24576, 256, 1024])
@pytest.mark.parametrize("k", [128, 496, 1024])
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(capability < 89,
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm(m: int, n: int, k: int, per_act_token: bool,
                          per_out_ch: bool, use_bias: bool):
    cutlass_fp8_gemm_helper(m, n, k, per_act_token, per_out_ch, use_bias)


@pytest.mark.parametrize("m", [1, 16, 32, 64, 128, 256, 512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 8192, 16384, 256, 1024])
@pytest.mark.parametrize("k", [128, 496, 1024])
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_cutlass_int8_gemm(m: int, n: int, k: int, per_act_token: bool,
                           per_out_ch: bool, use_bias: bool):
    cutlass_int8_gemm_helper(m, n, k, per_act_token, per_out_ch, use_bias)


@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_bias", [True, False])
def test_cutlass_int8_gemm_output_dtype(per_act_token: bool, per_out_ch: bool,
                                        out_dtype: Type[torch.dtype],
                                        use_bias: bool):
    cutlass_int8_gemm_helper(512,
                             512,
                             512,
                             per_act_token,
                             per_out_ch,
                             use_bias,
                             out_dtype=out_dtype)


@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(capability < 89,
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm_output_dtype(per_act_token: bool, per_out_ch: bool,
                                       out_dtype: Type[torch.dtype],
                                       use_bias: bool):
    cutlass_fp8_gemm_helper(512,
                            512,
                            512,
                            per_act_token,
                            per_out_ch,
                            use_bias,
                            out_dtype=out_dtype)


@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.skipif(capability < 89,
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm_devices(per_act_token: bool, per_out_ch: bool,
                                  use_bias: bool, device: str):
    cutlass_fp8_gemm_helper(512, 512, 512, per_act_token, per_out_ch, use_bias,
                            torch.bfloat16, device)


@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_cutlass_int8_gemm_devices(per_act_token: bool, per_out_ch: bool,
                                   use_bias: bool, device: str):
    cutlass_int8_gemm_helper(512,
                             512,
                             512,
                             per_act_token,
                             per_out_ch,
                             use_bias,
                             out_dtype=torch.bfloat16,
                             device=device)


# For the following two tests:
# N and K correspond to the size of the weight matrix and likely to be multiples
# of a large power of two. In any case, the kernel will have a naive fallback
# when N and K are not divisible by 16. But M is the number of tokens and the
# kernel must handle any M thrown at it.
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(capability < 89,
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm_m_sweep(per_act_token: bool, per_out_ch: bool,
                                  use_bias: bool):
    for nk in range(32, 128, 32):
        for m in range(1, 128):
            cutlass_fp8_gemm_helper(m, nk, nk, per_act_token, per_out_ch,
                                    use_bias)


@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_cutlass_int8_gemm_m_sweep(per_act_token: bool, per_out_ch: bool,
                                   use_bias: bool):
    for nk in range(32, 128, 32):
        for m in range(1, 128):
            cutlass_int8_gemm_helper(m, nk, nk, per_act_token, per_out_ch,
                                     use_bias)


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

    out = ops.cutlass_scaled_mm(a,
                                b,
                                scale_a,
                                scale_b,
                                out_dtype=torch.bfloat16)
    baseline = baseline_scaled_mm(a,
                                  b,
                                  scale_a,
                                  scale_b,
                                  out_dtype=torch.bfloat16)

    assert torch.allclose(out, baseline, rtol=1e-1, atol=1e0)


# Test to make sure cuda graphs work
class CutlassLayer(torch.nn.Module):

    def __init__(self, b, scale_a, scale_b, out_dtype):
        super().__init__()
        self.b = b
        self.scale_a = scale_a
        self.scale_b = scale_b
        self.out_dtype = out_dtype

    def forward(self, a):
        return ops.cutlass_scaled_mm(a, self.b, self.scale_a, self.scale_b,
                                     self.out_dtype)


@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
def test_cutlass_cuda_graph(per_act_token: bool, per_out_ch: bool):
    m, n, k = 512, 512, 512

    a = to_int8(torch.randn((m, k), device="cuda"))
    b = to_int8(torch.randn((n, k), device="cuda").t())

    m_a_scales = m if per_act_token else 1
    n_b_scales = n if per_out_ch else 1

    scale_a = (torch.randn(
        (m_a_scales, 1), device="cuda", dtype=torch.float32) / 10)
    scale_b = (torch.randn(
        (1, n_b_scales), device="cuda", dtype=torch.float32) / 10)

    # Construct a trivial model with a single layer that calls a CUTLASS kernel
    model = CutlassLayer(b, scale_a, scale_b, torch.bfloat16)

    # Run the model with a cuda graph
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = model(a)
    out.zero_()
    g.replay()

    baseline = torch.mm(scale_a * a.to(dtype=torch.float32),
                        scale_b * b.to(dtype=torch.float32)).to(torch.bfloat16)
    assert torch.allclose(out, baseline, rtol=1e-1, atol=1e0)
