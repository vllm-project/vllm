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

def saturating_conversion(tensor: torch.tensor, out_dtype: torch.dtype):
    dummy = torch.tensor([], dtype=out_dtype)
    if torch.is_floating_point(dummy):
        finfo = torch.finfo(out_dtype)
        max = finfo.max
        min = finfo.min
    else: 
        iinfo = torch.iinfo(out_dtype)
        max = iinfo.max
        min = iinfo.min

    return tensor.clamp(min=min, max=max).to(out_dtype)

def to_fp8(tensor: torch.tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)


def to_int8(tensor: torch.tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def cutlass_fp8_gemm_helper(m: int,
                            n: int,
                            k: int,
                            out_dtype: Type[torch.dtype] = torch.bfloat16,
                            device: str = "cuda"):
    a = to_fp8(torch.randn((m, k), device=device))
    b = to_fp8(torch.randn((n, k), device=device).t())

    out = ops.cutlass_gemm(a, b, out_dtype)
    baseline = torch.mm(a.to(dtype=torch.float32),
                        b.to(dtype=torch.float32)).to(out_dtype)

#Convert outputs to fp32, since allclose is not implemented for fp8_e4m3
    assert torch.allclose(out.to(torch.float32),
                          baseline.to(torch.float32),
                          rtol=1e-2,
                          atol=1e-1)


def cutlass_int8_gemm_helper(m: int,
                             n: int,
                             k: int,
                             out_dtype: Type[torch.dtype] = torch.bfloat16,
                             device: str = "cuda"):
    a = to_int8(torch.randn((m, k), device=device) * 5)
    b = to_int8(torch.randn((n, k), device=device).t() * 5)

    out = ops.cutlass_gemm(a, b, out_dtype)
    baseline = saturating_conversion(torch.mm(a.to(dtype=torch.float32),
                        b.to(dtype=torch.float32)), out_dtype)
    torch.mm(a.to(dtype=torch.float32),
                        b.to(dtype=torch.float32)).to(dtype=out_dtype)

    assert torch.allclose(out, baseline, rtol=1e-1, atol=1e0)


@pytest.mark.parametrize("m", [512, 222, 100, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 496, 1024])
@pytest.mark.skipif(capability < 89,
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm(m: int, n: int, k: int):
    cutlass_fp8_gemm_helper(m, n, k)


@pytest.mark.parametrize("m", [512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 496, 1024])
def test_cutlass_int8_gemm(m: int, n: int, k: int):
    cutlass_int8_gemm_helper(m, n, k)


@pytest.mark.parametrize("out_dtype",
                         [torch.bfloat16, torch.float16, torch.int8])
def test_cutlass_int8_gemm_output_dtype(out_dtype: Type[torch.dtype]):
    cutlass_int8_gemm_helper(512, 512, 512, out_dtype)


@pytest.mark.parametrize("out_dtype",
                         [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
@pytest.mark.skipif(capability < 89,
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm_output_dtype(out_dtype: Type[torch.dtype]):
    cutlass_fp8_gemm_helper(512, 512, 512, out_dtype)


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.skipif(capability < 89,
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm_devices(device: str):
    cutlass_fp8_gemm_helper(512, 512, 512, torch.bfloat16, device)


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_cutlass_int8_gemm_devices(device: str):
    cutlass_int8_gemm_helper(512, 512, 512, torch.bfloat16, device)

#For the following two tests:
#N and K correspond to the size of the weight matrix and likely to be multiples
#of a large power of two.In any case, the kernel will have a naive fallback
#when N and K are not divisible by 16. But M is the number of tokens and the
#kernel must handle any M thrown at it.
@pytest.mark.skipif(capability < 89,
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm_m_sweep():
    for nk in range(32, 128, 32):
        for m in range(1, 128):
            cutlass_fp8_gemm_helper(m, nk, nk)


def test_cutlass_int8_gemm_m_sweep():
    for nk in range(32, 128, 32):
        for m in range(1, 128):
            cutlass_int8_gemm_helper(m, nk, nk)

#Test working with a subset of A and B
def test_cutlass_subset():
    big_m, big_n, big_k = 1024, 1024, 1024
    m, n, k = 512, 512, 512

    whole_a = to_int8(torch.randn((big_m, big_k), device="cuda") * 5)
    whole_b = to_int8(torch.randn((big_n, big_k), device="cuda").t() * 5)
    a = whole_a[0:m, 0:k]
    b = whole_b[0:k, 0:n]

    out = ops.cutlass_gemm(a,
                           b,
                           out_dtype=torch.bfloat16)
    baseline = torch.mm(a.to(dtype=torch.float32),
                        b.to(dtype=torch.float32)).to(dtype=torch.bfloat16)

    assert torch.allclose(out, baseline, rtol=1e-1, atol=1e0)

#Test to make sure cuda graphs work
class CutlassLayer(torch.nn.Module):

    def __init__(self, b, out_dtype):
        super().__init__()
        self.b = b
        self.out_dtype = out_dtype

    def forward(self, a):
        return ops.cutlass_gemm(a, self.b, self.out_dtype)


def test_cutlass_cuda_graph():
    m, n, k = 512, 512, 512

    a = to_int8(torch.randn((m, k), device="cuda"))
    b = to_int8(torch.randn((n, k), device="cuda").t())

#Construct a trivial model with a single layer that calls a CUTLASS kernel
    model = CutlassLayer(b, torch.bfloat16)

#Run the model with a cuda graph
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = model(a)
    out.zero_()
    g.replay()

    baseline = torch.mm(a.to(dtype=torch.float32),
                        b.to(dtype=torch.float32)).to(torch.bfloat16)
    assert torch.allclose(out, baseline, rtol=1e-1, atol=1e0)
