"""Tests for the scaled_mm_triton kernel

Run `pytest tests/kernels/test_scaled_mm_triton.py`.
"""
from typing import Optional, Type

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.scaled_mm_triton import scaled_mm_triton # noqa
from vllm.utils import seed_everything

device = "cuda"


def scaled_mm_torch(a: torch.Tensor,
                    b: torch.Tensor,
                    scale_a: torch.Tensor,
                    scale_b: torch.Tensor,
                    out_dtype: Type[torch.dtype],
                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    out = scale_a * out
    out = scale_b.T * out
    out = out.to(out_dtype)
    if bias is not None:
        out = out + bias

    return out


@pytest.mark.parametrize("M", [1, 16, 32, 64, 128, 256, 512, 222, 33, 1])
@pytest.mark.parametrize("N", [2048, 8192, 16384, 256, 1024])
@pytest.mark.parametrize("K", [128, 496, 1024])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("in_dtype", [torch.int8])
@pytest.mark.parametrize("use_scalar_scale_a", [True, False])
@pytest.mark.parametrize("use_scalar_scale_b", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_scaled_mm(M, N, K, in_dtype, out_dtype, use_scalar_scale_a,
                   use_scalar_scale_b, use_bias):
    is_floating_point_type = lambda t: torch.tensor([1, 1], dtype=t
                                                    ).is_floating_point()

    seed_everything(0)

    # NOTE: There are cases, where if the matrix is large enough, an output
    # like 65504.4 can be produced, and can easily turn into inf when
    # multiplied when using float16/bfloat16.  This means one function, e.g.,
    # testing function, and another function, e.g. golden function, can
    # produce a non-inf value while the other produces an inf value, and
    # will cause assert_close/allclose to fail, even though if overflow
    # wouldn't have occurred, the values would have been "close."
    #
    # So, the values here are kept small enough to avoid this situation.
    if is_floating_point_type(in_dtype):
        a = (0.25 * torch.rand(
            (M, K), dtype=torch.float32, device=device)).to(in_dtype)
        b = (0.25 * torch.rand(
            (K, N), dtype=torch.float32, device=device)).to(in_dtype)
    else:
        a = torch.randint(-32, 32, (M, K), dtype=in_dtype, device=device)
        b = torch.randint(-32, 32, (K, N), dtype=in_dtype, device=device)

    if use_scalar_scale_a:
        scale_a = torch.rand((1, 1), device=device)
    else:
        scale_a = 0.25 * torch.rand((M, 1), device=device)

    if use_scalar_scale_b:
        scale_b = torch.rand((1, 1), device=device)
    else:
        scale_b = 0.25 * torch.rand((1, 1), device=device)

    bias = None
    if use_bias:
        bias = torch.rand((N, ), device=device, dtype=out_dtype)

    c_check = scaled_mm_triton(a, b, scale_a, scale_b, out_dtype, bias)

    a_cpu = a.cpu()
    b_cpu = b.cpu()
    scale_a_cpu = scale_a.cpu()
    scale_b_cpu = scale_b.cpu()
    bias_cpu = None if bias is None else bias.cpu()

    c_actual = scaled_mm_torch(a_cpu, b_cpu, scale_a_cpu, scale_b_cpu,
                               out_dtype, bias_cpu)

    c_check_cpu = c_check.cpu()
    torch.testing.assert_close(c_check_cpu, c_actual, rtol=1e-1, atol=1e-1)
