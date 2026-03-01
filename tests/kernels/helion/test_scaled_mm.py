# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the scaled_mm helion kernel

Run `pytest tests/kernels/helion/test_scaled_mm.py`.
"""

import importlib

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


device = "cuda"

helion_scaled_mm_module = importlib.import_module("vllm.kernels.helion.scaled_mm")
helion_scaled_mm = helion_scaled_mm_module.scaled_mm


def torch_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    out = scale_a * out
    out = scale_b.T * out
    out = out.to(out_dtype)
    if bias is not None:
        out = out + bias

    return out


def get_8bit_types():
    types = [torch.int8]
    if current_platform.supports_fp8():
        types.append(current_platform.fp8_dtype())
    return types


MNK_FACTORS = [
    (1, 256, 128),
    (33, 256, 496),
    (64, 971, 1024),
    (64, 20486, 128),
    (512, 256, 496),
    (512, 20486, 1024),
]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
@pytest.mark.parametrize("M,N,K", MNK_FACTORS)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
@pytest.mark.parametrize("in_dtype", get_8bit_types())
@pytest.mark.parametrize("use_scalar_scale_a", [True, False])
@pytest.mark.parametrize("use_scalar_scale_b", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_scaled_mm(
    M, N, K, in_dtype, out_dtype, use_scalar_scale_a, use_scalar_scale_b, use_bias
):
    is_floating_point_type = lambda t: torch.tensor([1, 1], dtype=t).is_floating_point()

    set_random_seed(0)

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
        a = (0.25 * torch.rand((M, K), dtype=torch.float32, device=device)).to(in_dtype)
        b = (0.25 * torch.rand((K, N), dtype=torch.float32, device=device)).to(in_dtype)
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
        scale_b = 0.25 * torch.rand((N, 1), device=device)

    bias = None
    if use_bias:
        bias = torch.rand((N,), device=device, dtype=out_dtype)

    c_check = helion_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    c_actual = torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    torch.testing.assert_close(c_check, c_actual, rtol=1e-1, atol=1e-1)
