# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LUT-B weight x MXFP8 activation linear (emulated and Rubin SM107)."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.online.lut_b import (
    apply_lut_b_linear_emulation,
    apply_lut_b_linear_sm107,
)
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    dequantize_lut_b,
    lut_b_sm107_supported,
    lut_b_to_sm107_tiles,
    mxfp8_qdq,
    quantize_lut_b,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform

requires_gpu = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like GPU"
)
requires_sm107 = pytest.mark.skipif(
    not lut_b_sm107_supported(),
    reason="requires a Rubin (SM107) GPU and the compiled LUT-B kernel",
)


def _make_problem(
    shape: tuple[int, ...], n: int, k: int, dtype: torch.dtype, with_bias: bool
):
    x = torch.randn(*shape, k, device="cuda", dtype=dtype)
    weight = torch.randn(n, k, device="cuda", dtype=dtype) / k**0.5
    bias = torch.randn(n, device="cuda", dtype=dtype) if with_bias else None
    return x, quantize_lut_b(weight), bias


@requires_gpu
@pytest.mark.parametrize("shape", [(1,), (37,), (2, 5)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_emulation_matches_reference(shape: tuple[int, ...], with_bias: bool):
    torch.manual_seed(0)
    n, k = 72, 192
    x, q, bias = _make_problem(shape, n, k, torch.bfloat16, with_bias)
    weight = dequantize_lut_b(*q, torch.bfloat16)

    actual = apply_lut_b_linear_emulation(x, weight, bias)

    expected = F.linear(mxfp8_qdq(x).float(), weight.float())
    if bias is not None:
        expected += bias.float()
    assert actual.shape == (*shape, n)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)


@requires_sm107
@pytest.mark.parametrize("shape", [(1,), (129,), (300,), (3, 50)])
@pytest.mark.parametrize("n,k", [(128, 128), (384, 512)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_sm107_matches_decoded_reference(
    shape: tuple[int, ...], n: int, k: int, dtype: torch.dtype, with_bias: bool
):
    """The native GEMM against an FP64 product of the decoded operands."""
    torch.manual_seed(1)
    x, q, bias = _make_problem(shape, n, k, dtype, with_bias)
    tiles = lut_b_to_sm107_tiles(*(t[None] for t in q))

    actual = apply_lut_b_linear_sm107(x, tiles, bias)

    x_q, x_s = mxfp8_e4m3_quantize(x.reshape(-1, k))
    x_ref = dequant_mxfp8_to_bf16(x_q, x_s.view(x_q.size(0), -1)).double()
    expected = x_ref @ dequantize_lut_b(*q, torch.float64).T
    if bias is not None:
        expected += bias.double()
    assert actual.shape == (*shape, n) and actual.dtype == dtype
    torch.testing.assert_close(
        actual.reshape(-1, n).double(), expected, rtol=1e-2, atol=1e-2
    )


@requires_sm107
def test_sm107_matches_emulation():
    torch.manual_seed(2)
    n, k = 256, 384
    x, q, bias = _make_problem((77,), n, k, torch.bfloat16, True)
    native = apply_lut_b_linear_sm107(
        x, lut_b_to_sm107_tiles(*(t[None] for t in q)), bias
    )
    emulated = apply_lut_b_linear_emulation(
        x, dequantize_lut_b(*q, torch.bfloat16), bias
    )
    torch.testing.assert_close(native.float(), emulated.float(), rtol=2e-2, atol=2e-2)
