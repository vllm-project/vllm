# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    _FP32_ONE_MINUS_EPS,
    _FP32_TINY,
    _FP64_ONE_MINUS_EPS,
)

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for Gumbel sampler tests", allow_module_level=True)


@triton.jit
def _gumbel_nonfinite_logits_regression_kernel(
    out_value,
    out_is_nan,
    USE_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < BLOCK_SIZE

    if USE_FP64:
        logits = tl.full((BLOCK_SIZE,), float("-inf"), tl.float64)
        # Deterministically reproduce the upper-endpoint uniform draw that
        # previously made Gumbel noise infinite.
        u = tl.full((BLOCK_SIZE,), 1.0, tl.float64)
        u = tl.minimum(u, _FP64_ONE_MINUS_EPS)
    else:
        logits = tl.full((BLOCK_SIZE,), float("-inf"), tl.float32)
        u = tl.full((BLOCK_SIZE,), 1.0, tl.float32)
        u = tl.minimum(tl.maximum(u, _FP32_TINY), _FP32_ONE_MINUS_EPS)

    gumbel_noise = -tl.log(-tl.log(u))
    finite = logits > float("-inf")
    logits = tl.where(mask & finite, logits + gumbel_noise, float("-inf"))

    value = tl.max(logits, axis=0)
    tl.store(out_value, value)
    tl.store(out_is_nan, (value != value).to(tl.int32))


@pytest.mark.parametrize("use_fp64", [False, True])
def test_gumbel_noise_does_not_turn_negative_inf_logits_into_nan(use_fp64: bool):
    value_dtype = torch.float64 if use_fp64 else torch.float32
    value = torch.empty((), device="cuda", dtype=value_dtype)
    is_nan = torch.empty((), device="cuda", dtype=torch.int32)

    _gumbel_nonfinite_logits_regression_kernel[(1,)](
        value,
        is_nan,
        USE_FP64=use_fp64,
        BLOCK_SIZE=1024,
        num_warps=1,
    )
    torch.cuda.synchronize()

    assert not bool(is_nan.item())
    assert torch.isneginf(value).item()
