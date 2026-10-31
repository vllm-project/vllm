# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers import utils as layer_utils
from vllm.utils.flashinfer import is_flashinfer_cutedsl_bf16_gemm_supported


@pytest.fixture(scope="module", autouse=True)
def require_flashinfer_bf16_cutedsl() -> None:
    if not torch.accelerator.is_available():
        pytest.skip("CUDA is required")
    if not is_flashinfer_cutedsl_bf16_gemm_supported():
        pytest.skip("FlashInfer BF16 cute-dsl backend is unavailable")


@pytest.mark.parametrize("n,k", [(1024, 1024), (2048, 3072)])
@pytest.mark.parametrize(
    "m,use_bias,pdl",
    [
        (1, False, False),
        (8, True, False),
        (16, False, True),
        (32, True, True),
        (64, True, True),
        (512, False, False),
    ],
)
def test_flashinfer_bf16_cutedsl_correctness(
    m: int,
    n: int,
    k: int,
    use_bias: bool,
    pdl: bool,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    bias = (
        torch.randn(n, device="cuda", dtype=torch.bfloat16) * 0.1 if use_bias else None
    )

    actual = layer_utils.cuda_flashinfer_bf16_gemm_impl(
        x, weight, bias, pdl, "flashinfer_cutedsl"
    )
    expected = F.linear(x, weight, bias)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize(
    "m,n,k", [(1, 1024, 1024), (64, 2048, 3072), (512, 1024, 1000)]
)
@pytest.mark.parametrize("tune", [False, True])
def test_flashinfer_bf16_auto_correctness(m: int, n: int, k: int, tune: bool) -> None:
    """--linear-backend=flashinfer dispatches to mm_bf16(backend="auto") without
    pdl, with and without autotuning across FlashInfer's BF16 backends."""
    from flashinfer import autotune

    gemm = layer_utils.dispatch_unquantized_gemm("flashinfer")
    # A plain function here means the F.linear fallback.
    assert isinstance(gemm, functools.partial)
    assert gemm.keywords == {"vllm_backend": "flashinfer", "pdl": None}

    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1

    with autotune(tune):
        actual = gemm(None, x, weight)
    expected = F.linear(x, weight)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-1)
