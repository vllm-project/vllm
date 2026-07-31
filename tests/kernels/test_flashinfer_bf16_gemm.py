# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers import utils as layer_utils


@pytest.fixture(scope="module", autouse=True)
def require_flashinfer_bf16_cutedsl() -> None:
    if not torch.accelerator.is_available():
        pytest.skip("CUDA is required")
    if not layer_utils._is_flashinfer_cutedsl_bf16_supported():
        pytest.skip("FlashInfer BF16 cute-dsl backend is unavailable")


@pytest.mark.parametrize("n,k", [(1024, 1024), (2048, 3072)])
@pytest.mark.parametrize(
    "m,use_bias,pdl",
    [
        (1, False, False),
        (8, True, False),
        (16, False, True),
        (32, True, True),
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
        x, weight, bias, pdl, "cute-dsl"
    )
    expected = F.linear(x, weight, bias)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-1)


def test_flashinfer_bf16_custom_op_dynamic_compile() -> None:
    n, k = 1024, 1024
    layer = torch.nn.Identity()
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1

    def gemm(x: torch.Tensor) -> torch.Tensor:
        return layer_utils.cuda_flashinfer_bf16_gemm(
            layer,
            x,
            weight,
            backend="cute-dsl",
            pdl=False,
        )

    compiled_gemm = torch.compile(gemm, dynamic=True, fullgraph=True)
    for m in (8, 33):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        torch.testing.assert_close(
            compiled_gemm(x),
            F.linear(x, weight),
            rtol=2e-2,
            atol=2e-1,
        )


def test_flashinfer_bf16_custom_op_cuda_graph_replay() -> None:
    m, n, k = 8, 1024, 1024
    layer = torch.nn.Identity()
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) * 0.1

    def gemm() -> torch.Tensor:
        return layer_utils.cuda_flashinfer_bf16_gemm(
            layer,
            x,
            weight,
            bias,
            backend="cute-dsl",
            pdl=False,
        )

    gemm()
    torch.accelerator.synchronize()

    graph = torch.accelerator.Graph()
    with graph:
        actual = gemm()
    graph.replay()

    torch.testing.assert_close(actual, F.linear(x, weight, bias), rtol=2e-2, atol=2e-1)
