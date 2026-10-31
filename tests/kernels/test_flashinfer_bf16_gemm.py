# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


@pytest.fixture
def pinned_gemm():
    if not layer_utils._use_sm103_pinned_bf16_gemm():
        pytest.skip("Pinned BF16 GEMMs require SM103 and FlashInfer")
    gemm = layer_utils.dispatch_unquantized_gemm()
    assert gemm is layer_utils.cuda_pinned_bf16_gemm
    return gemm


@pytest.mark.parametrize("n,k", list(layer_utils._SM103_PINNED_BF16_GEMMS))
def test_sm103_pinned_bf16_gemm_correctness(
    pinned_gemm, n: int, k: int, monkeypatch
) -> None:
    """Every pinned (N, K, M) runs its kernel and matches F.linear; other M
    fall back to F.linear."""
    torch.manual_seed(0)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    planned = sorted(layer_utils._SM103_PINNED_BF16_GEMMS[(n, k)])
    pinned_runs = []
    get_runner = layer_utils._pinned_bf16_gemm_runner

    def record_runner(n: int, k: int, rows: int):
        runner = get_runner(n, k, rows)
        if runner is None:
            return None

        def run(*args, **kwargs):
            pinned_runs.append(rows)
            return runner(*args, **kwargs)

        return run

    monkeypatch.setattr(layer_utils, "_pinned_bf16_gemm_runner", record_runner)
    for m in [*planned, 5]:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        torch.testing.assert_close(
            pinned_gemm(None, x, weight), F.linear(x, weight), rtol=2e-2, atol=2e-2
        )
    assert pinned_runs == planned


@pytest.mark.parametrize("m", [1, 4, 16])
def test_sm103_pinned_bf16_gemm_cuda_graph(pinned_gemm, m: int) -> None:
    """The direct, warp and cluster kernels replay from a captured CUDA graph."""
    torch.manual_seed(0)
    weight = torch.randn(34816, 5120, device="cuda", dtype=torch.bfloat16) * 0.1
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16) * 0.1

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        pinned_gemm(None, x, weight)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = pinned_gemm(None, x, weight)
    x.add_(0.01)
    graph.replay()
    torch.testing.assert_close(captured, F.linear(x, weight), rtol=2e-2, atol=2e-2)
