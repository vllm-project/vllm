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
def pinned_gate_up_gemm():
    if not layer_utils._use_sm103_pinned_splitk_bf16_gemm():
        pytest.skip("Pinned split-K tactics require SM103 and FlashInfer")
    gemm = layer_utils.dispatch_unquantized_gemm()
    assert gemm is layer_utils.cuda_pinned_splitk_bf16_gemm
    torch.manual_seed(0)
    weight = torch.randn(34816, 5120, device="cuda", dtype=torch.bfloat16) * 0.1
    return gemm, weight


@pytest.mark.parametrize("m", [8, 16])
def test_sm103_pinned_splitk_gate_up_correctness(
    pinned_gate_up_gemm, m: int, monkeypatch
) -> None:
    """Only M=16 runs the pinned kernel; both paths match F.linear."""
    gemm, weight = pinned_gate_up_gemm
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16) * 0.1
    pinned_runs = []
    get_runner = layer_utils._pinned_splitk_bf16_runner

    def record_runner(n: int, k: int, rows: int):
        runner = get_runner(n, k, rows)
        if runner is None:
            return None

        def run(*args):
            pinned_runs.append(rows)
            return runner(*args)

        return run

    monkeypatch.setattr(layer_utils, "_pinned_splitk_bf16_runner", record_runner)
    actual = gemm(None, x, weight)

    assert pinned_runs == ([16] if m == 16 else [])
    torch.testing.assert_close(actual, F.linear(x, weight), rtol=2e-2, atol=2e-2)


def test_sm103_pinned_splitk_dispatch_under_fullgraph_compile(
    pinned_gate_up_gemm,
) -> None:
    """Callers that dispatch inside compiled forward code still trace."""
    _, weight = pinned_gate_up_gemm
    x = torch.randn(16, 5120, device="cuda", dtype=torch.bfloat16) * 0.1

    @torch.compile(fullgraph=True)
    def forward(x: torch.Tensor) -> torch.Tensor:
        return layer_utils.dispatch_unquantized_gemm()(None, x, weight)

    torch.testing.assert_close(forward(x), F.linear(x, weight), rtol=2e-2, atol=2e-2)


def test_sm103_pinned_splitk_gate_up_cuda_graph(pinned_gate_up_gemm) -> None:
    """The pinned kernel replays correctly from a captured CUDA graph."""
    gemm, weight = pinned_gate_up_gemm
    x = torch.randn(16, 5120, device="cuda", dtype=torch.bfloat16) * 0.1

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        gemm(None, x, weight)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = gemm(None, x, weight)
    x.add_(0.01)
    graph.replay()
    torch.testing.assert_close(captured, F.linear(x, weight), rtol=2e-2, atol=2e-2)
