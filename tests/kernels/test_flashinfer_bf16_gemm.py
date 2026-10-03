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


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
def test_flashinfer_tgv_auto_dispatch(m: int, monkeypatch) -> None:
    if not layer_utils._FLASHINFER_BF16_BACKENDS["auto"].is_supported():
        pytest.skip("Automatic TGV dispatch requires SM100 and FlashInfer TGV")

    torch.manual_seed(0)
    x = torch.randn(m, 6144, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(5120, 6144, device="cuda", dtype=torch.bfloat16) * 0.1
    backends = []
    mm = layer_utils.flashinfer_bf16_mm

    def record_backend(*args):
        backends.append(args[-1])
        return mm(*args)

    monkeypatch.setattr(layer_utils, "flashinfer_bf16_mm", record_backend)
    gemm = layer_utils.dispatch_unquantized_gemm()
    actual = gemm(None, x, weight)
    assert backends == (["tgv"] if m <= 8 else [])
    torch.testing.assert_close(actual, F.linear(x, weight), rtol=2e-2, atol=2e-2)

    if m == 4:
        compiled = torch.compile(gemm, backend="eager", fullgraph=True)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            compiled(None, x, weight)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = compiled(None, x, weight)
        x.add_(0.01)
        graph.replay()
        torch.testing.assert_close(captured, F.linear(x, weight), rtol=2e-2, atol=2e-2)
