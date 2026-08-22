# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The unquantized BF16 linear path routes through FlashInfer's ``mm_bf16``."""

import pytest
import torch

from vllm.model_executor.layers import utils
from vllm.model_executor.layers.utils import (
    _use_flashinfer_bf16_gemm,
    default_unquantized_gemm,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_non_bf16_stays_on_torch_linear(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    """Only BF16 is eligible; mm_bf16 rejects every other dtype."""
    monkeypatch.setattr(utils, "_flashinfer_bf16_gemm_available", lambda: True)
    x = torch.empty(2, 8, dtype=dtype)
    weight = torch.empty(4, 8, dtype=dtype)

    assert not _use_flashinfer_bf16_gemm(x, weight, None)


def test_non_contiguous_weight_stays_on_torch_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``weight.t()`` is only the column-major operand for a packed weight."""
    monkeypatch.setattr(utils, "_flashinfer_bf16_gemm_available", lambda: True)
    x = torch.empty(2, 8, dtype=torch.bfloat16)
    weight = torch.empty(8, 4, dtype=torch.bfloat16).t()

    assert not weight.is_contiguous()
    assert not _use_flashinfer_bf16_gemm(x, weight, None)


def test_batch_invariant_stays_on_torch_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils, "_flashinfer_bf16_gemm_available", lambda: True)
    monkeypatch.setattr(utils.envs, "VLLM_BATCH_INVARIANT", True)
    x = torch.empty(2, 8, dtype=torch.bfloat16)
    weight = torch.empty(4, 8, dtype=torch.bfloat16)

    assert not _use_flashinfer_bf16_gemm(x, weight, None)


@pytest.mark.parametrize("m,k,n", [(1, 7168, 3072), (7, 1023, 129), (256, 4096, 512)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_matches_torch_linear(m: int, k: int, n: int, with_bias: bool) -> None:
    if not utils._flashinfer_bf16_gemm_available():
        pytest.skip("FlashInfer mm_bf16 is unavailable on this platform")
    torch.manual_seed(42)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    # A large bias makes a backend that silently drops it fail the comparison.
    bias = (
        torch.full((n,), 100.0, dtype=torch.bfloat16, device="cuda")
        if with_bias
        else None
    )
    assert _use_flashinfer_bf16_gemm(x, weight, bias)

    output = default_unquantized_gemm(None, x, weight, bias)

    reference = torch.nn.functional.linear(x, weight, bias)
    torch.testing.assert_close(output.float(), reference.float(), rtol=2e-2, atol=3e-1)


def test_preserves_leading_dims() -> None:
    if not utils._flashinfer_bf16_gemm_available():
        pytest.skip("FlashInfer mm_bf16 is unavailable on this platform")
    torch.manual_seed(42)
    x = torch.randn(2, 3, 1024, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(512, 1024, dtype=torch.bfloat16, device="cuda")

    output = default_unquantized_gemm(None, x, weight)

    assert output.shape == (2, 3, 512)
    torch.testing.assert_close(
        output.float(),
        torch.nn.functional.linear(x, weight).float(),
        rtol=2e-2,
        atol=3e-1,
    )


def test_cuda_graph_capture() -> None:
    """Decode replays the GEMM from a graph, so capture must not fail."""
    if not utils._flashinfer_bf16_gemm_available():
        pytest.skip("FlashInfer mm_bf16 is unavailable on this platform")
    torch.manual_seed(42)
    x = torch.randn(2, 4096, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
    default_unquantized_gemm(None, x, weight)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = default_unquantized_gemm(None, x, weight)
    graph.replay()
    torch.accelerator.synchronize()

    torch.testing.assert_close(
        output.float(),
        torch.nn.functional.linear(x, weight).float(),
        rtol=2e-2,
        atol=3e-1,
    )
