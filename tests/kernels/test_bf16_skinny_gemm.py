# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the explicitly-invoked CuTe DSL BF16 skinny GEMM.

The kernel has no automatic shape dispatch: callers pick it deliberately and
pass their own :class:`SkinnyGemmConfig` (or let the heuristic pick one), so
these tests drive it directly rather than through a model.
"""

import pytest
import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    ShapeDynamicSkinnyGemm,
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)


@pytest.fixture(scope="module", autouse=True)
def require_cutedsl() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not shape_dynamic_skinny_gemm.is_available():
        pytest.skip("CuTe DSL is not available")


def _operands(
    num_tokens: int, n: int, k: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    return x, weight


def _assert_close(output: torch.Tensor, reference: torch.Tensor) -> None:
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-1)


# A spread over the config space: wide/narrow blocks, k_unroll, vector_width,
# multi-output blocks and a static-K specialization.
CONFIG_CASES = [
    (3072, 7168, SkinnyGemmConfig(1, 224, 3, 4)),
    (3072, 7168, SkinnyGemmConfig(2, 128, 3, 2)),
    (7168, 3072, SkinnyGemmConfig(2, 32, 4, 4)),
    (7168, 4224, SkinnyGemmConfig(1, 96, 4, 2, vector_width=4)),
    (7168, 14336, SkinnyGemmConfig(1, 256, 2, vector_width=4, static_k=14336)),
    (1536, 7168, SkinnyGemmConfig(16, 128, 1, 4, vector_width=4)),
]


@pytest.mark.parametrize("n,k,config", CONFIG_CASES)
def test_explicit_config(n: int, k: int, config: SkinnyGemmConfig) -> None:
    x, weight = _operands(config.num_rows, n, k, seed=42)

    output = shape_dynamic_skinny_gemm(x, weight, config)

    _assert_close(output, x.float() @ weight.float().t())


@pytest.mark.parametrize("num_tokens", range(1, 17))
def test_heuristic_config_all_supported_token_counts(num_tokens: int) -> None:
    n, k = 64, 512
    x, weight = _operands(num_tokens, n, k, seed=42 + num_tokens)

    output = shape_dynamic_skinny_gemm(x, weight)

    _assert_close(output, x.float() @ weight.float().t())


@pytest.mark.parametrize("num_tokens", range(1, 17))
def test_residual_epilogue_all_supported_token_counts(num_tokens: int) -> None:
    n, k = 64, 512
    x, weight = _operands(num_tokens, n, k, seed=7 + num_tokens)
    residual = torch.randn(num_tokens, n, dtype=torch.bfloat16, device="cuda")
    config = ShapeDynamicSkinnyGemm._config(num_tokens, n, k)

    output = shape_dynamic_skinny_gemm(x, weight, config, residual)

    _assert_close(output, x.float() @ weight.float().t() + residual.float())


@pytest.mark.parametrize("use_residual", [False, True])
def test_cuda_graph_capture(use_residual: bool) -> None:
    n, k, num_tokens = 7168, 3584, 2
    x, weight = _operands(num_tokens, n, k, seed=11)
    residual = (
        torch.randn(num_tokens, n, dtype=torch.bfloat16, device="cuda")
        if use_residual
        else None
    )
    config = ShapeDynamicSkinnyGemm._config(num_tokens, n, k)
    shape_dynamic_skinny_gemm(x, weight, config, residual)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = shape_dynamic_skinny_gemm(x, weight, config, residual)
    graph.replay()
    torch.accelerator.synchronize()

    reference = x.float() @ weight.float().t()
    if residual is not None:
        reference = reference + residual.float()
    _assert_close(output, reference)


def test_rejects_unsupported_token_count() -> None:
    x, weight = _operands(17, 64, 512, seed=1)

    with pytest.raises(ValueError, match="1 <= M <= 16"):
        shape_dynamic_skinny_gemm(x, weight)


def test_rejects_noncontiguous_activation() -> None:
    storage = torch.randn(2, 512 + 16, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(ValueError, match="contiguous"):
        shape_dynamic_skinny_gemm(storage[:, :512], weight)
