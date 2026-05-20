# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for `fused_shared_expert_gate`.

Run `pytest tests/kernels/moe/test_fused_shared_expert_gate.py`.

The Triton fusion replaces the three-kernel `F.sigmoid(linear(x)) * out`
tail of `Qwen2MoeMLP.forward` / `Qwen3MoeMLP.forward`. This test
parametrizes over real Qwen3-Next-style shapes (`K=2048`, hidden=2048)
plus a smaller config and mask-boundary token counts (`N=1`, `N=7`) to
exercise the within-block masking path.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.shared_expert_gate import (
    fused_shared_expert_gate,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_shared_expert_gate requires a Triton-capable GPU (CUDA or ROCm).",
)


def _reference(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    return F.sigmoid(F.linear(x, weight)) * out


@pytest.mark.parametrize("num_tokens", [1, 7, 33, 1024, 7177, 8192])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_shared_expert_gate_matches_reference(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn((num_tokens, hidden_size), device=device, dtype=dtype)
    weight = torch.randn((1, hidden_size), device=device, dtype=dtype)
    out = torch.randn((num_tokens, hidden_size), device=device, dtype=dtype)

    expected = _reference(x, weight, out)
    actual = fused_shared_expert_gate(x, weight, out)

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    # Tolerance matches the existing pattern in tests/kernels/ for bf16
    # row-fused ops; fp16 comfortably fits the same bound.
    torch.testing.assert_close(actual, expected, atol=3.125e-2, rtol=2e-2)


def test_fused_shared_expert_gate_fallback_on_unsupported_shape():
    """A non-2D `x` must fall back to the PyTorch reference."""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    # 3D input -- the Triton kernel is 2D-only, so the wrapper must
    # dispatch to the PyTorch reference path. `F.sigmoid(F.linear(...))`
    # broadcasts the `[B, N, 1]` gate against the `[B, N, K]` output, so
    # the reference expression is well-defined and we can compare equality.
    x = torch.randn((2, 16, 2048), device=device, dtype=dtype)
    out = torch.randn((2, 16, 2048), device=device, dtype=dtype)
    weight = torch.randn((1, 2048), device=device, dtype=dtype)

    expected = _reference(x, weight, out)
    actual = fused_shared_expert_gate(x, weight, out)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=3.125e-2, rtol=2e-2)
