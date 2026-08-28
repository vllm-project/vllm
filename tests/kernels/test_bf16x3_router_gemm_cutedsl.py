# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the experimental SM100 BF16x3 router GEMM."""

import pytest
import torch

from vllm.utils.import_utils import has_cutedsl


def _requires_sm100_cutedsl():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip("bf16x3 router GEMM requires SM100-class GPU")
    if not has_cutedsl():
        pytest.skip("cutedsl (cutlass) not installed")


@pytest.mark.parametrize(
    ("num_tokens", "hidden_dim", "num_experts"),
    [(48, 6144, 128), (96, 3072, 256), (129, 3072, 17)],
)
def test_bf16x3_router_gemm_matches_reference(
    num_tokens: int, hidden_dim: int, num_experts: int
):
    _requires_sm100_cutedsl()
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl import (  # noqa: E501
        bf16x3_router_gemm,
    )

    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(num_experts, hidden_dim, dtype=torch.float32, device="cuda")
    # Match the observed router weight scale
    w *= 0.053
    out = bf16x3_router_gemm(x, w)
    ref = torch.nn.functional.linear(x.float(), w)

    assert out.shape == (num_tokens, num_experts)
    assert out.dtype == torch.float32
    assert torch.mean(torch.abs(out - ref)).item() < 5e-6
