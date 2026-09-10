# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_swiglu_gemm import fused_swiglu_gemm
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(120),
    reason="the fused SwiGLU GEMM is only selected on SM120",
)


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    gate, up = F.linear(x, weight).chunk(2, dim=-1)
    return F.silu(gate) * up


@pytest.mark.parametrize("num_tokens", [1024, 4096, 8192])
def test_fused_swiglu_gemm(num_tokens: int) -> None:
    torch.manual_seed(0)
    x = torch.randn((num_tokens, 1024), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((7168, 1024), device="cuda", dtype=torch.bfloat16)

    output = fused_swiglu_gemm(x, weight)

    torch.testing.assert_close(output, _reference(x, weight), rtol=0.02, atol=0.02)


@pytest.mark.parametrize("num_tokens", [1, 128, 2048])
def test_fused_swiglu_gemm_falls_back_for_other_token_counts(
    num_tokens: int,
) -> None:
    torch.manual_seed(0)
    x = torch.randn((num_tokens, 1024), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((7168, 1024), device="cuda", dtype=torch.bfloat16)

    output = fused_swiglu_gemm(x, weight)

    torch.testing.assert_close(output, _reference(x, weight), rtol=0, atol=0)


def test_fused_swiglu_gemm_falls_back_for_noncontiguous_input() -> None:
    torch.manual_seed(0)
    x = torch.randn((1024, 2048), device="cuda", dtype=torch.bfloat16)[:, ::2]
    weight = torch.randn((7168, 1024), device="cuda", dtype=torch.bfloat16)
    assert not x.is_contiguous()

    output = fused_swiglu_gemm(x, weight)

    torch.testing.assert_close(output, _reference(x, weight), rtol=0, atol=0)
