# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the fused MLA up-projection + FP8 static quant kernel.

Compares the fused kernel (vllm.v1.attention.ops.mla_v_up_proj_quant) against
the unfused reference path it is meant to replace: a plain batched matmul in
fp32, followed by the same static-scale FP8 quantization convention used by
QuantFP8 elsewhere in vLLM (quantize = clamp(x / scale), dequant = fp8 * scale).
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.mla_v_up_proj_quant import (
    FP8_DTYPE,
    FP8_MAX,
    FP8_MIN,
    v_up_proj_fp8_static_quant,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton MLA kernels are CUDA-only"
)


def _reference(
    x: torch.Tensor, w: torch.Tensor, output_scale: torch.Tensor
) -> torch.Tensor:
    """Unfused reference: fp32 batched matmul, then static FP8 quant."""
    acc = torch.bmm(x.float(), w.float())  # [N, B, V]
    quantized = (acc / output_scale).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    return quantized.transpose(0, 1).contiguous()  # [B, N, V]


def _cosine_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return 1 - 2 * (a * b).sum().item() / max((a * a + b * b).sum().item(), 1e-12)


@pytest.mark.parametrize("num_heads,kv_lora_rank,v_head_dim", [(16, 512, 128), (128, 512, 128)])
@pytest.mark.parametrize("batch_size", [1, 7, 64])
def test_v_up_proj_fp8_static_quant_matches_reference(
    num_heads: int, kv_lora_rank: int, v_head_dim: int, batch_size: int
):
    torch.manual_seed(0)
    device = "cuda"

    x = torch.randn(
        num_heads, batch_size, kv_lora_rank, dtype=torch.bfloat16, device=device
    )
    w = torch.randn(
        num_heads, kv_lora_rank, v_head_dim, dtype=torch.bfloat16, device=device
    ) * 0.1
    # A representative static scale: large enough that most values don't
    # saturate FP8's narrow dynamic range.
    output_scale = torch.tensor([0.05], dtype=torch.float32, device=device)

    expected = _reference(x, w, output_scale)
    # Exercises whichever backend this GPU's compute capability selects
    # (portable matmul+quant on Ampere, the fused Triton kernel on sm_90+).
    actual = v_up_proj_fp8_static_quant(x, w, output_scale)

    assert actual.shape == expected.shape
    assert actual.dtype == FP8_DTYPE
    diff = _cosine_diff(expected.float(), actual.float())
    assert diff < 1e-3, f"cosine diff {diff} too high vs reference"
