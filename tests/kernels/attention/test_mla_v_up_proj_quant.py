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


@pytest.mark.parametrize(
    "num_heads,kv_lora_rank,v_head_dim", [(16, 512, 128), (128, 512, 128)]
)
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


def test_v_up_proj_fused_quant_writes_through_to_caller_buffer():
    """Reproduces the exact view/slice arithmetic MLAAttention.forward_impl
    uses when it fuses this kernel into the MQA call site: a 2D [tokens,
    N*V] destination buffer, sliced to the decode rows and viewed as
    [B, N, V] for the kernel's `out=`. This only proves something if the
    write lands in the original buffer's memory, not a disconnected copy.
    """
    torch.manual_seed(0)
    device = "cuda"
    num_heads, kv_lora_rank, v_head_dim = 16, 512, 128
    num_mqa_tokens, num_mha_tokens = 5, 3
    num_total_tokens = num_mqa_tokens + num_mha_tokens

    attn_out = torch.randn(
        num_mqa_tokens,
        num_heads * kv_lora_rank,
        dtype=torch.bfloat16,
        device=device,
    )
    w = torch.randn(
        num_heads, kv_lora_rank, v_head_dim, dtype=torch.bfloat16, device=device
    ) * 0.1
    output_scale = torch.tensor([0.05], dtype=torch.float32, device=device)

    # Mirrors forward_impl's `quant_output` buffer: real [tokens, N*V] FP8
    # storage the caller ultimately returns, pre-filled with a sentinel so a
    # no-op write (or writing to the wrong rows) is visible.
    quant_output = torch.full(
        (num_total_tokens, num_heads * v_head_dim),
        -1.0,
        dtype=FP8_DTYPE,
        device=device,
    )

    x_t = attn_out.view(-1, num_heads, kv_lora_rank).transpose(0, 1)
    quant_out_view = quant_output[:num_mqa_tokens].view(
        num_mqa_tokens, num_heads, v_head_dim
    )
    v_up_proj_fp8_static_quant(x_t, w, output_scale, out=quant_out_view)

    expected_mqa_rows = _reference(x_t, w, output_scale).reshape(
        num_mqa_tokens, num_heads * v_head_dim
    )
    # The write must be visible on `quant_output` itself (view, not a copy).
    # Cosine similarity, not assert_close: both sides are already FP8-rounded,
    # and the hardware quant kernel's rounding can differ from this test's
    # hand-rolled reference at individual bucket boundaries (same reasoning
    # as test_v_up_proj_fp8_static_quant_matches_reference above).
    diff = _cosine_diff(
        quant_output[:num_mqa_tokens].float(), expected_mqa_rows.float()
    )
    assert diff < 1e-3, f"cosine diff {diff} too high vs reference"
    # Rows past the decode slice (the would-be mha rows) must be untouched.
    assert (quant_output[num_mqa_tokens:] == -1.0).all()
