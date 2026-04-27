# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for fused_indexer_q_rope_quant.

Compares the fused Triton kernel against the unfused reference flow used by
the DeepseekV4 indexer in model_tracking:
    q_rot = ops.rotary_embedding(positions, q, None, head_dim, cos_sin_cache,
                                 is_neox_style=False,
                                 rope_dim_offset=head_dim - rope_dim)
    q_fp8, q_scale = per_token_group_quant_fp8(q_rot, head_dim, use_ue8m0=True)
    weights_out = weights * q_scale * softmax_scale * head_scale

Expects bit-exact equality on both q_fp8 and weights_out.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.v1.attention.ops.deepseek_v4_ops.fused_indexer_q import (
    fused_indexer_q_rope_quant,
)

HEAD_DIM = 128
ROPE_DIM = 64
N_HEAD = 64
MAX_POS = 4096


def _reference(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rot = q.clone()
    ops.rotary_embedding(
        positions,
        q_rot,
        None,
        HEAD_DIM,
        cos_sin_cache,
        False,  # is_neox_style=False → GPT-J interleaved
        HEAD_DIM - ROPE_DIM,  # rope_dim_offset → rotate the tail
        False,
    )
    q_fp8, q_scale = per_token_group_quant_fp8(
        q_rot.view(-1, HEAD_DIM).contiguous(),
        HEAD_DIM,
        use_ue8m0=True,
    )
    q_fp8 = q_fp8.view(-1, N_HEAD, HEAD_DIM)
    q_scale = q_scale.view(-1, N_HEAD)

    weights_out = weights.to(torch.float32) * q_scale * softmax_scale * head_scale
    return q_fp8, weights_out


@pytest.mark.parametrize("num_tokens", [1, 7, 32, 257])
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_fused_indexer_q_rope_quant_matches_unfused(num_tokens, cache_dtype):
    device = "cuda"
    torch.manual_seed(0)

    q = torch.randn(num_tokens, N_HEAD, HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.randint(
        0, MAX_POS, (num_tokens,), dtype=torch.int64, device=device
    )
    cos_sin_cache = torch.randn(MAX_POS, ROPE_DIM, dtype=cache_dtype, device=device)
    weights = torch.randn(num_tokens, N_HEAD, dtype=torch.bfloat16, device=device)
    softmax_scale = HEAD_DIM**-0.5
    head_scale = N_HEAD**-0.5

    q_fp8_ref, weights_ref = _reference(
        positions, q, cos_sin_cache, weights, softmax_scale, head_scale
    )
    q_fp8_fused, weights_fused = fused_indexer_q_rope_quant(
        positions, q.clone(), cos_sin_cache, weights, softmax_scale, head_scale
    )

    # fp8 tensors aren't directly comparable via torch.equal — reinterpret as int8.
    ref_bits = q_fp8_ref.view(torch.int8)
    fused_bits = q_fp8_fused.view(torch.int8)
    assert torch.equal(ref_bits, fused_bits), (
        f"q_fp8 mismatch: "
        f"{(ref_bits != fused_bits).sum().item()} / {ref_bits.numel()} bytes differ"
    )

    assert torch.equal(weights_ref, weights_fused), (
        f"weights mismatch: max abs diff "
        f"{(weights_ref - weights_fused).abs().max().item()}"
    )
