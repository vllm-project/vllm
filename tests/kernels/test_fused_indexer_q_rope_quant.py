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


def quantize_to_mxfp4(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference MXFP4 quantization.

    Args:
        x: [..., head_dim] where head_dim is divisible by 32
    Returns:
        packed: [..., head_dim//2]  uint8   2 E2M1 nibbles/byte, low nibble = even index
        scales: [..., head_dim//32] uint8   1 ue8m0 byte
    """
    MXFP4_BLOCK_SIZE = 32
    orig_shape = x.shape
    head_dim = orig_shape[-1]
    n_blocks = head_dim // MXFP4_BLOCK_SIZE

    x_f32 = x.float().reshape(-1, n_blocks, MXFP4_BLOCK_SIZE)

    # Per-block ue8m0 scale: 2^ceil(log2(amax / 6.0)), stored as byte = exp + 127
    # 6 * 2^-126 is from https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py#L163
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=6 * (2**-126))
    log2_ratio = (amax * (1.0 / 6.0)).log2().ceil().clamp(-127.0, 127.0)
    scale = log2_ratio.exp2()
    ue8m0 = (log2_ratio + 127.0).to(torch.uint8)  # [*, n_blocks]

    # E2M1 round-to-nearest-even: midpoints round to the even code.
    # E2M1 values: [0.00, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00]
    # boundaries:  [   0.25, 0.75, 1.25, 1.75, 2.50, 3.50, 5.00]
    x_scaled = (x_f32 / scale).clamp(-6.0, 6.0)
    abs_x = x_scaled.abs()
    code = torch.zeros_like(abs_x, dtype=torch.int32)
    code = torch.where(abs_x > 0.25, 1, code)
    code = torch.where(abs_x >= 0.75, 2, code)
    code = torch.where(abs_x > 1.25, 3, code)
    code = torch.where(abs_x >= 1.75, 4, code)
    code = torch.where(abs_x > 2.5, 5, code)
    code = torch.where(abs_x >= 3.5, 6, code)
    code = torch.where(abs_x > 5.0, 7, code)
    sign = ((x_scaled.view(torch.int32) >> 31) & 1).to(torch.uint8)
    nibble = code.to(torch.uint8) | (sign << 3)

    # Pack: even-index element → low nibble, odd-index → high nibble
    nibble_flat = nibble.reshape(-1, head_dim)
    packed = (nibble_flat[:, 0::2] | (nibble_flat[:, 1::2] << 4)).contiguous()
    packed = packed.reshape(*orig_shape[:-1], head_dim // 2)

    scales = ue8m0.view(*orig_shape[:-1], n_blocks)
    return packed, scales


def _reference(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    use_fp4: bool = False,
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

    if use_fp4:
        q_packed, ue8m0 = quantize_to_mxfp4(q_rot.view(-1, N_HEAD, HEAD_DIM))
        # Pack 4 ue8m0 bytes into 1 int32
        q_scale = ue8m0.view(torch.int32).squeeze(-1)
        # FP4 path: q_scale stays separate (cannot be folded into a per-token scalar)
        weights_out = weights.to(torch.float32) * softmax_scale * head_scale
        return (q_packed, q_scale), weights_out

    else:
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
@pytest.mark.parametrize("use_fp4", [False, True])
@torch.inference_mode()
def test_fused_indexer_q_rope_quant_matches_unfused(num_tokens, cache_dtype, use_fp4):
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

    q_quant_ref, weights_ref = _reference(
        positions, q, cos_sin_cache, weights, softmax_scale, head_scale, use_fp4
    )
    q_quant_fused, weights_fused = fused_indexer_q_rope_quant(
        positions, q.clone(), cos_sin_cache, weights, softmax_scale, head_scale, use_fp4
    )

    if use_fp4:
        q_quant_ref, q_scale_ref = q_quant_ref
        q_quant_fused, q_scale_fused = q_quant_fused

        assert torch.equal(q_scale_ref, q_scale_fused), (
            f"q_scale mismatch: "
            f"{(q_scale_ref != q_scale_fused).sum().item()} "
            f"/ {q_scale_ref.numel()} bytes differ"
        )

    # fp8 tensors aren't directly comparable via torch.equal — reinterpret as int8.
    ref_bits = q_quant_ref.view(torch.int8)
    fused_bits = q_quant_fused.view(torch.int8)
    assert torch.equal(ref_bits, fused_bits), (
        f"q_quant_fused mismatch: "
        f"{(ref_bits != fused_bits).sum().item()} / {ref_bits.numel()} bytes differ"
    )

    assert weights_fused.dtype == torch.float32
    assert torch.equal(weights_ref, weights_fused), (
        f"weights mismatch: max abs diff "
        f"{(weights_ref - weights_fused).abs().max().item()}"
    )
