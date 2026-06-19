# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 reference inverse-RoPE + WO_A einsum for the o-projection.

This file lives next to its Triton siblings under ``hw_agnostic/ops/``
and uses the ``triton_`` naming for consistency, but the implementation
itself is pure PyTorch — there is no Triton kernel here. Kept in the
same package so the agnostic attention layer imports all four
o-projection / sparse-MLA helpers from one place.

Inputs:
  * ``o``: attention output, ``[num_tokens, n_local_heads, head_dim]``,
    BF16. The trailing ``rope_head_dim`` elements still carry the
    forward-RoPE rotation applied earlier in the layer.
  * ``positions``: ``[num_tokens]`` int64.
  * ``wo_a``: linear module exposing ``weight`` (and optionally
    ``weight_scale_inv`` for FP8 quantization). The weights are
    dequantized to BF16 inline; this is intentionally a *reference*
    path, not a fused FP8 kernel.

Output: ``[num_tokens, n_local_groups, o_lora_rank]`` in BF16, ready
for ``wo_b`` (a RowParallelLinear) to consume.
"""

import math

import torch


def _decode_e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        return _upcast_e8m0_to_fp32(scale).contiguous()
    return scale.to(torch.float32)


def _expand_2d_block_scales(
    scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Expand a block-quantization scale tensor to the full ``[rows, cols]``."""
    scale = _decode_e8m0_scales(scale)
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = math.ceil(rows / row_blocks)
    col_block = math.ceil(cols / col_blocks)
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    scale = torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]
    return scale


def _apply_gptj_inv_rope_ref(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Inverse GPT-J interleaved RoPE on the trailing ``rope_dim`` of ``x``.

    Inverse rotation: ``[c -s; s c] -> [c s; -s c]``. Uses the same
    ``cos_sin_cache`` layout as the forward kernel — first half cos,
    second half sin.
    """
    if rope_dim == 0 or x.numel() == 0:
        return x
    half_rot = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    dtype = x.dtype
    x = x.to(torch.float32)
    cache = cos_sin_cache.index_select(0, positions.to(torch.long))
    cos = cache[:, :half_rot].to(torch.float32)
    sin = cache[:, half_rot : 2 * half_rot].to(torch.float32)
    view_shape = (positions.shape[0],) + (1,) * (x.dim() - 2) + (half_rot,)
    cos = cos.view(view_shape)
    sin = sin.view(view_shape)
    rope = x[..., nope_dim:]
    y_even = rope[..., 0::2]
    y_odd = rope[..., 1::2]
    rope_out = torch.stack(
        (y_even * cos + y_odd * sin, y_odd * cos - y_even * sin),
        dim=-1,
    ).flatten(-2)
    x = x.clone()
    x[..., nope_dim:] = rope_out
    return x.to(dtype)


def _apply_inv_rope_ref(
    rotary_emb: torch.nn.Module,
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Inverse RoPE with a fast path for rotary modules that expose one."""
    if hasattr(rotary_emb, "forward_native"):
        try:
            query, _ = rotary_emb.forward_native(
                positions,
                x.clone(),
                None,
                inverse=True,
            )
            return query
        except TypeError:
            pass
    return _apply_gptj_inv_rope_ref(x, positions, rotary_emb.cos_sin_cache, rope_dim)


def triton_inv_rope_einsum(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    wo_a: torch.nn.Module,
) -> torch.Tensor:
    """Inverse-RoPE the attention output and project through ``wo_a``.

    Returns a BF16 ``[num_tokens, n_local_groups, o_lora_rank]`` tensor.
    The caller flattens groups before applying the residual ``wo_b``.
    """
    o_ref = _apply_inv_rope_ref(rotary_emb, o, positions, rope_head_dim).to(
        torch.bfloat16
    )
    o_ref = o_ref.view(o.shape[0], n_local_groups, -1)

    hidden_dim = o_ref.shape[-1]
    if hasattr(wo_a, "weight_scale_inv"):
        wo_a_weight = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(
            torch.float32
        )
        wo_a_scale = _expand_2d_block_scales(
            wo_a.weight_scale_inv.view(
                n_local_groups, -1, wo_a.weight_scale_inv.shape[-1]
            ),
            o_lora_rank,
            hidden_dim,
        )
        wo_a_weight = (wo_a_weight * wo_a_scale).to(torch.bfloat16)
    else:
        wo_a_weight = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(
            torch.bfloat16
        )

    return torch.einsum("tgd,grd->tgr", o_ref, wo_a_weight)
