# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = True,
    enable_fp32_compute: bool = False,
) -> torch.Tensor:
    origin_dtype = x.dtype
    if enable_fp32_compute:
        x = x.float()
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        output = torch.cat((o1, o2), dim=-1)
    else:
        output = torch.stack((o1, o2), dim=-1).flatten(-2)
    return output.to(origin_dtype) if enable_fp32_compute else output


def native_rope(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply cached RoPE without modifying Q/K.

    Cache and Q/K must have matching devices and dtypes. Positions flatten
    to one index per token; Q/K end in heads * head_size or (heads, head_size).
    Platform implementations may modify Q/K in place, so callers must consume
    the returned tensors rather than rely on input mutation.
    """
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos, sin = cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(num_tokens, -1, head_size)
        rotated = apply_rotary_emb(x[..., :rotary_dim], cos, sin, is_neox_style)
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1).reshape(shape)

    return rotate(query), rotate(key) if key is not None else None
