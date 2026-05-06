# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from ..op import register_op


def _rotate_neox(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: Tensor) -> Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _rotate_with_cos_sin(
    x_rot: Tensor,
    cos: Tensor,
    sin: Tensor,
    is_neox_style: bool,
) -> Tensor:
    rotated = _rotate_neox(x_rot) if is_neox_style else _rotate_gptj(x_rot)
    return x_rot * cos + rotated * sin


def _apply_rotary_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    is_neox_style: bool,
) -> Tensor:
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def _apply_standard_rope(
    x: Tensor,
    num_tokens: int,
    head_size: int,
    rotary_dim: int,
    cos: Tensor,
    sin: Tensor,
    is_neox_style: bool,
) -> Tensor:
    x_shape = x.shape
    x = x.view(num_tokens, -1, head_size)
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    x_rot = _apply_rotary_emb(
        x_rot,
        cos.to(x_rot.dtype),
        sin.to(x_rot.dtype),
        is_neox_style,
    )

    return torch.cat((x_rot, x_pass), dim=-1).reshape(x_shape)


def _apply_deepseek_rope(
    x: Tensor,
    rotary_dim: int,
    head_size: int,
    cos: Tensor,
    sin: Tensor,
    is_neox_style: bool,
) -> Tensor:
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:] if rotary_dim < head_size else None

    x_rot = _rotate_with_cos_sin(
        x_rot,
        cos.to(x_rot.dtype),
        sin.to(x_rot.dtype),
        is_neox_style,
    )
    if x_pass is None:
        return x_rot
    return torch.cat((x_rot, x_pass), dim=-1)


@register_op(activations=["query", "key"], allow_inplace=True)
def rotary_embedding(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
    offsets: Tensor | None = None,
    cos_sin_format: str = "standard",
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embedding to query and key."""
    if cos_sin_format == "standard":
        positions = positions.flatten()
        cos_sin = cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        query_out = _apply_standard_rope(
            query,
            positions.shape[0],
            head_size,
            rotary_dim,
            cos,
            sin,
            is_neox_style,
        )
        key_out = _apply_standard_rope(
            key,
            positions.shape[0],
            head_size,
            rotary_dim,
            cos,
            sin,
            is_neox_style,
        )
    elif cos_sin_format == "deepseek":
        pos = torch.add(positions, offsets) if offsets is not None else positions
        cos_sin = cos_sin_cache[pos]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if is_neox_style:
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
        query_out = _apply_deepseek_rope(
            query,
            rotary_dim,
            head_size,
            cos,
            sin,
            is_neox_style,
        )
        key_out = _apply_deepseek_rope(
            key,
            rotary_dim,
            head_size,
            cos,
            sin,
            is_neox_style,
        )
    else:
        raise ValueError(f"Unsupported cos_sin_format={cos_sin_format!r}")
    return query_out, key_out
