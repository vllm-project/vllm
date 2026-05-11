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
    rope_dim_offset: int = 0,
) -> Tensor:
    x_shape = x.shape
    x = x.view(num_tokens, -1, head_size)
    if rope_dim_offset == 0:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_rot = _apply_rotary_emb(
            x_rot,
            cos.to(x_rot.dtype),
            sin.to(x_rot.dtype),
            is_neox_style,
        )
        return torch.cat((x_rot, x_pass), dim=-1).reshape(x_shape)
    x_before = x[..., :rope_dim_offset]
    x_rot = x[..., rope_dim_offset : rope_dim_offset + rotary_dim]
    x_after = x[..., rope_dim_offset + rotary_dim :]
    x_rot = _apply_rotary_emb(
        x_rot,
        cos.to(x_rot.dtype),
        sin.to(x_rot.dtype),
        is_neox_style,
    )
    return torch.cat((x_before, x_rot, x_after), dim=-1).reshape(x_shape)


def _apply_deepseek_rope(
    x: Tensor,
    num_tokens: int,
    rotary_dim: int,
    head_size: int,
    cos: Tensor,
    sin: Tensor,
    is_neox_style: bool,
    rope_dim_offset: int = 0,
) -> Tensor:
    x_shape = x.shape
    x = x.view(num_tokens, -1, head_size)
    orig_dtype = x.dtype
    x_before = x[..., :rope_dim_offset]
    x_rot = x[..., rope_dim_offset : rope_dim_offset + rotary_dim]
    x_after = x[..., rope_dim_offset + rotary_dim :]

    # cos/sin may be higher precision (e.g. fp32) than x (e.g. bf16).
    # Let type promotion handle the upcast; cast result back to orig_dtype.
    x_rot = _rotate_with_cos_sin(x_rot, cos, sin, is_neox_style).to(orig_dtype)
    return torch.cat((x_before, x_rot, x_after), dim=-1).reshape(x_shape)


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
    inverse: bool = False,
    rope_dim_offset: int = 0,
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embedding to query and key."""
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    if offsets is not None:
        offsets = offsets.flatten()

    if cos_sin_format == "standard":
        cos_sin = cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if inverse:
            sin = -sin
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        query_out = _apply_standard_rope(
            query,
            num_tokens,
            head_size,
            rotary_dim,
            cos,
            sin,
            is_neox_style,
            rope_dim_offset,
        )
        key_out = _apply_standard_rope(
            key,
            num_tokens,
            head_size,
            rotary_dim,
            cos,
            sin,
            is_neox_style,
            rope_dim_offset,
        )
    elif cos_sin_format == "deepseek":
        pos = torch.add(positions, offsets) if offsets is not None else positions
        cos_sin = cos_sin_cache[pos]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if inverse:
            sin = -sin
        if is_neox_style:
            cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
            sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
        query_out = _apply_deepseek_rope(
            query,
            num_tokens,
            rotary_dim,
            head_size,
            cos,
            sin,
            is_neox_style,
            rope_dim_offset,
        )
        key_out = _apply_deepseek_rope(
            key,
            num_tokens,
            rotary_dim,
            head_size,
            cos,
            sin,
            is_neox_style,
            rope_dim_offset,
        )
    else:
        raise ValueError(f"Unsupported cos_sin_format={cos_sin_format!r}")
    return query_out, key_out


@register_op(activations=["query"], allow_inplace=True)
def rotary_embedding_query_only(
    positions: Tensor,
    query: Tensor,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
    offsets: Tensor | None = None,
    cos_sin_format: str = "standard",
    inverse: bool = False,
    rope_dim_offset: int = 0,
) -> Tensor:
    """Apply rotary position embedding to query only."""
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    if offsets is not None:
        offsets = offsets.flatten()

    if cos_sin_format == "standard":
        cos_sin = cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if inverse:
            sin = -sin
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        return _apply_standard_rope(
            query,
            num_tokens,
            head_size,
            rotary_dim,
            cos,
            sin,
            is_neox_style,
            rope_dim_offset,
        )
    elif cos_sin_format == "deepseek":
        pos = torch.add(positions, offsets) if offsets is not None else positions
        cos_sin = cos_sin_cache[pos]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if inverse:
            sin = -sin
        if is_neox_style:
            cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
            sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
        return _apply_deepseek_rope(
            query,
            num_tokens,
            rotary_dim,
            head_size,
            cos,
            sin,
            is_neox_style,
            rope_dim_offset,
        )
    else:
        raise ValueError(f"Unsupported cos_sin_format={cos_sin_format!r}")
