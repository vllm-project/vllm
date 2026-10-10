# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton

from .mrope import MRotaryEmbedding


def _validate_bailing_mrope_config(
    head_size: int,
    rotary_dim: int,
    mrope_section: list[int],
) -> int:
    if rotary_dim <= 0 or rotary_dim % 2 != 0 or rotary_dim > head_size:
        raise ValueError(
            "Bailing M-RoPE requires an even rotary_dim in (0, head_size], "
            f"got rotary_dim={rotary_dim} and head_size={head_size}"
        )
    if len(mrope_section) != 3:
        raise ValueError(
            "Bailing M-RoPE expects [T, H, W] sections, "
            f"got mrope_section={mrope_section}"
        )

    temporal_size, height_size, width_size = mrope_section
    if any(section_size < 0 for section_size in mrope_section):
        raise ValueError(
            "Bailing M-RoPE sections must be non-negative, "
            f"got mrope_section={mrope_section}"
        )
    if height_size != width_size:
        raise ValueError(
            "Bailing M-RoPE requires equal H and W sections, "
            f"got height_size={height_size} and width_size={width_size}"
        )
    spatial_size = height_size + width_size
    if spatial_size + temporal_size != rotary_dim // 2:
        raise ValueError(
            "Bailing M-RoPE sections must sum to rotary_dim // 2, got "
            f"mrope_section={mrope_section} and rotary_dim={rotary_dim}"
        )
    return spatial_size


def _validate_bailing_positions(
    positions: torch.Tensor,
    *,
    require_2d: bool = False,
) -> None:
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "Bailing M-RoPE positions must have int32 or int64 dtype, got "
            f"dtype={positions.dtype}"
        )
    if positions.ndim == 1 and not require_2d:
        return
    if positions.ndim == 2:
        if positions.shape[0] == 3:
            return
        raise ValueError(
            "Bailing M-RoPE expects T/H/W positions with shape [3, N], "
            f"got shape={tuple(positions.shape)}"
        )
    if require_2d:
        raise ValueError(
            "Bailing M-RoPE expects T/H/W positions with shape [3, N], "
            f"got shape={tuple(positions.shape)}"
        )
    raise ValueError(
        f"Bailing M-RoPE positions must be 1D or 2D, got shape={tuple(positions.shape)}"
    )


@triton.jit
def _triton_bailing_mrope_forward(
    q_ptr,
    k_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    q_stride_token,
    q_stride_head,
    q_stride_dim,
    k_stride_token,
    k_stride_head,
    k_stride_dim,
    positions_stride_axis,
    positions_stride_token,
    cache_stride_position,
    cache_stride_dim,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    rotary_dim: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_half_rotary_dim: tl.constexpr,
    spatial_size: tl.constexpr,
):
    """Apply GPT-J-style H/W-interleaved, T-tail M-RoPE in place."""
    token_idx = tl.program_id(0)
    position_offset = token_idx * positions_stride_token
    temporal_position = tl.load(positions_ptr + position_offset).to(tl.int64)
    height_position = tl.load(
        positions_ptr + positions_stride_axis + position_offset
    ).to(tl.int64)
    width_position = tl.load(
        positions_ptr + 2 * positions_stride_axis + position_offset
    ).to(tl.int64)

    frequency_offsets = tl.arange(0, pad_half_rotary_dim)
    frequency_mask = frequency_offsets < rotary_dim // 2
    spatial_position = tl.where(
        frequency_offsets % 2 == 0,
        height_position,
        width_position,
    )
    selected_position = tl.where(
        frequency_offsets < spatial_size,
        spatial_position,
        temporal_position,
    )
    cache_offsets = (
        selected_position * cache_stride_position + frequency_offsets * cache_stride_dim
    )
    cos_row = tl.load(
        cos_sin_cache_ptr + cache_offsets,
        mask=frequency_mask,
        other=0.0,
    ).to(tl.float32)
    sin_row = tl.load(
        cos_sin_cache_ptr + cache_offsets + (rotary_dim // 2) * cache_stride_dim,
        mask=frequency_mask,
        other=0.0,
    ).to(tl.float32)

    q_head_offsets = tl.arange(0, pad_n_qh)[:, None]
    k_head_offsets = tl.arange(0, pad_n_kh)[:, None]
    pair_offsets = frequency_offsets[None, :]
    q_mask = (q_head_offsets < n_qh) & frequency_mask[None, :]
    k_mask = (k_head_offsets < n_kh) & frequency_mask[None, :]

    q_base = q_ptr + token_idx * q_stride_token
    q_even_offsets = q_head_offsets * q_stride_head + 2 * pair_offsets * q_stride_dim
    q_odd_offsets = q_even_offsets + q_stride_dim
    q_even = tl.load(q_base + q_even_offsets, mask=q_mask, other=0.0).to(tl.float32)
    q_odd = tl.load(q_base + q_odd_offsets, mask=q_mask, other=0.0).to(tl.float32)
    q_even_cos = (q_even * cos_row[None, :]).to(q_ptr.type.element_ty).to(tl.float32)
    q_odd_sin = (q_odd * sin_row[None, :]).to(q_ptr.type.element_ty).to(tl.float32)
    q_odd_cos = (q_odd * cos_row[None, :]).to(q_ptr.type.element_ty).to(tl.float32)
    q_even_sin = (q_even * sin_row[None, :]).to(q_ptr.type.element_ty).to(tl.float32)
    tl.store(
        q_base + q_even_offsets,
        q_even_cos - q_odd_sin,
        mask=q_mask,
    )
    tl.store(
        q_base + q_odd_offsets,
        q_odd_cos + q_even_sin,
        mask=q_mask,
    )

    k_base = k_ptr + token_idx * k_stride_token
    k_even_offsets = k_head_offsets * k_stride_head + 2 * pair_offsets * k_stride_dim
    k_odd_offsets = k_even_offsets + k_stride_dim
    k_even = tl.load(k_base + k_even_offsets, mask=k_mask, other=0.0).to(tl.float32)
    k_odd = tl.load(k_base + k_odd_offsets, mask=k_mask, other=0.0).to(tl.float32)
    k_even_cos = (k_even * cos_row[None, :]).to(k_ptr.type.element_ty).to(tl.float32)
    k_odd_sin = (k_odd * sin_row[None, :]).to(k_ptr.type.element_ty).to(tl.float32)
    k_odd_cos = (k_odd * cos_row[None, :]).to(k_ptr.type.element_ty).to(tl.float32)
    k_even_sin = (k_even * sin_row[None, :]).to(k_ptr.type.element_ty).to(tl.float32)
    tl.store(
        k_base + k_even_offsets,
        k_even_cos - k_odd_sin,
        mask=k_mask,
    )
    tl.store(
        k_base + k_odd_offsets,
        k_odd_cos + k_even_sin,
        mask=k_mask,
    )


def triton_bailing_mrope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    spatial_size: int,
    head_size: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Bailing M-RoPE for packed or strided Q/K."""
    _validate_bailing_positions(positions, require_2d=True)
    num_tokens = positions.shape[-1]
    if num_tokens == 0:
        return query, key

    query_3d = query.view(num_tokens, -1, head_size)
    key_3d = key.view(num_tokens, -1, head_size)
    n_q_heads = query_3d.shape[1]
    n_k_heads = key_3d.shape[1]
    pad_n_q_heads = triton.next_power_of_2(n_q_heads)
    pad_n_k_heads = triton.next_power_of_2(n_k_heads)
    pad_half_rotary_dim = triton.next_power_of_2(rotary_dim // 2)
    _triton_bailing_mrope_forward[(num_tokens,)](
        query_3d,
        key_3d,
        positions,
        cos_sin_cache,
        query_3d.stride(0),
        query_3d.stride(1),
        query_3d.stride(2),
        key_3d.stride(0),
        key_3d.stride(1),
        key_3d.stride(2),
        positions.stride(0),
        positions.stride(1),
        cos_sin_cache.stride(0),
        cos_sin_cache.stride(1),
        n_q_heads,
        n_k_heads,
        rotary_dim,
        pad_n_q_heads,
        pad_n_k_heads,
        pad_half_rotary_dim,
        spatial_size,
        num_warps=4,
    )
    return query, key


class BailingMRotaryEmbedding(MRotaryEmbedding):
    """Bailing M-RoPE with alternating H/W frequencies and a temporal tail."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: list[int],
    ) -> None:
        self.spatial_size = _validate_bailing_mrope_config(
            head_size, rotary_dim, mrope_section
        )
        super().__init__(
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
            mrope_section=mrope_section,
        )

    def select_cos_sin(
        self,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_bailing_positions(positions)
        if cos_sin_cache is None:
            cos_sin_cache = self.cos_sin_cache
        cos_sin = cos_sin_cache[positions.to(torch.long)]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if positions.ndim == 1:
            return cos, sin

        spatial_mask = torch.arange(self.spatial_size, device=cos.device) % 2 == 0
        spatial_mask = spatial_mask.unsqueeze(0)
        spatial_cos = torch.where(
            spatial_mask,
            cos[1, :, : self.spatial_size],
            cos[2, :, : self.spatial_size],
        )
        spatial_sin = torch.where(
            spatial_mask,
            sin[1, :, : self.spatial_size],
            sin[2, :, : self.spatial_size],
        )
        temporal_slice = slice(self.spatial_size, self.rotary_dim // 2)
        return (
            torch.cat((spatial_cos, cos[0, :, temporal_slice]), dim=-1),
            torch.cat((spatial_sin, sin[0, :, temporal_slice]), dim=-1),
        )

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if key is None:
            raise ValueError("Bailing M-RoPE requires key states")

        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        cos, sin = self.select_cos_sin(positions, cos_sin_cache)
        num_tokens = positions.shape[-1]
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = self.apply_rotary_emb.forward_native(
            query[..., : self.rotary_dim],
            cos,
            sin,
        )
        query = torch.cat((query_rot, query[..., self.rotary_dim :]), dim=-1)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = self.apply_rotary_emb.forward_native(
            key[..., : self.rotary_dim],
            cos,
            sin,
        )
        key = torch.cat((key_rot, key[..., self.rotary_dim :]), dim=-1)
        return query.reshape(query_shape), key.reshape(key_shape)

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if positions.ndim == 2 and not self.is_neox_style:
            if key is None:
                raise ValueError("Bailing M-RoPE requires key states")
            cos_sin_cache = self._match_cos_sin_cache_dtype(query)
            return triton_bailing_mrope(
                query,
                key,
                positions,
                cos_sin_cache,
                self.spatial_size,
                self.head_size,
                self.rotary_dim,
            )
        return self.forward_native(positions, query, key, offsets)

    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key, offsets)
