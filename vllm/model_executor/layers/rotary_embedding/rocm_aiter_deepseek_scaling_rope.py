# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

import vllm.envs as envs
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform

from .base import RotaryEmbedding
from .common import (yarn_find_correction_range, yarn_get_mscale,
                     yarn_linear_ramp_mask)


def is_rocm_rotatry_embedding_enabled() -> bool:
    return (current_platform.is_rocm() and envs.VLLM_ROCM_USE_AITER)


class AiterRotaryEmbedding(RotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        CustomOp.__init__(self)
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cos, sin = self._compute_cos_sin_cache()
        cos = cos.to(dtype)
        sin = sin.to(dtype)
        self.cos_cache: torch.Tensor
        self.sin_cache: torch.Tensor
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos().unsqueeze(-2).unsqueeze(-2)
        sin = freqs.sin().unsqueeze(-2).unsqueeze(-2)
        return cos, sin

    def forward_hip(
        self,
        positions: torch.Tensor,
        # if     is_nope_first
        # [[batch_size, seq_len, num_heads, nope_size+rope_size]
        # if NOT is_nope_first
        # [[batch_size, seq_len, num_heads, rope_size+nope_size],
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        is_nope_first=False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import aiter as ops

        self.cos_cache = self.cos_cache.to(query.device, dtype=query.dtype)
        self.sin_cache = self.sin_cache.to(query.device, dtype=query.dtype)
        cos, sin = self.cos_cache, self.sin_cache

        rotate_style = 0 if self.is_neox_style else 1

        num_tokens = positions.numel()

        query_shape = query.shape
        query = query.view(1, num_tokens, -1, self.head_size)
        if key is not None:
            key_shape = key.shape
            key = key.view(1, num_tokens, -1, self.head_size)

        positions = positions.view(*query.shape[:2])
        if offsets is not None:
            offsets = offsets.view(*query.shape[:2])

        if not is_nope_first:
            query_ = query[..., :self.rotary_dim]
            key_ = key[..., :self.rotary_dim] if key is not None else None
        else:
            query_ = query[..., -self.rotary_dim:]
            key_ = key[..., -self.rotary_dim:] if key is not None else None

        if key_ is not None:
            if offsets is None:
                ops.rope_cached_positions_2c_fwd_inplace(
                    query_,
                    key_,
                    cos,
                    sin,
                    positions,
                    rotate_style,
                    reuse_freqs_front_part=True,
                    nope_first=is_nope_first,
                )
            else:
                ops.rope_cached_positions_offsets_2c_fwd_inplace(
                    query_,
                    key_,
                    cos,
                    sin,
                    positions,
                    offsets,
                    rotate_style,
                    reuse_freqs_front_part=True,
                    nope_first=is_nope_first,
                )
            return query.view(query_shape), key.view(key_shape)
        else:
            if offsets is None:
                ops.rope_cached_positions_fwd_inplace(
                    query_,
                    cos,
                    sin,
                    positions,
                    rotate_style,
                    reuse_freqs_front_part=True,
                    nope_first=is_nope_first,
                )
            else:
                ops.rope_cached_positions_offsets_fwd_inplace(
                    query_,
                    cos,
                    sin,
                    positions,
                    offsets,
                    rotate_style,
                    reuse_freqs_front_part=True,
                    nope_first=is_nope_first,
                )
            return query.view(query_shape)


class AiterDeepseekScalingRotaryEmbedding(AiterRotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale)) /
            yarn_get_mscale(self.scaling_factor, float(mscale_all_dim)) *
            attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base**(
            torch.arange(0,
                         self.rotary_dim,
                         2,
                         dtype=torch.float32,
                         device=current_platform.device_type) /
            self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2,
            dtype=torch.float32)) * self.extrapolation_factor
        inv_freq = (inv_freq_interpolation * (1 - inv_freq_mask) +
                    inv_freq_extrapolation * inv_freq_mask)
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            self.max_position_embeddings * self.scaling_factor,
            device="cuda",
            dtype=torch.float32,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cos = freqs.cos().unsqueeze(-2).unsqueeze(-2)
        sin = freqs.sin().unsqueeze(-2).unsqueeze(-2)
        return cos, sin

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, key = super().forward(positions, query, key, offsets)
        if positions.numel() == 1:
            key = key.clone()
        return query, key
