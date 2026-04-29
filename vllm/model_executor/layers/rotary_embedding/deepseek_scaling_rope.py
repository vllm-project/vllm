# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

from .base import RotaryEmbeddingBase
from .common import (
    rotate_gptj,
    rotate_neox,
    yarn_find_correction_range,
    yarn_linear_ramp_mask,
)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekScalingRotaryEmbedding(RotaryEmbeddingBase):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
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
        init_cache: bool = True,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale))
            / yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))
            * attn_factor
        )
        self.use_flashinfer = (
            self.enabled()
            and dtype in (torch.float16, torch.bfloat16)
            and current_platform.is_cuda()
            and has_flashinfer()
            and head_size in [64, 128, 256, 512]
        )
        super().__init__(
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
            init_cache=init_cache,
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(
                0,
                self.rotary_dim,
                2,
                dtype=torch.float,
            )
            / self.rotary_dim
        )
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
        inv_freq_mask = (
            1
            - yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)
        ) * self.extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            self.max_position_embeddings * self.scaling_factor,
            dtype=torch.float32,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """PyTorch-native implementation equivalent to forward()."""
        assert key is not None
        return self.forward_static(
            positions,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            self.cos_sin_cache,
            self.is_neox_style,
            offsets,
        )

    @staticmethod
    def forward_static(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        head_size: int,
        rotary_dim: int,
        cos_sin_cache: torch.Tensor,
        is_neox_style: bool,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A static implementation of forward()."""
        assert key is not None
        query_rot = query[..., :rotary_dim]
        key_rot = key[..., :rotary_dim]
        if rotary_dim < head_size:
            query_pass = query[..., rotary_dim:]
            key_pass = key[..., rotary_dim:]

        cos_sin = cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if is_neox_style:
            cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
            sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = rotate_neox if is_neox_style else rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if rotary_dim < head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key

    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return torch.ops.vllm.xpu_ops_deepseek_scaling_rope(
            positions,
            query,
            key,
            offsets,
            self._match_cos_sin_cache_dtype(query),
            self.rotary_dim,
            self.is_neox_style,
        )

    def forward_hip(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key, offsets)

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_flashinfer:
            torch.ops.vllm.flashinfer_rotary_embedding(
                torch.add(positions, offsets) if offsets is not None else positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
            return query, key
        else:
            return self.forward_native(positions, query, key, offsets)


class DeepseekV4ScalingRotaryEmbedding(DeepseekScalingRotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn

    Compared to DeepseekScalingRotaryEmbedding:
    - Applies RoPE to the last rotary_dim
    - The forward method requires an inverse parameter to indicate
      whether to negate the sin
    - Supports applying RoPE to query only (without key)
    - cos_sin_cache stored as fp32 for higher precision RoPE
    """

    def __init__(self, *args, **kwargs):
        # Avoid compute cache repeatedly
        kwargs.pop("init_cache", None)
        super().__init__(*args, **kwargs, init_cache=False)
        cache_fp32 = self._compute_cos_sin_cache()
        self.register_buffer("cos_sin_cache", cache_fp32, persistent=False)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            self.max_position_embeddings * self.scaling_factor,
            device=current_platform.device_type,
            dtype=torch.float32,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """PyTorch-native implementation equivalent to forward()."""

        head_size = query.size(-1)
        query_rot = query[..., -self.rotary_dim :]
        key_rot = key[..., -self.rotary_dim :] if key is not None else None

        if self.rotary_dim < head_size:
            query_pass = query[..., : -self.rotary_dim]
            key_pass = key[..., : -self.rotary_dim] if key is not None else None

        cos_sin = self.cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
            sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
        if inverse:
            sin = -sin
        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
        orig_dtype = query.dtype
        query_rot = (query_rot * cos + rotate_fn(query_rot) * sin).to(orig_dtype)
        if key_rot is not None:
            key_rot = (key_rot * cos + rotate_fn(key_rot) * sin).to(orig_dtype)

        if self.rotary_dim < head_size:
            query = torch.cat((query_pass, query_rot), dim=-1)
            key = torch.cat((key_pass, key_rot), dim=-1) if key is not None else None
        else:
            query = query_rot
            key = key_rot

        return query, key

    def forward_hip(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key, offsets)

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from vllm import _custom_ops as ops

        # The indexer and attention have different head_dim,
        # we obtain the corresponding head_dim via the query.
        head_size = query.size(-1)
        rope_dim_offset = head_size - self.rotary_dim
        # ops.rotary_embedding() is an in-place operation
        # that updates the query and key tensors.
        ops.rotary_embedding(
            torch.add(positions, offsets) if offsets is not None else positions,
            query,
            key,
            head_size,
            self.cos_sin_cache,
            self.is_neox_style,
            rope_dim_offset=rope_dim_offset,
            inverse=inverse,
        )
        return query, key
