# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm import ir
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

from .base import RotaryEmbeddingBase
from .common import (
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
        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        query_out, key_out = ir.ops.rotary_embedding.maybe_inplace(
            positions,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            cos_sin_cache,
            self.is_neox_style,
            offsets,
            "deepseek",
        )
        return query_out, key_out

    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key, offsets)

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
        # Keep cache in fp32 for higher precision RoPE; type promotion in the
        # IR op will upcast query to fp32 during computation and cast back.
        cos_sin_cache = self.cos_sin_cache.to(device=query.device)
        head_size = query.size(-1)
        rope_dim_offset = head_size - self.rotary_dim
        if key is None:
            query_out = ir.ops.rotary_embedding_query_only(
                positions,
                query,
                head_size,
                self.rotary_dim,
                cos_sin_cache,
                self.is_neox_style,
                offsets,
                "deepseek",
                inverse,
                rope_dim_offset,
            )
            return query_out, None
        query_out, key_out = ir.ops.rotary_embedding.maybe_inplace(
            positions,
            query,
            key,
            head_size,
            self.rotary_dim,
            cos_sin_cache,
            self.is_neox_style,
            offsets,
            "deepseek",
            inverse,
            rope_dim_offset,
        )
        return query_out, key_out

    def forward_hip(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key, offsets, inverse)

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key, offsets, inverse)
