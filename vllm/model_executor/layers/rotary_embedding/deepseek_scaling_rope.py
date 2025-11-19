# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.forward_context import get_forward_context
import math

import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
try:
    import flashinfer
except:
    flashinfer = None
from vllm.logger import init_logger

from .base import RotaryEmbeddingBase
from .common import (
    rotate_gptj,
    rotate_neox,
    yarn_find_correction_range,
    yarn_linear_ramp_mask,
)

import vllm.envs as envs


logger = init_logger(__name__)
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
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, has_flashinfer
        )
        self.fuse_rotary_fp8 = (
            self.dtype == current_platform.fp8_dtype()
            and envs.VLLM_FLASHINFER_FUSE_MLA_ROPE_FP8
            and has_flashinfer()
        )
        if self.fuse_rotary_fp8:
            self.cos_sin_cache_float32 = self.cos_sin_cache.to(torch.float32)
            logger.info("fuse_rotary_fp8")
            assert flashinfer is not None

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(
                0,
                self.rotary_dim,
                2,
                dtype=torch.float,
                device=current_platform.device_type,
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
        query_nope: torch.Tensor | None = None,
        key_nope: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        assert key is not None
        self._match_cos_sin_cache_dtype(query)
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]
        if query_nope is not None and key_nope is not None:
            q_out = torch.empty_like(query, dtype=self.dtype)
            k_out = torch.empty_like(key, dtype=self.dtype)
            q_nope_out = torch.empty_like(query_nope, dtype=self.dtype)
            k_nope_out = torch.empty_like(key_nope, dtype=self.dtype)
            flashinfer.rope.mla_rope_quantize_fp8(
                query,
                key,
                query_nope,
                key_nope,
                self.cos_sin_cache_float32,
                positions,
                is_neox=self.is_neox_style,
                q_rope_out=q_out,
                k_rope_out=k_out,
                q_nope_out=q_nope_out,
                k_nope_out=k_nope_out,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                enable_pdl=False,
            )
            return q_out, k_out, q_nope_out, k_nope_out
        cos_sin = self.cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
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
        query_nope: torch.Tensor | None = None,
        key_nope: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key, offsets, query_nope, key_nope)

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        query_nope: torch.Tensor | None = None,
        key_nope: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key, offsets, query_nope, key_nope)
