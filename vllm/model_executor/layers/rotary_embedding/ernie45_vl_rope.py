# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from .common import apply_rotary_emb_dispatch
from .mrope import MRotaryEmbedding


class Ernie4_5_VLRotaryEmbedding(MRotaryEmbedding):
    """3D rotary positional embedding. 3D is t:time h:height w:width"""

    def forward_native(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if positions.ndim == 2:
            assert self.mrope_section

            section_h = self.mrope_section[0]  # 22
            section_w = self.mrope_section[1]  # 22
            section_t = self.mrope_section[2]  # 20
            assert section_h == section_w
            # Split according to [h w h w h w h w... t t t...]
            section_cos_t = cos[..., -section_t:]
            section_cos_h = cos[..., : section_h + section_w : 2]
            section_cos_w = cos[..., 1 : section_h + section_w : 2]

            cos_t, cos_h, cos_w = section_cos_t[0], section_cos_h[1], section_cos_w[2]
            cos_hw = torch.stack([cos_h, cos_w], dim=-1).reshape(
                cos_h.shape[:-1] + (cos_h.shape[-1] * 2,)
            )
            cos = torch.cat([cos_hw, cos_t], dim=-1)

            section_sin_t = sin[..., -section_t:]
            section_sin_h = sin[..., : section_h + section_w : 2]
            section_sin_w = sin[..., 1 : section_h + section_w : 2]

            sin_t, sin_h, sin_w = section_sin_t[0], section_sin_h[1], section_sin_w[2]
            sin_hw = torch.stack([sin_h, sin_w], dim=-1).reshape(
                sin_h.shape[:-1] + (sin_h.shape[-1] * 2,)
            )
            sin = torch.cat([sin_hw, sin_t], dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = apply_rotary_emb_dispatch(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = apply_rotary_emb_dispatch(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_cuda(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key)
