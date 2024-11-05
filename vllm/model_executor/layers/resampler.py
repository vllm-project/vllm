# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
#
# Copyright 2023 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Shared resampler perceiver network used in multimodal models and
related helpers for sincos positional embeddings.

Example models: Qwen (Qwen-VL), Minicpmv2.0
"""
import math
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig

DEFAULT_LN = partial(nn.LayerNorm, eps=1e-6)


def get_abs_pos(abs_pos: torch.Tensor, tgt_size: Union[torch.Tensor,
                                                       int]) -> torch.Tensor:
    # abs_pos: L, C
    # tgt_size: (H, W)
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    dtype = abs_pos.dtype
    if isinstance(tgt_size, int):
        tgt_size = (tgt_size, tgt_size)
    if (src_size == tgt_size[0] and src_size == tgt_size[1]):
        return abs_pos
    return (F.interpolate(
        abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
        size=(tgt_size[0], tgt_size[1]),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype))


# sin/cos positional embedding helpers are adapted from:
# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: np.ndarray,
    version: Tuple[int, int] = (2, 0)) -> torch.Tensor:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,) / (H, W)
    out: (M, D) / (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    if version == (2, 0):
        pos = pos.reshape(-1)  # (M,)
        out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    else:
        out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product
        emb_sin = np.sin(out)  # (H, W, D/2)
        emb_cos = np.cos(out)  # (H, W, D/2)
        emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: np.ndarray,
    version: Tuple[int, int] = (2, 0)) -> torch.Tensor:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0], version)  # (H*W, D/2) or (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1], version)  # (H*W, D/2) or (H, W, D/2)

    if version == (2, 0):
        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    else:
        emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_2d_sincos_pos_embed(
        embed_dim: int,
        grid_size: Union[int, Tuple[int, int]],
        cls_token: bool = False,
        version: Tuple[int, int] = (2, 0),
) -> torch.Tensor:
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or
                [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_h_size, grid_w_size = grid_size, grid_size
    else:
        grid_h_size, grid_w_size = grid_size[0], grid_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    assert isinstance(grid, np.ndarray) and \
        grid.shape == (2, grid_h_size, grid_w_size)

    if version == (2, 0):
        grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, version)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed],
                                       axis=0)
    else:
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, version)
    return pos_embed


class BaseResampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb.
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(self,
                 num_queries: int,
                 embed_dim: int,
                 num_heads: int,
                 kv_dim: Optional[int] = None,
                 norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
                 do_post_projection: bool = True,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()

        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=0.02)
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = ReplicatedLinear(kv_dim,
                                            embed_dim,
                                            bias=False,
                                            quant_config=quant_config,
                                            prefix=prefix)
        else:
            # Maintain the same return value with ReplicatedLinear.forward
            self.kv_proj = lambda *args, **kwargs: (  # type: ignore # noqa 
                nn.Identity()(*args, **kwargs),
                None,
            )
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        self.do_post_projection = do_post_projection
        self.ln_post = norm_layer(embed_dim) if do_post_projection else None
        self.proj = nn.Parameter(
            (embed_dim**-0.5) *
            torch.randn(embed_dim, embed_dim)) if do_post_projection else None

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class Resampler2(BaseResampler):
    """Resampler-perceiver network to be used for a variety of model types,
    e.g., Qwen-vl / Minicpmv 2.0. The main difference is the addition of the
    do_post_projection arg, which indicates whether or not there should be
    a post layer normalization and projector after the attention. This is
    present in minicpmv2.0, but not qwen-vl.
    """

    def __init__(self,
                 grid_size: int,
                 embed_dim: int,
                 num_heads: int,
                 kv_dim: Optional[int] = None,
                 norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
                 adaptive: bool = False,
                 do_post_projection: bool = True,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__(grid_size**2,
                         embed_dim,
                         num_heads,
                         kv_dim,
                         norm_layer,
                         do_post_projection=do_post_projection,
                         quant_config=quant_config,
                         prefix=prefix)

        self.adaptive = adaptive
        pos_embed_arr = get_2d_sincos_pos_embed(embed_dim,
                                                grid_size,
                                                version=(2, 0))

        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed_arr).requires_grad_(False))

        self.apply(self._init_weights)

    def forward(
        self,
        x: torch.Tensor,
        tgt_sizes: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tgt_sizes is None:
            tgt_sizes = int(math.sqrt(x.size(1)))
        if self.adaptive:
            pos_embed_arr = get_2d_sincos_pos_embed(self.embed_dim,
                                                    tgt_sizes,
                                                    version=(2, 0))
            pos_embed = torch.from_numpy(pos_embed_arr).to(device=x.device,
                                                           dtype=x.dtype)
        else:
            pos_embed = get_abs_pos(self.pos_embed,
                                    tgt_sizes).to(device=x.device,
                                                  dtype=x.dtype)

        x, _ = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)

        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1),
            x + pos_embed.unsqueeze(1),
            x,
            attn_mask=attn_mask,
        )[0]
        x = out.permute(1, 0, 2)
        if self.do_post_projection:
            x = self.ln_post(x)
            x = x @ self.proj
        return x
