# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
# This file is shared by kimi_vl.py and locate_anything.py (both use MoonViT)
# Copyright 2025 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
#
# The code is based on llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py), but modified for KimiVL.
#
# Licensing Information:
# - Code derived from llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py) is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the MIT License.
#
# Apache License, Version 2.0:
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
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math
from collections.abc import Sequence
from copy import deepcopy
from functools import cached_property
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel

from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.moonvit import MoonViTConfig
from vllm.utils.torch_utils import async_tensor_h2d


def get_num_image_tokens(
    *,
    image_width: int,
    image_height: int,
    patch_size: int,
    merge_kernel_size: tuple[int, int],
    in_token_limit: int,
) -> int:
    """Number of merged vision tokens MoonViT emits for one image.

    Shared by KimiVL and LocateAnything, which both feed images through the
    MoonViT image processor (downscale to ``in_token_limit`` patches, pad to a
    multiple of ``merge_kernel_size * patch_size``, then 2x2-merge). Keeping the
    geometry in one place ensures the placeholder count stays in sync with the
    vision tower's output length for both models.
    """
    height = image_height
    width = image_width
    assert isinstance(height, int), f"height must be int, current height {height}"
    assert isinstance(width, int), f"width must be int, current width {width}"
    assert merge_kernel_size is not None, "merge_kernel_size must be specified"

    if (width // patch_size) * (height // patch_size) > in_token_limit:
        scale = math.sqrt(
            in_token_limit / ((width // patch_size) * (height // patch_size))
        )
        new_w, new_h = int(width * scale), int(height * scale)
        width, height = new_w, new_h

    kernel_height, kernel_width = merge_kernel_size

    pad_height = (
        kernel_height * patch_size - height % (kernel_height * patch_size)
    ) % (kernel_height * patch_size)
    pad_width = (kernel_width * patch_size - width % (kernel_width * patch_size)) % (
        kernel_width * patch_size
    )

    token_height = (height + pad_height) // (kernel_height * patch_size)
    token_width = (width + pad_width) // (kernel_width * patch_size)
    return int(token_height * token_width)


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Learnable2DInterpPosEmb(nn.Module):
    def __init__(
        self, height: int, width: int, dim: int, interpolation_mode: str = "bicubic"
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def get_pos_embeds(
        self,
        grid_hws_list: list[list[int]] | list[tuple[int, int]],
    ) -> torch.Tensor:
        """Build packed per-token positional embeddings for a list of grids.

        Returns a tensor of shape ``(sum(h * w), dim)`` formed by interpolating
        the learned ``(height, width, dim)`` weight to each ``(h, w)`` grid and
        concatenating the flattened results in the same order as
        ``grid_hws_list``. Lives outside the captured CUDA graph so the
        per-grid Python iteration is safe.
        """
        weight_shape = list(self.weight.shape[:-1])
        pos_embs: list[torch.Tensor] = []
        for shape in grid_hws_list:
            shape_list = [int(shape[0]), int(shape[1])]
            if shape_list == weight_shape:
                pos_embs.append(self.weight.flatten(end_dim=1))
            else:
                pos_embs.append(
                    F.interpolate(
                        self.weight.permute((2, 0, 1)).unsqueeze(0),
                        size=tuple(shape_list),
                        mode=self.interpolation_mode,
                    )
                    .squeeze(0)
                    .permute((1, 2, 0))
                    .flatten(end_dim=1)
                )
        if not pos_embs:
            return self.weight.new_zeros((0, self.weight.shape[-1]))
        return torch.cat(pos_embs)

    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        pos_embs = self.get_pos_embeds(grid_hws.tolist())
        out = x + pos_embs
        return out


class MoonVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
    ):
        super().__init__()
        assert isinstance(patch_size, (int, Sequence)), (
            f"Invalid patch_size type: {type(patch_size)}"
        )
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert len(patch_size) == 2, (
            f"Expected patch_size to be a tuple of 2, got {patch_size}"
        )
        self.patch_size = patch_size

        self.proj = Conv2dLayer(
            in_dim, out_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_emb = Learnable2DInterpPosEmb(
            height=pos_emb_height, width=pos_emb_width, dim=out_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_hw: torch.Tensor | None = None,
        *,
        pos_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_hw (N, 2): grid height and width
            pos_embeds: precomputed positional embeddings of shape
                ``(L, Cout)``. When provided, ``grid_hw`` is unused and the
                CUDA-graph-incompatible interpolation in ``self.pos_emb`` is
                skipped.

        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).view(x.size(0), -1)
        if pos_embeds is not None:
            return x + pos_embeds
        return self.pos_emb(x, grid_hw)


class Rope2DPosEmb(nn.Module):
    """2D rotary position embedding with multi-resolution support.

    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.

    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py

    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(
        self,
        dim: int,
        max_height: int,
        max_width: int,
        theta_base=10000,
        device=current_platform.device_type,
    ):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base
        self.device = device

    def extra_repr(self):
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"

    @cached_property
    def precomputed_freqs_cis(self) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.

        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(self.device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(self.device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis_by_seqlens_list(
        self,
        grid_hws_list: list[list[int]] | list[tuple[int, int]],
    ) -> torch.Tensor:
        """List-based variant of :meth:`get_freqs_cis_by_seqlens`.

        Accepts a Python list of ``(h, w)`` pairs so callers that already
        operate outside the captured CUDA graph can avoid materializing a
        tensor + ``.tolist()`` round-trip.
        """
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width
            for h, w in grid_hws_list
        ), (
            grid_hws_list,
            self.max_height,
            self.max_width,
        )
        if not grid_hws_list:
            return self.precomputed_freqs_cis.new_zeros((0, self.dim // 2))
        freqs_cis = torch.cat(
            [
                self.precomputed_freqs_cis[:h, :w].reshape(-1, self.dim // 2)
                for h, w in grid_hws_list
            ],
            dim=0,
        )
        return freqs_cis

    def get_freqs_cis_by_seqlens(self, grid_hws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_hws (torch.Tensor): containing list of (height, width) or (t, height, width) tuples.
        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        return self.get_freqs_cis_by_seqlens_list(grid_hws.tolist())

    def get_freqs_cis_by_idx(
        self, pos_idx: torch.Tensor, pos_idx_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pos_idx: tensor of shape (..., 2), It contains the (h, w) position indices of each 2D token.
            pos_idx_mask: a mask of shape (...), the leading dimensions should be the same as pos_idx.
                Rope will only be applied to the tokens with True mask. `freqs_cis` for the tokens with False mask with be ones.
        Return:
            freqs_cis: tensor of shape (..., dim//2)
        """
        assert (
            pos_idx.shape[:-1] == pos_idx_mask.shape
            and pos_idx.shape[-1] == 2
            and pos_idx.ndim == pos_idx_mask.ndim + 1
        ), (pos_idx.shape, pos_idx_mask.shape)
        assert pos_idx_mask.dtype == torch.bool, pos_idx_mask.dtype

        shp = pos_idx_mask.shape + (self.dim // 2,)  # ..., head_dim/2
        freqs_cis = torch.ones(
            shp, dtype=torch.complex64, device=self.device
        )  # ..., head_dim/2
        freqs_cis[pos_idx_mask] = self.precomputed_freqs_cis[
            pos_idx[..., 0][pos_idx_mask], pos_idx[..., 1][pos_idx_mask]
        ]
        return freqs_cis


class MLP2(nn.Module):
    """
    Args:
        dims: [in_dim, hidden_dim, out_dim]
        bias: whether to use bias in linear layer.
    """

    def __init__(
        self,
        dims: list[int],
        activation,
        bias: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        assert len(dims) == 3
        self.use_data_parallel = is_vit_use_data_parallel()
        self.fc0 = ColumnParallelLinear(
            dims[0],
            dims[1],
            bias=bias,
            prefix=maybe_prefix(prefix, "fc0"),
            disable_tp=self.use_data_parallel,
        )
        self.fc1 = RowParallelLinear(
            dims[1],
            dims[2],
            bias=bias,
            prefix=maybe_prefix(prefix, "fc1"),
            disable_tp=self.use_data_parallel,
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc0(x)
        x = self.activation(x)
        x, _ = self.fc1(x)
        return x


class MoonVitEncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        prefix: str = "",
        *,
        activation=F.gelu,
        attn_bias: bool = False,
    ):
        super().__init__()
        self.use_data_parallel = is_vit_use_data_parallel()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads
        self.tp_size = (
            1 if self.use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.num_attention_heads_per_partition = divide(num_heads, self.tp_size)

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP2(
            [hidden_dim, mlp_dim, hidden_dim],
            activation,
            prefix=f"{prefix}.mlp",
        )
        self.wqkv = QKVParallelLinear(
            hidden_size=hidden_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=attn_bias,
            prefix=f"{prefix}.wqkv",
            disable_tp=self.use_data_parallel,
        )
        self.wo = RowParallelLinear(
            hidden_dim,
            hidden_dim,
            bias=attn_bias,
            prefix=f"{prefix}.wo",
            disable_tp=self.use_data_parallel,
        )
        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            scale=self.hidden_size_per_attention_head**-0.5,
            prefix=f"{prefix}.attn",
        )

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ):
        """
        Args:
            x (torch.Tensor): (seqlen, hidden_dim)
            cu_seqlens (torch.Tensor):
            max_seqlen: Optional precomputed scalar tensor. When omitted it
                is derived from ``cu_seqlens``, which produces a GPU scalar
                that breaks CUDA graph capture.
        """
        seq_length = x.size(0)
        xqkv, _ = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (batch_size, seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_out = self.attn(
            xq.unsqueeze(0),
            xk.unsqueeze(0),
            xv.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        attn_out = attn_out.reshape(
            seq_length,
            self.num_attention_heads_per_partition
            * self.hidden_size_per_attention_head,
        )
        attn_out, _ = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: non-packed (B, N, D) or packed (L, D). if non-packed, seqlens should be None, if packed, seqlens should be set
            max_seqlen: optional precomputed max-sequence-length scalar.

        Returns:
            output: same shape of input, non-packed (B, N, D) for non-packed input, (L, D) for packed input
        """
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        attn_out = self.attention_qkvpacked(
            hidden_states,
            cu_seqlens,
            rope_freqs_cis=rope_freqs_cis,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.mlp(self.norm1(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


class MoonVitEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.rope_2d = Rope2DPosEmb(
            block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512
        )
        self.blocks = nn.ModuleList(
            [
                MoonVitEncoderLayer(
                    prefix=f"{prefix}.blocks.{layer_idx}",
                    **block_cfg,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def get_rope_freqs_cis(
        self,
        grid_hws_list: list[list[int]] | list[tuple[int, int]],
    ) -> torch.Tensor:
        return self.rope_2d.get_freqs_cis_by_seqlens_list(grid_hws_list)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_hw: torch.Tensor | None = None,
        *,
        cu_seqlens: torch.Tensor | None = None,
        rope_freqs_cis: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rope_freqs_cis is None:
            rope_freqs_cis = self.rope_2d.get_freqs_cis_by_seqlens(grid_hws=grid_hw)

        if cu_seqlens is None:
            lengths = torch.cat(
                (
                    torch.zeros(1, device=hidden_states.device, dtype=grid_hw.dtype),
                    (grid_hw[:, 0] * grid_hw[:, 1]).to(hidden_states.device),
                )
            )
            cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)

        for _, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens,
                rope_freqs_cis=rope_freqs_cis,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


def patch_merger(
    x: torch.Tensor,
    grid_hw: torch.Tensor,
    merge_kernel_size: list[int, int] = (2, 2),
) -> list[torch.Tensor]:
    d_model = x.size(-1)

    outputs = []
    pre_sum = 0
    for x_shape in grid_hw.tolist():
        height, width = x_shape[0], x_shape[1]
        # Get the current sequence
        seq = x[pre_sum : pre_sum + height * width]
        # Reshape along self.merge_kernel_size and concat to the last dimension
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = height // kernel_height, width // kernel_width
        reshaped_seq = seq.view(
            new_height, kernel_height, new_width, kernel_width, d_model
        )
        reshaped_seq = reshaped_seq.permute(0, 2, 1, 3, 4).contiguous()
        padded_seq = reshaped_seq.view(
            new_height * new_width, kernel_height * kernel_width, -1
        )
        outputs.append(padded_seq)
        pre_sum += height * width

    return outputs


def patch_merger_packed(
    x: torch.Tensor,
    gather_idx: torch.Tensor,
    merge_kernel_size: tuple[int, int],
) -> torch.Tensor:
    """CUDA-graph-safe equivalent of :func:`patch_merger`.

    Uses a precomputed index tensor to gather the per-token reshape +
    permute that ``patch_merger`` does inside a Python loop. The output is
    the concatenated 3D tensor ``(sum(new_h * new_w), kh * kw, d_model)``,
    matching what ``torch.cat(patch_merger(...))`` would produce.
    """
    kh, kw = merge_kernel_size
    d_model = x.size(-1)
    return x.index_select(0, gather_idx).view(-1, kh * kw, d_model)


def _build_merge_gather_idx(
    grid_hws_list: list[list[int]] | list[tuple[int, int]],
    merge_kernel_size: tuple[int, int],
) -> np.ndarray:
    """Build the per-token gather indices used by :func:`patch_merger_packed`.

    For each item with grid (h, w) and merge kernel (kh, kw), the output
    block at position (nh, nw) gathers the kh*kw input tokens at rows
    (nh*kh + ih, nw*kw + iw) of that item, in (ih, iw) row-major order.
    """
    kh, kw = merge_kernel_size
    parts: list[np.ndarray] = []
    pre_sum = 0
    for h, w in grid_hws_list:
        new_h, new_w = h // kh, w // kw
        nh = np.arange(new_h, dtype=np.int64).reshape(new_h, 1, 1, 1)
        nw = np.arange(new_w, dtype=np.int64).reshape(1, new_w, 1, 1)
        ih = np.arange(kh, dtype=np.int64).reshape(1, 1, kh, 1)
        iw = np.arange(kw, dtype=np.int64).reshape(1, 1, 1, kw)
        # Linearized input row = (nh*kh + ih) * w + (nw*kw + iw), offset by
        # the per-item base ``pre_sum``. Output is laid out as
        # (new_h, new_w, kh, kw) which patch_merger flattens to
        # (new_h*new_w, kh*kw).
        idx = pre_sum + (nh * kh + ih) * w + (nw * kw + iw)
        parts.append(idx.reshape(-1))
        pre_sum += h * w
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(parts)


class MoonVitPretrainedModel(PreTrainedModel):
    config_class = MoonViTConfig
    model_type = "moonvit"
    _no_split_modules = ["PackingTransformer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(
        self,
        config: MoonViTConfig,
        prefix: str = "",
        *inputs,
        **kwargs,
    ):
        super().__init__(config, *inputs, **kwargs)
        config = deepcopy(config)
        self.merge_kernel_size = config.merge_kernel_size
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.vit_processing_type = "rope_2d"
        self.patch_embed = MoonVisionPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
        )

        self.encoder = MoonVitEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": ACT2FN["gelu_pytorch_tanh"],
                "attn_bias": True,
            },
            prefix=f"{prefix}.encoder",
        )

    def prepare_encoder_metadata(
        self,
        grid_hws_list: list[list[int]] | list[tuple[int, int]],
        *,
        max_batch_size: int | None = None,
        max_seqlen_override: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, Any]:
        """Precompute every grid-dependent input the encoder needs.

        Used by the CUDA graph capture and replay paths to precompute
        every grid-dependent input outside the captured graph, so per-grid
        Python iteration and ``.tolist()`` round-trips are fine; the
        values are then copied into fixed-shape buffers for replay.

        Args:
            grid_hws_list: List of ``(h, w)`` patch-grid sizes per image.
            max_batch_size: When set, ``cu_seqlens`` is right-padded with
                its last value so the buffer covers up to this many
                sequences. Required at CUDA graph capture/replay so the
                buffer shape matches what was recorded; padding entries
                are zero-length sequences and are ignored by varlen
                attention.
            max_seqlen_override: Override the per-replay max sequence
                length scalar. At capture this must be a safe upper bound
                (worst case: a single image consuming the full token
                budget) because the value is baked into the captured
                graph.
            device: Device for the metadata tensors. Defaults to the
                model's parameter device.
        """
        if device is None:
            device = next(self.parameters()).device

        # Normalize to a list of plain Python int pairs so the helpers
        # below never need ``.tolist()`` on a tensor.
        grid_pairs: list[tuple[int, int]] = [(int(h), int(w)) for h, w in grid_hws_list]

        metadata: dict[str, Any] = {}

        pos_embeds = self.patch_embed.pos_emb.get_pos_embeds(grid_pairs)
        metadata["pos_embeds"] = pos_embeds.to(device=device)

        rope_freqs_cis = self.encoder.get_rope_freqs_cis(grid_pairs)
        metadata["rope_freqs_cis"] = rope_freqs_cis.to(device=device)

        grid_arr = np.array(grid_pairs, dtype=np.int64)
        seq_lens = (grid_arr[:, 0] * grid_arr[:, 1]).astype(np.int32)
        cu_seqlens_np = np.concatenate(
            [
                np.zeros(1, dtype=np.int32),
                seq_lens.cumsum(dtype=np.int32),
            ]
        )

        if max_batch_size is not None:
            num_seqs = len(cu_seqlens_np) - 1
            if num_seqs < max_batch_size:
                cu_seqlens_np = np.concatenate(
                    [
                        cu_seqlens_np,
                        np.full(
                            max_batch_size - num_seqs,
                            cu_seqlens_np[-1],
                            dtype=np.int32,
                        ),
                    ]
                )
        metadata["cu_seqlens"] = async_tensor_h2d(cu_seqlens_np, device=device)

        if max_seqlen_override is not None:
            max_seqlen_val = int(max_seqlen_override)
        else:
            max_seqlen_val = int(seq_lens.max()) if len(seq_lens) > 0 else 0
        # Keep on CPU: attention wrappers may call .item() on this scalar
        # and we want that materialization to happen outside the captured
        # graph (the value is constant per capture anyway).
        metadata["max_seqlen"] = torch.tensor(max_seqlen_val, dtype=torch.int32)

        gather_idx_np = _build_merge_gather_idx(grid_pairs, self.merge_kernel_size)
        metadata["merge_gather_idx"] = async_tensor_h2d(gather_idx_np, device=device)

        return metadata

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_hw: torch.Tensor,
        *,
        encoder_metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_hw (torch.Tensor): The grid height and width.
            encoder_metadata: Optional precomputed metadata produced by
                :meth:`prepare_encoder_metadata`. When provided every
                ``.tolist()`` call in the forward path is skipped, the
                returned tensor is the packed
                ``(sum(new_h*new_w), kh*kw, hidden_size)`` form (suitable
                for CUDA graph capture/replay), and ``grid_hw`` is unused.
                When ``None`` the legacy path runs and returns a list of
                per-image tensors.
        """
        if encoder_metadata is not None:
            hidden_states = self.patch_embed(
                pixel_values, pos_embeds=encoder_metadata["pos_embeds"]
            )
            hidden_states = self.encoder(
                hidden_states,
                cu_seqlens=encoder_metadata["cu_seqlens"],
                rope_freqs_cis=encoder_metadata["rope_freqs_cis"],
                max_seqlen=encoder_metadata["max_seqlen"],
            )
            return patch_merger_packed(
                hidden_states,
                encoder_metadata["merge_gather_idx"],
                merge_kernel_size=self.merge_kernel_size,
            )

        hidden_states = self.patch_embed(pixel_values, grid_hw)
        hidden_states = self.encoder(hidden_states, grid_hw)
        hidden_states = patch_merger(
            hidden_states, grid_hw, merge_kernel_size=self.merge_kernel_size
        )
        return hidden_states
