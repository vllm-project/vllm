# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
# This file is meant to be used in kimi_vl.py only
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
from collections.abc import Sequence
from copy import deepcopy
from functools import cached_property

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import is_flash_attn_2_available

from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.moonvit import MoonViTConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
elif current_platform.is_xpu():
    from vllm.attention.utils.fa_utils import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None


def multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: torch.Tensor | None = None,
    k_cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-head attention using flash attention 2.

    Args:
        q: Query tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
        k: Key tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
        v: Value tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
        q_cu_seqlens (torch.Tensor): cumulative sequence lengths of q.
            The first element should be 0 and the last element should be q.shape[0].
        k_cu_seqlens (torch.Tensor): cumulative sequence lengths of k.
            The first element should be 0 and the last element should be k.shape[0].

    Returns:
        output: shape (batch_size, seqlen, dim) or (tot_seqlens, dim) if packing,
            where dim = num_heads * head_dim
    """
    # Unified format legal check
    assert q.dim() == k.dim() == v.dim() == 3, "q, k, v must have 3 dims"
    assert q_cu_seqlens[-1] == q.shape[0], "q_cu_seqlens must sum to q.shape[0]"
    assert k_cu_seqlens[-1] == k.shape[0] == v.shape[0], (
        "k_cu_seqlens must sum to k.shape[0]"
    )
    assert q.dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"unsupported dtype {q.dtype} for multihead attn"

    max_seqlen_q = (q_cu_seqlens[1:] - q_cu_seqlens[:-1]).max().item()
    max_seqlen_k = (k_cu_seqlens[1:] - k_cu_seqlens[:-1]).max().item()
    attn_out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=q_cu_seqlens,
        cu_seqlens_k=k_cu_seqlens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False,
    )
    attn_out = attn_out.flatten(start_dim=-2)

    return attn_out


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: torch.Tensor | None = None,
    k_cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """SDPA attention.

    Args:
        q: Query tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
        k: Key tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
        v: Value tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
        q_cu_seqlens: Optional cumulative sequence lengths of q.
        k_cu_seqlens: Optional cumulative sequence lengths of k.
    """
    seq_length = q.shape[0]
    attention_mask = torch.zeros(
        [1, seq_length, seq_length], device=q.device, dtype=torch.bool
    )
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[
            ...,
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


VL_VISION_ATTENTION_FUNCTIONS = {
    "flash_attention_2": multihead_attention,
    "sdpa": sdpa_attention,
}


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

    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for shape in grid_hws.tolist():
            if shape == self.weight.shape[:-1]:
                pos_embs.append(self.weight.flatten(end_dim=1))
            else:
                pos_embs.append(
                    F.interpolate(
                        self.weight.permute((2, 0, 1)).unsqueeze(0),
                        size=shape,
                        mode=self.interpolation_mode,
                    )
                    .squeeze(0)
                    .permute((1, 2, 0))
                    .flatten(end_dim=1)
                )
        out = x + torch.cat(pos_embs)
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

    def forward(self, x: torch.Tensor, grid_hw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_hw (N, 2): grid height and width

        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).view(x.size(0), -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_hw)
        return x


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

    def get_freqs_cis_by_seqlens(self, grid_hws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_hws (torch.Tensor): containing list of (height, width) or (t, height, width) tuples.
        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        shapes = grid_hws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = torch.cat(
            [
                self.precomputed_freqs_cis[:h, :w].reshape(-1, self.dim // 2)
                for h, w in shapes
            ],
            dim=0,
        )
        return freqs_cis

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
        use_data_parallel: bool = False,
    ):
        super().__init__()
        assert len(dims) == 3
        self.use_data_parallel = use_data_parallel
        self.fc0 = ReplicatedLinear(
            dims[0], dims[1], bias=bias, prefix=maybe_prefix(prefix, "fc0")
        )
        self.fc1 = ReplicatedLinear(
            dims[1], dims[2], bias=bias, prefix=maybe_prefix(prefix, "fc1")
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
        use_data_parallel: bool = False,
        *,
        attn_implementation: str = "sdpa",
        activation=F.gelu,
        attn_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads
        self.attn_implementation = attn_implementation
        # use fa2 in vllm by default
        if is_flash_attn_2_available() or current_platform.is_xpu():
            self.attn_implementation = "flash_attention_2"

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.use_data_parallel = use_data_parallel
        self.mlp = MLP2(
            [hidden_dim, mlp_dim, hidden_dim],
            activation,
            prefix=f"{prefix}.mlp",
            use_data_parallel=use_data_parallel,
        )
        self.wqkv = ReplicatedLinear(
            hidden_dim, hidden_dim * 3, bias=attn_bias, prefix=f"{prefix}.wqkv"
        )
        self.wo = ReplicatedLinear(
            hidden_dim, hidden_dim, bias=attn_bias, prefix=f"{prefix}.wo"
        )

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ):
        """
        Args:
            x (torch.Tensor): (batch_size, seqlen, hidden_dim)
            cu_seqlens (torch.Tensor):
        """
        xqkv, _ = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (batch_size, seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        attn_func = VL_VISION_ATTENTION_FUNCTIONS[self.attn_implementation]
        attn_out = attn_func(
            xq, xk, xv, q_cu_seqlens=cu_seqlens, k_cu_seqlens=cu_seqlens
        )
        attn_out, _ = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: non-packed (B, N, D) or packed (L, D). if non-packed, seqlens should be None, if packed, seqlens should be set

        Returns:
            output: same shape of input, non-packed (B, N, D) for non-packed input, (L, D) for packed input
        """
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        attn_out = self.attention_qkvpacked(
            hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis
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
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.rope_2d = Rope2DPosEmb(
            block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512
        )
        self.blocks = nn.ModuleList(
            [
                MoonVitEncoderLayer(
                    use_data_parallel=use_data_parallel,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                    **block_cfg,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self, hidden_states: torch.Tensor, grid_hw: torch.Tensor
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis_by_seqlens(grid_hws=grid_hw)

        lengths = torch.cat(
            (
                torch.zeros(1, device=hidden_states.device, dtype=grid_hw.dtype),
                (grid_hw[:, 0] * grid_hw[:, 1]).to(hidden_states.device),
            )
        )
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)

        for _, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis
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


class MoonVitVLProjector(nn.Module):
    def __init__(
        self,
        in_channels: int,
        merge_kernel_size: list[int, int],
        hidden_act: str = "gelu",
        ln_eps: float = 1e-5,
        out_dim: int = 4096,
    ):
        super().__init__()
        self.hidden_size = in_channels * merge_kernel_size[0] * merge_kernel_size[1]

        self.pre_norm = nn.nn.LayerNorm(in_channels, eps=ln_eps)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = ACT2FN[hidden_act]
        self.linear_2 = nn.Linear(self.hidden_size, out_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states).view(-1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class MoonVitPretrainedModel(PreTrainedModel):
    config_class = MoonViTConfig
    model_type = "moonvit"
    _no_split_modules = ["PackingTransformer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(
        self,
        config: MoonViTConfig,
        use_data_parallel: bool = False,
        prefix: str = "",
        *inputs,
        **kwargs,
    ):
        super().__init__(config, *inputs, **kwargs)
        config = deepcopy(config)
        self.use_data_parallel = use_data_parallel
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
                "attn_implementation": config._attn_implementation,
            },
            prefix=f"{prefix}.encoder",
        )

    def forward(
        self, pixel_values: torch.Tensor, grid_hw: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_hw (torch.Tensor): The grid height and width.

        Returns:
            torch.Tensor: The output tokens.
        """
        hidden_states = self.patch_embed(pixel_values, grid_hw)
        hidden_states = self.encoder(hidden_states, grid_hw)
        hidden_states = patch_merger(
            hidden_states, grid_hw, merge_kernel_size=self.merge_kernel_size
        )
        return hidden_states
