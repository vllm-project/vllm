# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The Baidu team.
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
"""Inference-only Erine VL model compatible with HuggingFace weights."""
import math
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Callable, Literal, Optional, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import BatchFeature

from vllm.attention.layer import check_upstream_fa_availability
from vllm.config import VllmConfig
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.activation import QuickGELU
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargsItems)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.platforms import _Backend, current_platform
from vllm.sequence import IntermediateTensors

from .ernie45_vl_moe import Ernie4_5_VLMoeForCausalLM
from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .utils import (AutoWeightsLoader, WeightsMapper, maybe_prefix,
                    merge_multimodal_embeddings)
from .vision import get_vit_attn_backend

logger = init_logger(__name__)

# === Vision Transformer === #


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1),
                         "... d two -> ... (d two)",
                         two=2)


def apply_rotary_emb_torch(x: torch.Tensor,
                           cos: torch.Tensor,
                           sin: torch.Tensor,
                           interleaved: bool = False) -> torch.Tensor:
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(
        cos,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(
        sin,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos +
            rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]
        ],
        dim=-1,
    )


def apply_rotary_pos_emb_vision(t: torch.Tensor,
                                freqs: torch.Tensor) -> torch.Tensor:
    t_ = t.float()
    cos = freqs.cos()
    sin = freqs.sin()
    apply_rotary_emb = apply_rotary_emb_torch
    if current_platform.is_cuda():
        from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb
    output = apply_rotary_emb(t_, cos, sin).type_as(t)
    return output


def all_gather_interleave(local_tensor, hidden_size: int, tp_size: int):
    """All-gather the input tensor interleavely across model parallel group."""
    import torch.distributed as dist
    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(tp_size)]
    dist.all_gather(gathered_tensors,
                    local_tensor,
                    group=parallel_state.get_tp_group().device_group)

    gathered_tensors_split = [
        torch.split(tensor, hidden_size // tp_size, -1)
        for tensor in gathered_tensors
    ]
    ordered_tensors = [
        tensor for pair in zip(*gathered_tensors_split) for tensor in pair
    ]
    result_tensor = torch.cat(ordered_tensors, dim=-1)
    return result_tensor


class Ernie4_5_VisionAttention(nn.Module):
    """VisionAttention using VLLM framework APIs"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size)

        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv")
        self.proj = RowParallelLinear(input_size=projection_size,
                                      output_size=embed_dim,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.proj")

        # Detect attention implementation.
        self.attn_backend = get_vit_attn_backend(
            head_size=self.hidden_size_per_attention_head,
            dtype=torch.get_default_dtype())

        self.use_upstream_fa = False
        if self.attn_backend != _Backend.FLASH_ATTN and \
            check_upstream_fa_availability(torch.get_default_dtype()):
            self.attn_backend = _Backend.FLASH_ATTN
            self.use_upstream_fa = True

        if self.attn_backend not in {
                _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS,
                _Backend.ROCM_AITER_FA
        }:
            raise RuntimeError(
                f"Ernie45-VL does not support {self.attn_backend} backend now."
            )
        self.is_flash_attn_backend = self.attn_backend in {
            _Backend.FLASH_ATTN, _Backend.ROCM_AITER_FA
        }

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # [s, b, 3 * head * head_dim]
        seq_len, bs, _ = qkv.shape
        if self.tp_size > 1:
            qkv = all_gather_interleave(qkv, self.qkv.hidden_size,
                                        self.tp_size)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head * head_dim]
        q, k, v = qkv.chunk(3, dim=2)

        # 3 * [s, b, head * head_dim]
        if self.tp_size > 1:
            splitter = partial(dist_utils.split_tensor_along_last_dim,
                               num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
            v = splitter(v)[self.tp_rank]

        # 3 * [s, b, head * head_dim] -> 3 * [s, b, head, head_dim]
        new_shape = (seq_len, bs, self.num_attention_heads_per_partition,
                     self.hidden_size_per_attention_head)
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def forward(
            self,
            x: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: torch.Tensor,
            max_seqlen: Optional[int] = None,  # Only used for Flash Attention
            seqlens: Optional[list[int]] = None,  # Only used for xFormers
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous()
                   for x in (q, k, v))
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        if self.is_flash_attn_backend:
            # from vllm_flash_attn.flash_attn_interface import (
            #   flash_attn_varlen_func)
            if self.attn_backend == _Backend.ROCM_AITER_FA:
                from aiter import flash_attn_varlen_func
            else:
                if self.use_upstream_fa:
                    from flash_attn import flash_attn_varlen_func
                else:
                    from vllm.vllm_flash_attn import flash_attn_varlen_func

            q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

            output = flash_attn_varlen_func(q,
                                            k,
                                            v,
                                            cu_seqlens_q=cu_seqlens,
                                            cu_seqlens_k=cu_seqlens,
                                            max_seqlen_q=max_seqlen,
                                            max_seqlen_k=max_seqlen,
                                            dropout_p=0.0,
                                            causal=False)

            context_layer = rearrange(output,
                                      "(b s) ... -> b s ...",
                                      b=batch_size)
        elif self.attn_backend == _Backend.TORCH_SDPA:
            # Execute attention entry by entry for speed & less VRAM.
            outputs = []
            for i in range(1, len(cu_seqlens)):
                start_idx = cu_seqlens[i - 1]
                end_idx = cu_seqlens[i]
                q_i = q[:, start_idx:end_idx]
                k_i = k[:, start_idx:end_idx]
                v_i = v[:, start_idx:end_idx]
                q_i, k_i, v_i = (rearrange(x, "b s h d -> b h s d")
                                 for x in [q_i, k_i, v_i])
                output_i = F.scaled_dot_product_attention(q_i,
                                                          k_i,
                                                          v_i,
                                                          dropout_p=0.0)
                output_i = rearrange(output_i, "b h s d -> b s h d ")
                outputs.append(output_i)
            context_layer = torch.cat(outputs, dim=1)
        elif self.attn_backend == _Backend.XFORMERS:
            from xformers import ops as xops
            from xformers.ops.fmha.attn_bias import BlockDiagonalMask

            attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seqlens,
                                                       kv_seqlen=None,
                                                       device=q.device)

            context_layer = xops.memory_efficient_attention_forward(
                q, k, v, attn_bias=attn_bias, p=0, scale=None)
        context_layer = rearrange(context_layer,
                                  "b s h d -> s b (h d)").contiguous()

        output, _ = self.proj(context_layer)
        return output


class Ernie4_5_VisionMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: type[nn.Module] = QuickGELU,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features,
                                        hidden_features,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.fc1")
        self.act = act_layer()
        self.fc2 = RowParallelLinear(hidden_features,
                                     in_features,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.fc2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_parallel, _ = self.fc1(x)
        x_parallel = self.act(x_parallel)
        x, _ = self.fc2(x_parallel)
        return x


class Ernie4_5_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        act_layer: type[nn.Module] = QuickGELU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.attn = Ernie4_5_VisionAttention(embed_dim=dim,
                                             num_heads=num_heads,
                                             projection_size=dim,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.attn")

        self.mlp = Ernie4_5_VisionMLP(dim,
                                      mlp_hidden_dim,
                                      act_layer=act_layer,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.mlp")

    def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: torch.Tensor,
            max_seqlen: Optional[int] = None,  # Only used for Flash Attention
            seqlens: Optional[list[int]] = None,  # Only used for xFormers
    ) -> torch.Tensor:

        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
            seqlens=seqlens,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Ernie4_5_VisionPatchEmbed(nn.Module):

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1280,
        prefix="",
    ) -> None:

        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Linear(in_channels * patch_size * patch_size,
                              embed_dim,
                              bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.to(target_dtype)
        hidden_states = self.proj(hidden_states)

        return hidden_states


class Ernie4_5_VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.inv_freq = 1.0 / theta**(
            torch.arange(start=0, end=dim, step=2, dtype=torch.float32) / dim)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen,
                           device=self.inv_freq.device,
                           dtype=self.inv_freq.dtype)
        freqs = torch.outer(input=seq, vec2=self.inv_freq)
        return freqs


class Ernie4_5_VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:

        super().__init__()
        patch_size = vision_config.patch_size
        spatial_merge_size = vision_config.spatial_merge_size
        in_channels = vision_config.in_channels
        hidden_size = vision_config.hidden_size
        embed_dim = vision_config.embed_dim
        depth = vision_config.depth
        num_heads = vision_config.num_heads
        mlp_ratio = vision_config.mlp_ratio

        self.spatial_merge_size = spatial_merge_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.patch_embed = Ernie4_5_VisionPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            prefix=f"{prefix}.patch_embed",
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = Ernie4_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            Ernie4_5_VisionBlock(dim=embed_dim,
                                 num_heads=num_heads,
                                 mlp_ratio=mlp_ratio,
                                 norm_layer=norm_layer,
                                 quant_config=quant_config,
                                 prefix=f"{prefix}.blocks.{layer_idx}")
            for layer_idx in range(depth)
        ])

        assert (hidden_size == embed_dim
                ), "vit's config.hidden must be equal to config.embed_dim"
        self.ln = nn.LayerNorm(hidden_size, eps=1e-6)

        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim, dtype=torch.get_default_dtype())
        if self.attn_backend != _Backend.FLASH_ATTN and \
        check_upstream_fa_availability(torch.get_default_dtype()):
            self.attn_backend = _Backend.FLASH_ATTN

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def compute_attn_mask_seqlen(
            self, cu_seqlens: torch.Tensor
    ) -> tuple[Optional[int], Optional[list[int]]]:
        max_seqlen, seqlens = None, None
        if self.attn_backend == _Backend.FLASH_ATTN:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        elif self.attn_backend == _Backend.XFORMERS:
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return max_seqlen, seqlens

    def forward(self,
                hidden_states: torch.Tensor,
                grid_thw: torch.Tensor,
                num_pad=0) -> torch.Tensor:

        hidden_states = self.patch_embed(hidden_states)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.to(hidden_states.device)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0, dtype=torch.int32)

        if num_pad > 0:
            cu_seqlens = F.pad(cu_seqlens, (1, 1), value=0)
            cu_seqlens[-1] = cu_seqlens[-2] + num_pad
        else:
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # add batch size
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(dim=1)

        # pre-compute seqlens for attn mask to reduce cuMemcpy operations
        max_seqlen, seqlens = self.compute_attn_mask_seqlen(cu_seqlens)

        for i, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen,
                seqlens=seqlens,
            )

        final_output = self.ln(hidden_states)

        if final_output.ndim == 3:
            final_output = final_output.squeeze(dim=1)

        return final_output

    def load_weights(self, weights) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# === Vision Inputs === #


class Ernie4_5_VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """Shape:
    `(num_patches, num_channels * patch_size * patch_size)`
    """

    grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


Ernie4_5_VLImageInputs = Ernie4_5_VLImagePixelInputs


class Ernie4_5_VLVideoPixelInputs(TypedDict):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: torch.Tensor
    """Shape:
    `(num_patches,
      num_channels * temporal_patch_size * patch_size * patch_size)`
    """

    video_grid_thw: torch.Tensor
    """Shape: `(num_videos, 3)`

    This should be in `(grid_t, grid_h, grid_w)` format.
    """


Ernie4_5_VLVideoInputs = Ernie4_5_VLImagePixelInputs

# === Vision Processor === #


def round_by_factor(number: Union[int, float], factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: Union[int, float], factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: Union[int, float], factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 16384 * 28 * 28,
):
    MAX_RATIO = 200
    if max(height, width) / min(height, width) > MAX_RATIO:
        if height > width:
            new_width = max(factor, round_by_factor(width, factor))
            new_height = floor_by_factor(new_width * MAX_RATIO, factor)
        else:
            new_height = max(factor, round_by_factor(height, factor))
            new_width = floor_by_factor(new_height * MAX_RATIO, factor)

        height = new_height
        width = new_width

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    if min_pixels > h_bar * w_bar or h_bar * w_bar > max_pixels:
        raise ValueError(f"encounter invalid h_bar: {h_bar}, w_bar: {w_bar}")

    return h_bar, w_bar


class VariableResolutionResamplerModel(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 spatial_conv_size,
                 temporal_conv_size,
                 config,
                 prefix: str = "") -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.config = config
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size
        self.use_temporal_conv = config.use_temporal_conv

        # compress 2d conv(picture) to 1d
        self.spatial_dim = (self.in_dim * self.spatial_conv_size *
                            self.spatial_conv_size)
        # compress 3d conv(video) to 1d
        self.temporal_dim = (self.in_dim * self.spatial_conv_size *
                             self.spatial_conv_size * self.temporal_conv_size)

        self.spatial_linear1 = ColumnParallelLinear(
            self.spatial_dim,
            self.spatial_dim,
            bias=True,
            gather_output=True,
            quant_config=getattr(config, 'quant_config', None),
            prefix=f"{prefix}.spatial_linear1",
        )

        self.spatial_gelu = nn.GELU()

        self.spatial_linear2 = ColumnParallelLinear(
            self.spatial_dim,
            self.spatial_dim,
            bias=True,
            gather_output=True,
            quant_config=getattr(config, 'quant_config', None),
            prefix=f"{prefix}.spatial_linear2",
        )

        self.spatial_norm = nn.LayerNorm(self.spatial_dim, eps=1e-6)

        if self.use_temporal_conv:
            self.temporal_linear1 = ColumnParallelLinear(
                self.temporal_dim,
                self.spatial_dim,
                bias=True,
                gather_output=True,
                quant_config=getattr(config, 'quant_config', None),
                prefix=f"{prefix}.temporal_linear1",
            )

            self.temporal_gelu = nn.GELU()

            self.temporal_linear2 = ColumnParallelLinear(
                self.spatial_dim,
                self.spatial_dim,
                bias=True,
                gather_output=True,
                quant_config=getattr(config, 'quant_config', None),
                prefix=f"{prefix}.temporal_linear2",
            )

            self.temporal_norm = nn.LayerNorm(self.spatial_dim, eps=1e-6)

        self.mlp = ColumnParallelLinear(
            self.spatial_dim,
            self.out_dim,
            bias=True,
            gather_output=True,
            quant_config=getattr(config, 'quant_config', None),
            prefix=f"{prefix}.mlp",
        )

        self.after_norm = RMSNorm(hidden_size=out_dim,
                                  eps=getattr(config, 'rms_norm_eps', 1e-6))

    def spatial_conv_reshape(self, x, spatial_conv_size):
        S, C = x.shape
        x = x.reshape([-1, C * (spatial_conv_size**2)])
        return x

    def forward(self, x, grid_thw):

        def fwd_spatial(x):
            x = self.spatial_conv_reshape(x, self.spatial_conv_size)

            x, _ = self.spatial_linear1(x)
            x = self.spatial_gelu(x)
            x, _ = self.spatial_linear2(x)
            x = self.spatial_norm(x)

            return x

        def fwd_placeholder(x, grid_thw, to_tensor=False):

            grid_thw_cpu = grid_thw.cpu().numpy()
            grid_t, grid_hw = grid_thw_cpu[:, 0], grid_thw_cpu[:, 1:]
            grid_hw_after_conv = grid_hw.prod(-1) // (self.spatial_conv_size**
                                                      2)

            tokens_per_img_or_vid = grid_thw_cpu.prod(-1) // (
                self.spatial_conv_size**2)
            batch_offset = np.empty(tokens_per_img_or_vid.size,
                                    dtype=tokens_per_img_or_vid.dtype)
            batch_offset[0] = 0
            batch_offset[1:] = tokens_per_img_or_vid.cumsum()[:-1]

            slice_offsets = []
            for temporoal_size, spatial_size, b_offset in zip(
                    grid_t, grid_hw_after_conv, batch_offset):
                for temp_offset in range(0, temporoal_size, 2):
                    slice_offsets.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        ))
            slice_offsets = torch.tensor(np.concatenate(slice_offsets,
                                                        axis=-1)).to(x.device)

            slice_offsets2 = []
            for temporoal_size, spatial_size, b_offset in zip(
                    grid_t, grid_hw_after_conv, batch_offset):
                for temp_offset in range(1 if temporoal_size > 1 else 0,
                                         temporoal_size, 2):
                    slice_offsets2.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        ))
            slice_offsets2 = torch.tensor(
                np.concatenate(slice_offsets2, axis=-1)).to(x.device)

            x_timestep_1 = torch.index_select(x, dim=0, index=slice_offsets)
            x_timestep_2 = torch.index_select(x, dim=0, index=slice_offsets2)
            x = torch.concat([x_timestep_1, x_timestep_2], dim=-1)
            return x

        def fwd_temporal(x):
            x, _ = self.temporal_linear1(x)
            x = self.temporal_gelu(x)
            x, _ = self.temporal_linear2(x)
            x = self.temporal_norm(x)
            return x

        def fwd_mlp(x):
            x, _ = self.mlp(x)
            x = self.after_norm(x)
            return x

        x = fwd_spatial(x)
        if self.use_temporal_conv:
            x = fwd_placeholder(x, grid_thw)
            x = fwd_temporal(x)
        x = fwd_mlp(x)
        return x

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Ernie4_5_VLProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(use_fast=True, **kwargs)

    def get_image_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).image_processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        max_image_tokens = self.get_max_image_tokens()
        max_video_tokens = self.get_max_video_tokens(seq_len, mm_counts)
        return {"image": max_image_tokens, "video": max_video_tokens}

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
        image_processor: Optional[Any],
    ) -> tuple[ImageSize, int]:
        if image_processor is None:
            image_processor = self.get_image_processor()
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config

        patch_size = vision_config.patch_size
        spatial_conv_size = hf_config.spatial_conv_size
        temporal_conv_size = hf_config.temporal_conv_size

        if do_resize:
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * spatial_conv_size,
                min_pixels=image_processor.min_pixels,
                max_pixels=image_processor.max_pixels,
            )
            preprocessed_size = ImageSize(width=resized_width,
                                          height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width,
                                          height=image_height)

        grid_t = max(num_frames // temporal_conv_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (spatial_conv_size**2)

        return preprocessed_size, num_vision_tokens

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: Optional[Any],
    ) -> int:
        _, num_image_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            image_processor=image_processor,
        )
        return num_image_tokens

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
        image_processor: Optional[Any],
    ) -> int:
        _, num_video_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=num_frames,
            image_processor=image_processor,
        )
        return num_video_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        max_image_size, _ = self._get_vision_info(
            image_width=9999999,
            image_height=9999999,
            image_processor=None,
        )
        return max_image_size

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        num_image_tokens = self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            image_processor=None,
        )
        return num_image_tokens

    def _get_max_video_frames(self, max_tokens: int) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        num_frames = 0

        while True:
            next_num_frames = num_frames + 1
            next_max_tokens = self.get_num_video_tokens(
                image_width=target_width,
                image_height=target_height,
                num_frames=next_num_frames,
                image_processor=None,
            )

            if next_max_tokens > max_tokens:
                break

            num_frames = next_num_frames

        # If the number of frames is odd, discard one frame.
        if num_frames % 2 != 0:
            num_frames -= 1

        return num_frames

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_videos = mm_counts.get("video", 0)

        max_image_tokens = self.get_max_image_tokens() * max_images
        max_total_frames = self._get_max_video_frames(seq_len -
                                                      max_image_tokens)
        max_frames_per_video = max_total_frames // max(max_videos, 1)

        return max(max_frames_per_video, 2)

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=self.get_num_frames_with_most_features(
                seq_len, mm_counts),
            image_processor=None,
        )


class Ernie4_5VLMultiModalProcessor(
        BaseMultiModalProcessor[Ernie4_5_VLProcessingInfo]):

    def _pixel_values_norm(
        self,
        pixel_values: torch.Tensor,
        mm_kwargs: object,
    ) -> torch.Tensor:
        hf_config = self.info.get_hf_config()
        vision_config = hf_config.vision_config
        image_processor = self.info.get_image_processor(**mm_kwargs)
        image_mean_tensor = torch.tensor(image_processor.image_mean,
                                         dtype=torch.float32).reshape(
                                             [1, 3, 1, 1])
        image_std_tensor = torch.tensor(image_processor.image_std,
                                        dtype=torch.float32).reshape(
                                            [1, 3, 1, 1])
        rescale_factor = torch.tensor(image_processor.rescale_factor,
                                      dtype=torch.float32)
        patch_size_squared = vision_config.patch_size**2

        image_mean_tensor = (image_mean_tensor.squeeze(
            [-2, -1]).repeat_interleave(patch_size_squared, -1))
        image_std_tensor = (image_std_tensor.squeeze(
            [-2, -1]).repeat_interleave(patch_size_squared, -1))

        if not image_mean_tensor.is_contiguous():
            image_mean_tensor = image_mean_tensor.contiguous()
        if not image_std_tensor.is_contiguous():
            image_std_tensor = image_std_tensor.contiguous()

        pixel_values = (rescale_factor * pixel_values.to(torch.float32) -
                        image_mean_tensor) / image_std_tensor
        pixel_values = pixel_values.to(hf_config.torch_dtype)
        return pixel_values

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # when the prompt is not empty but the multimodal data is empty,
        # directly invoke the tokenizer.
        if "images" not in mm_data and "videos" not in mm_data and prompt != "":
            tokenizer = self.info.get_tokenizer()
            prompt_ids = tokenizer.encode(prompt)
            tokenizer_output = BatchFeature(dict(input_ids=[prompt_ids]),
                                            tensor_type="pt")
            return tokenizer_output

        if "images" not in mm_data:
            mm_data["images"] = []
        if "videos" not in mm_data:
            mm_data["videos"] = []
        processor_output = self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=[prompt],
                 images=mm_data["images"],
                 videos=mm_data["videos"]),
            dict(**mm_kwargs, **tok_kwargs),
        )

        # Divide the processor_output into two modalities: image and video.
        if processor_output is not None:
            pixel_values = processor_output['images']
            if pixel_values is not None:
                processor_output['images'] = self._pixel_values_norm(
                    pixel_values, mm_kwargs)
            for key in list(processor_output.keys()):
                if processor_output[key] is None:
                    del processor_output[key]
                    continue
                if key == "grid_thw":
                    grid_thw = processor_output['grid_thw']
                    pixel_values_all = processor_output['images']
                    # Identify elements where the first
                    # dimension is greater than 1 and
                    # treat them as the video modality
                    mask = grid_thw[:, 0] > 1
                    processor_output["video_grid_thw"] = grid_thw[mask]
                    processor_output["image_grid_thw"] = grid_thw[~mask]
                    image_patch_num = processor_output["image_grid_thw"].prod(
                        dim=1).sum()
                    processor_output[
                        'pixel_values'] = pixel_values_all[:image_patch_num]
                    processor_output['pixel_values_videos'] = pixel_values_all[
                        image_patch_num:]
                    del processor_output['images']

        return processor_output

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        before_placeholder = {
            "image": "<|image@placeholder|>",
            "video": "<|video@placeholder|>"
        }

        after_placeholder = {
            # image and video have same placeholder
            "image": "<|IMAGE_PLACEHOLDER|>",
            "video": "<|IMAGE_PLACEHOLDER|>"
        }

        merge_length = hf_processor.spatial_conv_size**2

        def get_replacement_ernie45vl(item_idx: int, modality: str):
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            if modality == "video":
                num_tokens = int(grid_thw.prod(
                )) // hf_processor.temporal_conv_size // merge_length
            else:
                num_tokens = int(grid_thw.prod()) // merge_length
            return after_placeholder[modality] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=before_placeholder[modality],
                replacement=partial(get_replacement_ernie45vl,
                                    modality=modality),
            ) for modality in ("image", "video")
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:

        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_grid_sizes = image_grid_thw.prod(-1)

        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        video_grid_sizes = video_grid_thw.prod(-1)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_sizes),
            video_grid_thw=MultiModalFieldConfig.batched("video"),
        )


class Ernie4_5_VLDummyInputsBuilder(
        BaseDummyInputsBuilder[Ernie4_5_VLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        prompt = ""
        for i in range(num_images):
            prompt += (f"Picture {i+1}:"
                       "<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>")

        for i in range(num_videos):
            prompt += (f"Video {i+1}:"
                       "<|VIDEO_START|><|video@placeholder|><|VIDEO_END|>")
        return prompt

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        target_num_frames = \
            self.info.get_num_frames_with_most_features(seq_len, mm_counts)

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
            "video":
            self._get_dummy_videos(width=target_width,
                                   height=target_height,
                                   num_frames=target_num_frames,
                                   num_videos=num_videos)
        }


@MULTIMODAL_REGISTRY.register_processor(
    Ernie4_5VLMultiModalProcessor,
    info=Ernie4_5_VLProcessingInfo,
    dummy_inputs=Ernie4_5_VLDummyInputsBuilder)
class Ernie4_5_VLMoeForConditionalGeneration(nn.Module, SupportsMultiModal,
                                             SupportsLoRA, SupportsPP):

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
            # model.resampler_model.-> language_model.model.resampler_model.
            # language_model.model.resampler_model. -> resampler_model.
            "language_model.model.resampler_model.": "resampler_model.",
        },
        # resampler_weight_mappings
        orig_to_new_substr={
            "spatial_linear.0.": "spatial_linear1.",
            "spatial_linear.2.": "spatial_linear2.",
            "spatial_linear.3.": "spatial_norm.",
            "temporal_linear.0.": "temporal_linear1.",
            "temporal_linear.2.": "temporal_linear2.",
            "temporal_linear.3.": "temporal_norm.",
        })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>"
        if modality.startswith("video"):
            return "<|VIDEO_START|><|video@placeholder|><|VIDEO_END|>"

        raise ValueError("Only image or video modality is supported")

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.vision_model = Ernie4_5_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision_model"),
        )

        self.language_model = Ernie4_5_VLMoeForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.resampler_model = VariableResolutionResamplerModel(
            self.config.pixel_hidden_size,
            self.config.hidden_size,
            self.config.spatial_conv_size,
            self.config.temporal_conv_size,
            config=self.config,
            prefix=maybe_prefix(prefix, "resampler_model"))

        self.visual_token_mask = None
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """compute logits"""
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def _vision_forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        if grid_thw is not None:
            grid_thw = grid_thw[grid_thw > 0]
            if grid_thw.numel() % 3 != 0:
                raise ValueError(
                    f"grid_thw has {grid_thw.numel()} elements after filtering,"
                    "which is not divisible by 3.")
            grid_thw = grid_thw.reshape(-1, 3)
            # example: [[1,64,64],[2,80,80]] -> [[1,64,64],[1,80,80],[1,80,80]]
            grid_thw = F.pad(
                torch.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0),
                [1, 0, 0, 0],
                value=1,
            )
        image_features = self.vision_model(pixel_values, grid_thw)
        return image_features

    def _set_visual_token_mask(self, input_ids: torch.Tensor) -> None:
        if getattr(self.config, "im_patch_id", None) is not None:
            self.visual_token_mask = (
                input_ids == self.config.im_patch_id).reshape(-1, 1)
        else:
            self.visual_token_mask = None

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                 f"Got ndim: {mm_input.ndim} "
                                 f"(shape={mm_input.shape})")
            return mm_input.reshape(-1, mm_input.shape[-1])
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Ernie4_5_VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Ernie4_5_VLImagePixelInputs(type="pixel_values",
                                               pixel_values=pixel_values,
                                               image_grid_thw=image_grid_thw)

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Optional[Ernie4_5_VLVideoInputs]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            return Ernie4_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

    def _process_image_input(
            self,
            image_input: Ernie4_5_VLImageInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values = image_input["pixel_values"].type(
            self.vision_model.dtype)
        image_features = self._vision_forward(pixel_values=pixel_values,
                                              grid_thw=grid_thw)
        image_embeds = self.resampler_model(image_features, grid_thw)

        merge_size = self.vision_model.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def _process_video_input(
            self,
            video_input: Ernie4_5_VLVideoInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values_videos = video_input["pixel_values_videos"].type(
            self.vision_model.dtype)
        video_features = self._vision_forward(pixel_values=pixel_values_videos,
                                              grid_thw=grid_thw)
        video_embeds = self.resampler_model(video_features, grid_thw)

        merge_size = self.vision_model.spatial_merge_size
        sizes = (grid_thw.prod(-1) //
                 self.config.temporal_conv_size) // merge_size // merge_size

        return video_embeds.split(sizes.tolist())

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)
            if input_key in ("pixel_values_videos",
                             "video_embeds") and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(
                    **kwargs)

        return modalities

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += vision_embeddings
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_video_input(video_input)
                multimodal_embeddings += video_embeddings

        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:

        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if multimodal_embeddings is None:
            return inputs_embeds

        self._set_visual_token_mask(input_ids)
        inputs_embeds = merge_multimodal_embeddings(input_ids, inputs_embeds,
                                                    multimodal_embeddings,
                                                    [self.config.im_patch_id])
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }

        if self.visual_token_mask is not None:

            if self.visual_token_mask.shape[0] != inputs_embeds.shape[0]:
                padding_len = inputs_embeds.shape[
                    0] - self.visual_token_mask.shape[0]
                # right pad False
                pad = torch.zeros(
                    (padding_len, self.visual_token_mask.shape[1]),
                    dtype=self.visual_token_mask.dtype,
                    device=self.visual_token_mask.device)
                self.visual_token_mask = torch.cat(
                    [self.visual_token_mask, pad], dim=0)

            forward_kwargs.update(
                {"visual_token_mask": self.visual_token_mask})
            self.visual_token_mask = None

        hidden_states = self.language_model.model(
            **forward_kwargs,
            **kwargs,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
