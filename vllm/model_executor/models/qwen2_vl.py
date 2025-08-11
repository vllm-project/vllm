# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/19e6e80e10118f855137b90740936c0b11ac397f/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# Copyright 2024 The Qwen team.
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
"""Inference-only Qwen2-VL model compatible with HuggingFace weights."""
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Callable, Literal, Optional, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import AutoConfig, BatchFeature
from transformers.models.qwen2_vl import (Qwen2VLImageProcessor,
                                          Qwen2VLProcessor)
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLConfig, Qwen2VLVisionConfig)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen2_vl.video_processing_qwen2_vl import (
    Qwen2VLVideoProcessor)

from vllm.config import VllmConfig
from vllm.distributed import parallel_state, tensor_model_parallel_all_gather
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.activation import QuickGELU
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (ImageItem, ModalityData,
                                    MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, VideoItem)
from vllm.multimodal.parse import (DictEmbeddingItems, ImageSize,
                                   ModalityDataItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.platforms import _Backend, current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import uses_mrope
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .utils import (AutoWeightsLoader, WeightsMapper,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)
from .vision import get_vit_attn_backend

logger = init_logger(__name__)

# For profile run
_MAX_FRAMES_PER_VIDEO = 16

# === Vision Inputs === #


class Qwen2VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """Shape:
    `(num_patches, num_channels * patch_size * patch_size)`
    """

    image_grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


class Qwen2VLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: torch.Tensor
    """Supported types:
    - list[`torch.Tensor`]: A list of tensors holding all images' features.
        Each tensor holds an image's features.
    - `torch.Tensor`: A tensor holding all images' features
        (concatenation of all images' feature tensors).
    
    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on
        the number and resolution of the images.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    image_grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


Qwen2VLImageInputs = Union[Qwen2VLImagePixelInputs,
                           Qwen2VLImageEmbeddingInputs]


class Qwen2VLVideoPixelInputs(TypedDict):
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


class Qwen2VLVideoEmbeddingInputs(TypedDict):
    type: Literal["video_embeds"]
    video_embeds: torch.Tensor
    """Supported types:
    - list[`torch.Tensor`]: A list of tensors holding all videos' features.
        Each tensor holds an video's features.
    - `torch.Tensor`: A tensor holding all videos' features
        (concatenation of all videos' feature tensors).
    
    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on 
        the number and resolution of the videos.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    video_grid_thw: torch.Tensor
    """Shape: `(num_videos, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


Qwen2VLVideoInputs = Union[Qwen2VLVideoPixelInputs,
                           Qwen2VLVideoEmbeddingInputs]

# === Vision Encoder === #


class Qwen2VisionMLP(nn.Module):

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


class Qwen2VisionAttention(nn.Module):

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
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_size = world_size
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, world_size)

        self.qkv = ColumnParallelLinear(input_size=embed_dim,
                                        output_size=3 * projection_size,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.qkv")
        self.proj = RowParallelLinear(input_size=projection_size,
                                      output_size=embed_dim,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.proj")

        # Detect attention implementation.
        self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)
        if self.attn_backend not in {
                _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS,
                _Backend.ROCM_AITER_FA
        }:
            raise RuntimeError(
                f"Qwen2-VL does not support {self.attn_backend} backend now.")
        self.is_flash_attn_backend = self.attn_backend in {
            _Backend.FLASH_ATTN, _Backend.ROCM_AITER_FA
        }

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # [s, b, 3 * head * head_dim]
        seq_len, bs, _ = qkv.shape
        if self.tp_size > 1:
            qkv = tensor_model_parallel_all_gather(qkv)

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

        # [s, b, c] --> [s, b, 3 * head * head_dim]
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
                from flash_attn import flash_attn_varlen_func

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


class Qwen2VisionBlock(nn.Module):

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

        self.attn = Qwen2VisionAttention(embed_dim=dim,
                                         num_heads=num_heads,
                                         projection_size=dim,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.attn")
        self.mlp = Qwen2VisionMLP(dim,
                                  mlp_hidden_dim,
                                  act_layer=act_layer,
                                  quant_config=quant_config,
                                  prefix=f"{prefix}.mlp")

    def forward(
            self,
            x: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: torch.Tensor,
            max_seqlen: Optional[int] = None,  # Only used for Flash Attention
            seqlens: Optional[list[int]] = None,  # Only used for xFormers
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
            seqlens=seqlens,
        )

        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2VisionPatchEmbed(nn.Module):

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels,
                              embed_dim,
                              kernel_size=kernel_size,
                              stride=kernel_size,
                              bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size,
                   self.patch_size)
        x = self.proj(x).view(L, self.embed_dim)
        return x


class Qwen2VisionPatchMerger(nn.Module):

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.ln_q = norm_layer(context_dim)
        self.mlp = nn.ModuleList([
            ColumnParallelLinear(self.hidden_size,
                                 self.hidden_size,
                                 bias=True,
                                 quant_config=quant_config,
                                 prefix=f"{prefix}.mlp.0"),
            nn.GELU(),
            RowParallelLinear(self.hidden_size,
                              d_model,
                              bias=True,
                              quant_config=quant_config,
                              prefix=f"{prefix}.mlp.2"),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class Qwen2VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta
                          **(torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (self.theta**(torch.arange(
                0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device)
                                                / self.dim))
            seq = torch.arange(seqlen,
                               device=self.inv_freq.device,
                               dtype=self.inv_freq.dtype)
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class Qwen2VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config: Qwen2VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
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

        self.patch_embed = Qwen2VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = Qwen2VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            Qwen2VisionBlock(dim=embed_dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             quant_config=quant_config,
                             prefix=f"{prefix}.blocks.{layer_idx}")
            for layer_idx in range(depth)
        ])
        self.merger = Qwen2VisionPatchMerger(
            d_model=hidden_size,
            context_dim=embed_dim,
            norm_layer=norm_layer,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )
        self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)

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
        if (self.attn_backend == _Backend.FLASH_ATTN
                or self.attn_backend == _Backend.ROCM_AITER_FA):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        elif self.attn_backend == _Backend.XFORMERS:
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return max_seqlen, seqlens

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        # transformers
        x = x.unsqueeze(1)

        # pre-compute seqlens for attn mask to reduce cuMemcpy operations
        max_seqlen, seqlens = self.compute_attn_mask_seqlen(cu_seqlens)
        for blk in self.blocks:
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen,
                seqlens=seqlens,
            )

        # adapter
        x = self.merger(x)

        return x

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


def _qwen2vl_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
    image_grid_sizes = image_grid_thw.prod(-1)

    video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
    video_grid_sizes = video_grid_thw.prod(-1)

    return dict(
        pixel_values=MultiModalFieldConfig.flat_from_sizes(
            "image", image_grid_sizes),
        image_embeds=MultiModalFieldConfig.flat_from_sizes(
            "image", image_grid_sizes),
        image_grid_thw=MultiModalFieldConfig.batched("image"),
        pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
            "video", video_grid_sizes),
        video_embeds=MultiModalFieldConfig.flat_from_sizes(
            "video", video_grid_sizes),
        video_grid_thw=MultiModalFieldConfig.batched("video"),
    )


class Qwen2VLMultiModalDataParser(MultiModalDataParser):

    def _parse_image_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[ImageItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields={"image_embeds", "image_grid_thw"},
                fields_factory=_qwen2vl_field_config,
            )

        return super()._parse_image_data(data)

    def _parse_video_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[VideoItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="video",
                required_fields={"video_embeds", "video_grid_thw"},
                fields_factory=_qwen2vl_field_config,
            )

        return super()._parse_video_data(data)


class Qwen2VLProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen2VLConfig)

    def get_hf_processor(self, **kwargs: object) -> Qwen2VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen2VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_image_processor(self, **kwargs: object) -> Qwen2VLImageProcessor:
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
        image_processor: Optional[Qwen2VLImageProcessor],
    ) -> tuple[ImageSize, int]:
        if image_processor is None:
            image_processor = self.get_image_processor()

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        if do_resize:
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=image_processor.min_pixels,
                max_pixels=image_processor.max_pixels,
            )
            preprocessed_size = ImageSize(width=resized_width,
                                          height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width,
                                          height=image_height)

        # NOTE: Frames are padded to be divisible by `temporal_patch_size`
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L294
        padded_num_frames = num_frames + num_frames % temporal_patch_size

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: Optional[Qwen2VLImageProcessor],
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
        image_processor: Optional[Qwen2VLImageProcessor],
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

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            image_processor=None,
        )

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
        max_frames_per_video = min(max_total_frames // max(max_videos, 1),
                                   _MAX_FRAMES_PER_VIDEO)

        return max(max_frames_per_video, 1)

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


class Qwen2VLDummyInputsBuilder(BaseDummyInputsBuilder[Qwen2VLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        hf_processor = self.info.get_hf_processor()
        image_token: str = hf_processor.image_token
        video_token: str = hf_processor.video_token

        return image_token * num_images + video_token * num_videos

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
            self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            )
        }


class Qwen2VLMultiModalProcessor(BaseMultiModalProcessor[Qwen2VLProcessingInfo]
                                 ):

    def _get_data_parser(self) -> MultiModalDataParser:
        return Qwen2VLMultiModalDataParser()

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        placeholder = {
            "image": vocab[hf_processor.image_token],
            "video": vocab[hf_processor.video_token],
        }

        merge_length = image_processor.merge_size**2

        def get_replacement_qwen2vl(item_idx: int, modality: str):
            grid_thw = out_mm_kwargs[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [placeholder[modality]] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement_qwen2vl,
                                    modality=modality),
            ) for modality in ("image", "video")
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _qwen2vl_field_config(hf_inputs)


@MULTIMODAL_REGISTRY.register_processor(Qwen2VLMultiModalProcessor,
                                        info=Qwen2VLProcessingInfo,
                                        dummy_inputs=Qwen2VLDummyInputsBuilder)
class Qwen2VLForConditionalGeneration(nn.Module, SupportsMultiModal,
                                      SupportsLoRA, SupportsPP):

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            # mapping for original checkpoint
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"

        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen2VLConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        if multimodal_config.get_limit_per_prompt("image") or \
            multimodal_config.get_limit_per_prompt("video"):
            self.visual = Qwen2VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=maybe_prefix(prefix, "visual"),
            )
        else:
            self.visual = None

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        # GPTQ configs do not have a list of ignored modules, however AutoGPTQ
        # seems to avoid vision encoder sections for some models.
        # See: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4
        if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
            return None
        return quant_config

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
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Qwen2VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen2VLImagePixelInputs(type="pixel_values",
                                           pixel_values=pixel_values,
                                           image_grid_thw=image_grid_thw)

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return Qwen2VLImageEmbeddingInputs(type="image_embeds",
                                               image_embeds=image_embeds,
                                               image_grid_thw=image_grid_thw)

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Optional[Qwen2VLVideoInputs]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            return Qwen2VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(
                video_embeds, "video embeds")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")
            return Qwen2VLVideoEmbeddingInputs(type="video_embeds",
                                               video_embeds=video_embeds,
                                               video_grid_thw=video_grid_thw)

    def _process_image_input(
            self, image_input: Qwen2VLImageInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"]
        else:
            pixel_values = image_input["pixel_values"]
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def _process_video_input(
            self, video_input: Qwen2VLVideoInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"]
        else:
            pixel_values_videos = video_input["pixel_values_videos"]
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

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

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []
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
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id])
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Optional[Qwen2VLImagePixelInputs] = None,
        video_input: Optional[Qwen2VLVideoPixelInputs] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_id,
            )

        if video_input is not None:
            video_embeds = self._process_video_input(video_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds,
                placeholder_token_id=self.config.video_token_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for Qwen2-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            pixel_values: Pixel values to be fed to a model.
                `None` if no images are passed.
            image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in LLM.
                `None` if no images are passed.
            pixel_values_videos: Pixel values of videos to be fed to a model.
                `None` if no videos are passed.
            video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in LLM.
                `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None
            else:
                if uses_mrope(self.config):
                    assert positions.ndim == 2 and positions.size(0) == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.size()}")
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input)
                input_ids = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:

        skip_prefixes = []
        if self.visual is None:
            skip_prefixes.extend(["visual."])
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger.",
            tower_model="visual.",
        )


class Tarsier2MultiModalProcessor(Qwen2VLMultiModalProcessor):
    pass


class Tarsier2ImageProcessor(Qwen2VLImageProcessor):

    def __init__(
        self,
        size: Optional[dict[str, int]] = None,
        **kwargs,
    ) -> None:
        if size is not None and "min_pixels" in size and "max_pixels" in size:
            # Remap if Tarsier2-specific format is provided
            remapped_size = {
                "shortest_edge": size["min_pixels"],
                "longest_edge": size["max_pixels"]
            }
            super().__init__(size=remapped_size, **kwargs)
        else:
            super().__init__(size=size, **kwargs)


class Tarsier2Processor(Qwen2VLProcessor):

    def __init__(
        self,
        vision_config: dict,
        tokenizer: AnyTokenizer,
        **kwargs,
    ):
        self.image_processor = Tarsier2ImageProcessor(**vision_config)
        super().__init__(
            image_processor=self.image_processor,
            tokenizer=tokenizer,
            video_processor=Qwen2VLVideoProcessor(**vision_config),
            chat_template=None,
            **kwargs)


class Tarsier2ProcessingInfo(Qwen2VLProcessingInfo):

    def get_hf_config(self) -> Qwen2VLConfig:
        model_path = self.ctx.model_config.model
        original_config = AutoConfig.from_pretrained(model_path)
        config_dict = original_config.to_dict()
        correct_config = Qwen2VLConfig.from_dict(config_dict)

        return correct_config

    def get_hf_processor(self, **kwargs: object) -> Tarsier2Processor:
        return Tarsier2Processor(
            vision_config=self.ctx.get_hf_image_processor_config(),
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )

    def get_image_processor(self) -> Tarsier2ImageProcessor:
        return Tarsier2ImageProcessor(
            **self.ctx.get_hf_image_processor_config())


@MULTIMODAL_REGISTRY.register_processor(Tarsier2MultiModalProcessor,
                                        info=Tarsier2ProcessingInfo,
                                        dummy_inputs=Qwen2VLDummyInputsBuilder)
class Tarsier2ForConditionalGeneration(Qwen2VLForConditionalGeneration):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "vision_tower.": "visual.",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Tarsier2 uses llava as model_type, which will create a Qwen2VLConfig
        # as text_config, we need to reconstruct Qwen2VLConfig from LlavaConfig.
        config = vllm_config.model_config.hf_config
        qwen2vl_config = config.text_config
        qwen2vl_config.architectures = config.architectures
        vllm_config.model_config.hf_config = qwen2vl_config
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:

        skip_prefixes = []
        if self.visual is None:
            skip_prefixes.extend(["visual."])
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
