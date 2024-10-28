# coding=utf-8
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
from functools import partial
from typing import (Any, Callable, Iterable, List, Literal, Mapping, Optional,
                    Tuple, Type, TypedDict, Union)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from transformers.image_utils import (get_image_size,
                                      infer_channel_dimension_format,
                                      to_numpy_array)
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLConfig, Qwen2VLVisionConfig)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    make_batched_images, make_batched_videos, smart_resize)

from vllm.attention import AttentionMetadata
from vllm.attention.selector import _Backend
from vllm.config import CacheConfig, MultiModalConfig
from vllm.distributed import get_pp_group, parallel_state
from vllm.distributed import utils as dist_utils
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, InputContext,
                         token_inputs)
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.activation import QuickGELU
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalDataDict,
                             MultiModalInputs)
from vllm.multimodal.base import MultiModalData
from vllm.multimodal.image import cached_get_image_processor
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import IntermediateTensors, SequenceData
from vllm.transformers_utils.config import uses_mrope
from vllm.transformers_utils.processor import cached_get_processor

from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (PPMissingLayer, get_vit_attn_backend,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory)

logger = init_logger(__name__)

# === Vision Inputs === #


class Qwen2VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape:
    `(num_patches, num_channels * patch_size * patch_size)`
    """

    image_grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


class Qwen2VLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`
    `hidden_size` must match the hidden size of language model backbone.
    """


Qwen2VLImageInputs = Union[Qwen2VLImagePixelInputs,
                           Qwen2VLImageEmbeddingInputs]


class Qwen2VLVideoInputs(TypedDict):
    pixel_values_videos: torch.Tensor
    """Shape:
    `(num_patches,
      num_channels * temporal_patch_size * patch_size * patch_size)`
    """

    video_grid_thw: torch.Tensor
    """Shape: `(num_videos, 3)`

    This should be in `(grid_t, grid_h, grid_w)` format.
    """


# === Vision Encoder === #


class Qwen2VisionMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        act_layer: Type[nn.Module] = QuickGELU,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features,
                                        hidden_features,
                                        quant_config=quant_config)
        self.act = act_layer()
        self.fc2 = RowParallelLinear(hidden_features,
                                     in_features,
                                     quant_config=quant_config)

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
    output = apply_rotary_emb_torch(t_, cos, sin).type_as(t)
    return output


class Qwen2VisionAttention(nn.Module):

    def __init__(
        self,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        projection_size: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, world_size)

        self.qkv = ColumnParallelLinear(input_size=embed_dim,
                                        output_size=3 * projection_size,
                                        quant_config=quant_config)
        self.proj = RowParallelLinear(input_size=projection_size,
                                      output_size=embed_dim,
                                      quant_config=quant_config)

        # Detect attention implementation.
        self.attn_backend: _Backend = get_vit_attn_backend()
        if self.attn_backend not in {
                _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS
        }:
            raise RuntimeError(
                f"Qwen2-VL does not support {self.attn_backend} backend now.")

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, head * 3 * head_dim] --> [s, b, head, 3 * head_dim]
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        x = x.view(*new_x_shape)

        # [s, b, head, 3 * head_dim] --> 3 [s, b, head, head_dim]
        q, k, v = dist_utils.split_tensor_along_last_dim(x, 3)
        batch_size = q.shape[1]

        q, k, v = [
            rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)
        ]
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        if self.attn_backend == _Backend.FLASH_ATTN:
            # from vllm_flash_attn.flash_attn_interface import (
            #   flash_attn_varlen_func)
            from flash_attn import flash_attn_varlen_func

            q, k, v = [rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]

            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            output = flash_attn_varlen_func(q,
                                            k,
                                            v,
                                            cu_seqlens_q=cu_seqlens,
                                            cu_seqlens_k=cu_seqlens,
                                            max_seqlen_q=max_seqlen,
                                            max_seqlen_k=max_seqlen,
                                            dropout_p=0,
                                            causal=False)

            context_layer = rearrange(output,
                                      "(b s) ... -> b s ...",
                                      b=batch_size)
        elif self.attn_backend == _Backend.TORCH_SDPA:
            seq_length = q.size(1)
            q, k, v = [rearrange(x, "b s h d -> b h s d") for x in [q, k, v]]
            attention_mask = torch.zeros([1, seq_length, seq_length],
                                         device=q.device,
                                         dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                               cu_seqlens[i - 1]:cu_seqlens[i]] = True
            output = F.scaled_dot_product_attention(q,
                                                    k,
                                                    v,
                                                    attention_mask,
                                                    dropout_p=0.0)
            context_layer = rearrange(output, "b h s d -> b s h d ")
        elif self.attn_backend == _Backend.XFORMERS:
            from xformers import ops as xops
            from xformers.ops.fmha.attn_bias import BlockDiagonalMask

            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seqlens,
                                                       kv_seqlen=None)

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
        act_layer: Type[nn.Module] = QuickGELU,
        norm_layer: Type[nn.Module] = None,
        quant_config: Optional[QuantizationConfig] = None,
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
                                         quant_config=quant_config)
        self.mlp = Qwen2VisionMLP(dim,
                                  mlp_hidden_dim,
                                  act_layer=act_layer,
                                  quant_config=quant_config)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x),
                          cu_seqlens=cu_seqlens,
                          rotary_pos_emb=rotary_pos_emb)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2VisionPatchEmbed(nn.Module):

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_chans,
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
        norm_layer: Type[nn.Module] = None,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
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
                                 quant_config=quant_config),
            nn.GELU(),
            RowParallelLinear(self.hidden_size,
                              d_model,
                              bias=True,
                              quant_config=quant_config),
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
    ) -> None:
        super().__init__()

        patch_size: int = vision_config.patch_size
        temporal_patch_size: int = vision_config.temporal_patch_size
        spatial_merge_size: int = vision_config.spatial_merge_size
        in_chans: int = vision_config.in_chans
        hidden_size: int = vision_config.hidden_size
        embed_dim: int = vision_config.embed_dim
        depth: int = vision_config.depth
        num_heads: int = vision_config.num_heads
        mlp_ratio: float = vision_config.mlp_ratio

        self.spatial_merge_size = spatial_merge_size

        self.patch_embed = Qwen2VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = Qwen2VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            Qwen2VisionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                quant_config=quant_config,
            ) for _ in range(depth)
        ])
        self.merger = Qwen2VisionPatchMerger(
            d_model=hidden_size,
            context_dim=embed_dim,
            norm_layer=norm_layer,
            quant_config=quant_config,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

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
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        # adapter
        x = self.merger(x)
        return x


# === Vision input helpers === #


def mm_input_mapper_for_qwen2_vl(
    ctx: InputContext,
    data: MultiModalData[object],
    data_type_key: str,
    *,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> MultiModalInputs:
    """Input mapper for Qwen2-VL."""
    if data_type_key == "image" and isinstance(data, dict):
        return MultiModalInputs({
            "image_embeds": data.get("image_embeds"),
            "image_grid_thw": data.get("image_grid_thw"),
        })
    model_config = ctx.model_config
    # Handle mm processor kwargs; we pass these at creation time
    # because preprocess() in transformers doesn't expose them
    mm_processor_kwargs = {}
    if min_pixels:
        mm_processor_kwargs["min_pixels"] = min_pixels
    if max_pixels:
        mm_processor_kwargs["max_pixels"] = max_pixels

    image_processor = cached_get_image_processor(
        model_config.model,
        trust_remote_code=model_config.trust_remote_code,
        **mm_processor_kwargs,
    )
    if image_processor is None:
        raise RuntimeError("No HuggingFace processor is available "
                           "to process the image object")

    images = None
    videos = None
    if data_type_key == "image":
        images = data
    else:
        assert data_type_key == "video"
        videos = data

    try:
        batch_data = image_processor \
            .preprocess(images=images, videos=videos, return_tensors="pt") \
            .data
    except Exception:
        logger.error("Failed to process image (%s)", data)
        raise

    return MultiModalInputs(batch_data)


image_input_mapper_for_qwen2_vl = partial(mm_input_mapper_for_qwen2_vl,
                                          data_type_key="image")
video_input_mapper_for_qwen2_vl = partial(mm_input_mapper_for_qwen2_vl,
                                          data_type_key="video")


def _get_vision_info(
    image_processor,
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    do_resize: bool = True,
    data_type_key: str = "image",
    mm_count: int = 1,
):
    """Get information (resized height / width and number of vision tokens)
    of input image / video frame."""

    if do_resize:
        resized_height, resized_width = smart_resize(
            height=height,
            width=width,
            factor=image_processor.patch_size * image_processor.merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    else:
        resized_height, resized_width = height, width

    if data_type_key == "image":
        grid_t = mm_count
    else:
        assert data_type_key == "video"
        grid_t = max(mm_count // image_processor.temporal_patch_size, 1)

    grid_h = resized_height // image_processor.patch_size
    grid_w = resized_width // image_processor.patch_size
    vision_tokens = grid_t * grid_h * grid_w
    llm_num_vision_tokens = (vision_tokens // image_processor.merge_size //
                             image_processor.merge_size)

    return resized_height, resized_width, llm_num_vision_tokens


def _get_max_image_info(
    image_processor,
    data_type_key: str = "image",
    mm_count: int = 1,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
):
    # Limit min / max pixels unless they're explicitly provided
    if min_pixels is None:
        min_pixels = max(image_processor.min_pixels, 28 * 28)
    if max_pixels is None:
        max_pixels = min(image_processor.max_pixels, 1280 * 28 * 28)

    return _get_vision_info(
        image_processor,
        height=9999999,
        width=9999999,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        data_type_key=data_type_key,
        mm_count=mm_count,
    )


def get_max_qwen2_vl_mm_tokens(ctx: InputContext,
                               data_type_key: str,
                               *,
                               min_pixels=None,
                               max_pixels=None) -> int:
    image_processor = cached_get_image_processor(ctx.model_config.model)
    max_resized_height, max_resized_width, max_llm_image_tokens = \
        _get_max_image_info(image_processor, data_type_key=data_type_key,
                            mm_count=1, min_pixels=min_pixels,
                            max_pixels=max_pixels)
    return max_llm_image_tokens


get_max_qwen2_vl_image_tokens = partial(get_max_qwen2_vl_mm_tokens,
                                        data_type_key="image")
get_max_qwen2_vl_video_tokens = partial(get_max_qwen2_vl_mm_tokens,
                                        data_type_key="video")


def dummy_data_for_qwen2_vl(
    ctx: InputContext,
    seq_len: int,
    mm_counts: Mapping[str, int],
    *,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None
) -> Tuple[SequenceData, Optional[MultiModalDataDict]]:
    image_processor = cached_get_image_processor(ctx.model_config.model)

    num_images = mm_counts["image"]
    max_resized_height, max_resized_width, max_llm_image_tokens = \
        _get_max_image_info(image_processor, data_type_key="image",
                            mm_count=num_images, min_pixels=min_pixels,
                            max_pixels=max_pixels)
    if seq_len - max_llm_image_tokens - 2 < 0:
        raise RuntimeError(
            f"Qwen2-VL cannot process {num_images} images in a prompt, "
            "please increase max_model_len or reduce image limit by "
            "--limit-mm-per-prompt.")

    # Check video counts.
    num_videos = mm_counts["video"]
    max_resized_height, max_resized_width, max_llm_video_tokens = \
        _get_max_image_info(image_processor, data_type_key="video",
                            mm_count=num_videos, min_pixels=min_pixels,
                            max_pixels=max_pixels)
    if seq_len - max_llm_video_tokens - 2 < 0:
        raise RuntimeError(
            f"Qwen2-VL cannot process {num_videos} videos in a prompt, "
            "please increase max_model_len or reduce video limit by "
            "--limit-mm-per-prompt.")

    hf_config = ctx.get_hf_config(Qwen2VLConfig)

    dummy_seqdata = SequenceData.from_prompt_token_counts(
        (hf_config.vision_start_token_id, 1),
        (hf_config.image_token_id, max_llm_image_tokens),
        (hf_config.vision_end_token_id, 1),
        (0, seq_len - max_llm_image_tokens - 2),
    )

    dummy_image = Image.new("RGB", (max_resized_width, max_resized_height),
                            color=0)

    return dummy_seqdata, {
        "image": dummy_image if num_images == 1 else [dummy_image] * num_images
    }


def _get_llm_num_vision_tokens(
    mm_inputs: list,
    data_type_key: str,
    image_processor,
    min_pixels: int,
    max_pixels: int,
):
    """Get number of vision tokens of multimodal inputs.

    This method is derived from `transformers.models.qwen2_vl.
    image_processing_qwen2_vl.Qwen2VLImageProcessor._preprocess`.
    """
    image = to_numpy_array(mm_inputs[0])
    input_data_format = infer_channel_dimension_format(image)
    height, width = get_image_size(image, channel_dim=input_data_format)

    _, _, llm_num_vision_tokens = _get_vision_info(
        image_processor,
        height=height,
        width=width,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        do_resize=image_processor.do_resize,
        data_type_key=data_type_key,
        mm_count=len(mm_inputs),
    )
    return llm_num_vision_tokens


def _expand_pad_tokens(inputs: list, token_id: int, make_batched_fn: Callable,
                       data_type_key: str, image_processor: Any,
                       prompt_token_ids: List[int], min_pixels: Optional[int],
                       max_pixels: Optional[int]) -> List[int]:
    """
    Expand pad tokens for multi-modal inputs (e.g., images or videos).

    Args:
        inputs (list): The multi-modal inputs (e.g., images or videos).
        token_id (int): The token ID used to represent the multi-modal input.
        make_batched_fn (Callable): A function to batch the inputs.
        data_type_key (str): The type of the multi-modal input.
        image_processor (Any): The image processor used to process the inputs.
        prompt_token_ids (List[int]): The list of token IDs in the prompt.
        min_pixels (int): min pixels to used for img processing
        max_pixels (int): max pixels to be used for img processing

    Returns:
        List[int]: The list of token IDs for the multi-modal inputs.
    """
    indices = [
        idx for idx, token in enumerate(prompt_token_ids) if token == token_id
    ]
    inputs = make_batched_fn(inputs)
    assert len(indices) == len(inputs)

    prompt_token_ids_with_data = []
    for cnt, data in enumerate(inputs):
        num_tokens = _get_llm_num_vision_tokens(
            [data] if data_type_key == "image" else data,
            data_type_key=data_type_key,
            image_processor=image_processor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if cnt == 0:
            end_idx = indices[cnt]
            non_data_tokens = prompt_token_ids[:end_idx]
        else:
            non_data_tokens = prompt_token_ids[indices[cnt - 1] +
                                               1:indices[cnt]]
        prompt_token_ids_with_data.extend(non_data_tokens)
        prompt_token_ids_with_data.extend(token_id for _ in range(num_tokens))
    prompt_token_ids_with_data.extend(prompt_token_ids[indices[-1] + 1:])
    return prompt_token_ids_with_data


def input_processor_for_qwen2_vl(
    ctx: InputContext,
    inputs: DecoderOnlyInputs,
    *,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> DecoderOnlyInputs:
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None:
        return inputs

    image_inputs = multi_modal_data.get("image", None)
    video_inputs = multi_modal_data.get("video", None)

    processor = cached_get_processor(ctx.model_config.model)
    image_processor = processor.image_processor
    # Apply processor kwarg overrides for image processor options
    min_pixels = min_pixels if min_pixels else image_processor.min_pixels
    max_pixels = max_pixels if max_pixels else image_processor.max_pixels

    model_config = ctx.model_config
    hf_config = ctx.get_hf_config(Qwen2VLConfig)

    # To avoid redundant processing of vision objects (resize, rescale, etc.),
    # we extract code of calculating number of vision tokens from
    # `transformers.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor`.
    #
    # The following code is equivalent to:
    #    prompt = inputs["prompt"]
    #    inputs = processor(text=[prompt],
    #                       images=image_inputs,
    #                       videos=video_inputs,
    #                       padding=True,
    #                       return_tensors="pt")
    #    prompt_token_ids = inputs["input_ids"][0].tolist()

    tokenizer = cached_get_tokenizer(
        model_config.tokenizer,
        trust_remote_code=model_config.trust_remote_code)

    prompt_token_ids = inputs["prompt_token_ids"]

    # Expand image pad tokens.

    if image_inputs is not None:
        if isinstance(image_inputs, dict):
            prompt_token_ids_with_image = []
            image_indices = [
                idx for idx, token in enumerate(prompt_token_ids)
                if token == hf_config.image_token_id
            ]
            image_cnt = len(image_indices)
            embed_dim = image_inputs.get('image_embeds').size(0)
            assert embed_dim % image_cnt == 0
            num_pad_tokens = embed_dim // image_cnt
            for idx, token in enumerate(prompt_token_ids):
                if idx in image_indices:
                    prompt_token_ids_with_image.extend([token] *
                                                       num_pad_tokens)
                else:
                    prompt_token_ids_with_image.append(token)
            prompt_token_ids = prompt_token_ids_with_image
        else:
            prompt_token_ids = _expand_pad_tokens(image_inputs,
                                                  hf_config.image_token_id,
                                                  make_batched_images,
                                                  "image",
                                                  image_processor,
                                                  prompt_token_ids,
                                                  min_pixels=min_pixels,
                                                  max_pixels=max_pixels)

    if video_inputs is not None:
        prompt_token_ids = _expand_pad_tokens(video_inputs,
                                              hf_config.video_token_id,
                                              make_batched_videos,
                                              "video",
                                              image_processor,
                                              prompt_token_ids,
                                              min_pixels=min_pixels,
                                              max_pixels=max_pixels)

    prompt = inputs.get("prompt")
    if prompt is None:
        prompt = tokenizer.decode(prompt_token_ids)

    return token_inputs(
        prompt_token_ids=prompt_token_ids,
        prompt=prompt,
        multi_modal_data=multi_modal_data,
    )


@MULTIMODAL_REGISTRY.register_image_input_mapper(
    image_input_mapper_for_qwen2_vl)
@MULTIMODAL_REGISTRY.register_input_mapper("video",
                                           video_input_mapper_for_qwen2_vl)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_qwen2_vl_image_tokens)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "video", get_max_qwen2_vl_video_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_qwen2_vl)
@INPUT_REGISTRY.register_input_processor(input_processor_for_qwen2_vl)
class Qwen2VLForConditionalGeneration(nn.Module, SupportsMultiModal,
                                      SupportsPP):

    def __init__(self,
                 config: Qwen2VLConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()

        assert not cache_config.enable_prefix_caching, \
            "Qwen2-VL currently does not support prefix caching"

        self.config = config
        self.multimodal_config = multimodal_config

        self.visual = Qwen2VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),

            # NOTE: Qwen2-VL vision encoder does not support any
            # quantization method now.
            quant_config=None,
        )

        self.model = Qwen2Model(config, cache_config, quant_config)

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def _validate_and_reshape_mm_tensor(self,
                                        mm_input: Union[torch.Tensor,
                                                        List[torch.Tensor]],
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                 f"Got ndim: {mm_input.ndim}")
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
                                           data=pixel_values,
                                           image_grid_thw=image_grid_thw)

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds")

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return Qwen2VLImageEmbeddingInputs(type="image_embeds",
                                               data=image_embeds)

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Optional[Qwen2VLVideoInputs]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None:
            return None

        pixel_values_videos = self._validate_and_reshape_mm_tensor(
            pixel_values_videos, "video pixel values")
        video_grid_thw = self._validate_and_reshape_mm_tensor(
            video_grid_thw, "video grid_thw")

        return Qwen2VLVideoInputs(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

    def _process_image_input(self,
                             image_input: Qwen2VLImageInputs) -> torch.Tensor:
        if image_input["type"] == "image_embeds":
            return image_input["data"].type(self.visual.dtype)

        pixel_values = image_input["data"].type(self.visual.dtype)
        image_embeds = self.visual(pixel_values,
                                   grid_thw=image_input["image_grid_thw"])
        return image_embeds

    def _process_video_input(self,
                             video_input: Qwen2VLVideoInputs) -> torch.Tensor:
        pixel_values_videos = video_input["pixel_values_videos"].type(
            self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos,
                                   grid_thw=video_input["video_grid_thw"])
        return video_embeds

    def _merge_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: torch.Tensor,
        placeholder_token_id: int,
    ) -> torch.Tensor:
        mask = (input_ids == placeholder_token_id)
        inputs_embeds[mask, :] = multimodal_embeddings
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
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
            input_ids = None
            inputs_embeds = None
        else:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None
            else:
                if uses_mrope(self.config):
                    assert positions.ndim == 2 and positions.size(0) == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.size()}")

                inputs_embeds = self.model.embed_tokens(input_ids)

                if image_input is not None:
                    image_embeds = self._process_image_input(image_input)
                    inputs_embeds = self._merge_multimodal_embeddings(
                        input_ids,
                        inputs_embeds,
                        image_embeds,
                        placeholder_token_id=self.config.image_token_id,
                    )

                if video_input is not None:
                    video_embeds = self._process_video_input(video_input)
                    inputs_embeds = self._merge_multimodal_embeddings(
                        input_ids,
                        inputs_embeds,
                        video_embeds,
                        placeholder_token_id=self.config.video_token_id,
                    )

                input_ids = None

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name and "qkv.weight" in name:
                    visual_num_heads = self.config.vision_config.num_heads
                    visual_embed_dim = self.config.vision_config.embed_dim
                    head_size = visual_embed_dim // visual_num_heads
                    loaded_weight = loaded_weight.view(3, visual_num_heads,
                                                       head_size,
                                                       visual_embed_dim)
                    loaded_weight = loaded_weight.transpose(0, 1)
                    loaded_weight = loaded_weight.reshape(-1, visual_embed_dim)
                elif "visual" in name and "qkv.bias" in name:
                    visual_num_heads = self.config.vision_config.num_heads
                    visual_embed_dim = self.config.vision_config.embed_dim
                    head_size = visual_embed_dim // visual_num_heads
                    loaded_weight = loaded_weight.view(3, visual_num_heads,
                                                       head_size)
                    loaded_weight = loaded_weight.transpose(0, 1)
                    loaded_weight = loaded_weight.reshape(-1)
                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                except KeyError:
                    raise ValueError(f"Unexpected weight: {name}") from None

                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
