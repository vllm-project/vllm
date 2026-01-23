# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/Glm4v/modeling_Glm4v.py
# Copyright 2025 The vLLM team.
# Copyright 2025 The ZhipuAI Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
"""Inference-only GLM-4V model compatible with HuggingFace weights."""

import itertools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BatchFeature, Glm4vProcessor
from transformers.models.glm4v.configuration_glm4v import Glm4vVisionConfig
from transformers.models.glm4v.image_processing_glm4v import (
    Glm4vImageProcessor,
    smart_resize,
)
from transformers.models.glm4v.video_processing_glm4v import Glm4vVideoProcessor
from transformers.video_utils import VideoMetadata

from vllm.config import MultiModalConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions, VideoDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size, parallel_state
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer, Conv3dLayer
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    VideoItem,
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from ..layers.activation import SiluAndMul
from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from .qwen2_vl import _create_qwen2vl_field_factory
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import (
    get_vit_attn_backend,
    run_dp_sharded_mrope_vision_model,
)

logger = init_logger(__name__)

# For profile run
_MAX_FRAMES_PER_VIDEO = 600

# === Vision Inputs === #


class Glm4vImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - np: Number of patches
        - cpp: Number of channels * patch_size * patch_size
        - ni: Number of images
        - g: Grid dimensions (3 for grid_t, grid_h, grid_w)
    """

    type: Literal["pixel_values"] = "pixel_values"

    pixel_values: Annotated[torch.Tensor, TensorShape("np", "cpp")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


class Glm4vImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - f: Number of image features (varies based on image resolution)
        - h: Hidden size (must match language model backbone)
        - n: Number of images
        - g: Grid dimensions (3 for grid_t, grid_h, grid_w)
    """

    type: Literal["image_embeds"] = "image_embeds"

    image_embeds: Annotated[torch.Tensor, TensorShape("f", "h")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("n", 3)]


Glm4vImageInputs: TypeAlias = Glm4vImagePixelInputs | Glm4vImageEmbeddingInputs


class Glm4vVideoPixelInputs(TensorSchema):
    """
    Dimensions:
        - np: Number of patches
        - ctpp: Number of channels * temporal_patch_size *
            patch_size * patch_size
        - f: Number of frames
        - g: Grid dimensions (3 for grid_t which is usually 1 for processed
          video, grid_h, grid_w)
    """

    type: Literal["pixel_values_videos"] = "pixel_values_videos"

    pixel_values_videos: Annotated[torch.Tensor, TensorShape("np", "ctpp")]
    video_grid_thw: Annotated[torch.Tensor, TensorShape("f", 3)]


class Glm4vVideoEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - p: Number of video patches across all frames
        - h: Hidden size (must match language model backbone)
        - f: Number of frames
        - g: Grid dimensions (3 for grid_t which is usually 1 for processed
          video, grid_h, grid_w)
    """

    type: Literal["video_embeds"] = "video_embeds"

    video_embeds: Annotated[torch.Tensor, TensorShape("p", "h")]
    video_grid_thw: Annotated[torch.Tensor, TensorShape("f", 3)]


Glm4vVideoInputs: TypeAlias = Glm4vVideoPixelInputs | Glm4vVideoEmbeddingInputs

# ==== Vision Encoder ==== #


class Glm4vVisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        use_data_parallel = (
            multimodal_config.mm_encoder_tp_mode == "data"
            if multimodal_config
            else False
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=use_data_parallel,
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=use_data_parallel,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


def all_gather_interleave(local_tensor, hidden_size: int, tp_size: int):
    """All-gather the input tensor interleavely across model parallel group."""
    import torch.distributed as dist

    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(tp_size)]
    dist.all_gather(
        gathered_tensors,
        local_tensor,
        group=parallel_state.get_tp_group().device_group,
    )

    gathered_tensors_split = [
        torch.split(tensor, hidden_size // tp_size, -1) for tensor in gathered_tensors
    ]
    ordered_tensors = [
        tensor for pair in zip(*gathered_tensors_split) for tensor in pair
    ]
    result_tensor = torch.cat(ordered_tensors, dim=-1)
    return result_tensor


class Glm4vVisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        use_data_parallel = (
            multimodal_config.mm_encoder_tp_mode == "data"
            if multimodal_config
            else False
        )
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = (
            0 if use_data_parallel else parallel_state.get_tensor_model_parallel_rank()
        )
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size
        )

        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=False,
            quant_config=quant_config,
            # Change qkv prefix to align with GLM-4.5V-FP8 quantization cfg
            prefix=f"{prefix}.qkv_proj" if quant_config else f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )
        self.proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            bias=False,
            disable_tp=use_data_parallel,
        )

        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            scale=self.hidden_size_per_attention_head**-0.5,
            multimodal_config=multimodal_config,
        )

        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # [s, b, 3 * head * head_dim]
        seq_len, bs, _ = qkv.shape

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head * head_dim]
        q, k, v = qkv.chunk(3, dim=2)

        # 3 * [s, b, head * head_dim] -> 3 * [s, b, head, head_dim]
        new_shape = (
            seq_len,
            bs,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v))
        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            # [2 * b, s, heads, head_dim]
            qk_concat = torch.cat([q, k], dim=0)
            qk_rotated = self.apply_rotary_emb(
                qk_concat,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
            )
            q, k = torch.chunk(qk_rotated, 2, dim=0)

        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        context_layer = rearrange(context_layer, "b s h d -> s b (h d)").contiguous()

        output, _ = self.proj(context_layer)
        return output


class Glm4vVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Glm4vVisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = Glm4vVisionMLP(
            dim,
            mlp_hidden_dim,
            bias=False,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        x_attn = self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
        )
        x_fused_norm, residual = self.norm2(x, residual=x_attn)
        x = residual + self.mlp(x_fused_norm)

        return x


class Glm4vVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 1,
        in_channels: int = 3,
        hidden_size: int = 1536,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = Conv3dLayer(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.proj(x).view(L, self.hidden_size)
        return x


class Glm4vPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = (
            multimodal_config.mm_encoder_tp_mode == "data"
            if multimodal_config
            else False
        )
        self.hidden_size = d_model
        self.proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=bias,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )
        self.post_projection_norm = nn.LayerNorm(self.hidden_size)
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[context_dim] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=use_data_parallel,
        )
        self.down_proj = RowParallelLinear(
            context_dim,
            self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=use_data_parallel,
        )
        self.act_fn = SiluAndMul()
        self.extra_activation_func = nn.GELU()

    def forward(self, x: torch.Tensor):
        x, _ = self.proj(x)
        x = self.extra_activation_func(self.post_projection_norm(x))
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm4vVisionEmbeddings(nn.Module):
    def __init__(self, config: Glm4vVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self, embeddings, lengths, image_shapes, h_coords, w_coords
    ) -> torch.Tensor:
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]
        device = pos_embed_weight.device

        # Move coordinates to correct device
        h_coords, w_coords = h_coords.to(device), w_coords.to(device)

        # Handle empty sequence case
        if total_seq == 0:
            adapted_pos_embed = torch.empty(
                0, hidden_size, device=device, dtype=pos_embed_weight.dtype
            )
        else:
            # Convert inputs to tensors if needed
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths, device=device, dtype=torch.long)
            if not isinstance(image_shapes, torch.Tensor):
                image_shapes = torch.tensor(
                    image_shapes, device=device, dtype=torch.long
                )

            # Prepare 2D position embedding
            orig_size_sq = pos_embed_weight.shape[0]
            orig_size = int(orig_size_sq**0.5)
            pos_embed_2d = (
                pos_embed_weight.view(orig_size, orig_size, hidden_size)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )

            # Calculate target dimensions for each patch
            # Add bounds checking for data parallel mode
            if len(lengths) > image_shapes.shape[0]:
                # In data parallel mode, some GPUs might not have all
                # image shapes
                # Use available image shapes, cycling if necessary
                target_h_list = []
                target_w_list = []
                for i in range(len(lengths)):
                    # Cycle through available shapes
                    shape_idx = i % image_shapes.shape[0]
                    target_h_list.append(image_shapes[shape_idx, 1].repeat(lengths[i]))
                    target_w_list.append(image_shapes[shape_idx, 2].repeat(lengths[i]))
                target_h = torch.cat(target_h_list).to(
                    device=device, dtype=torch.float32
                )
                target_w = torch.cat(target_w_list).to(
                    device=device, dtype=torch.float32
                )
            else:
                target_h = torch.cat(
                    [image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]
                ).to(device=device, dtype=torch.float32)
                target_w = torch.cat(
                    [image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]
                ).to(device=device, dtype=torch.float32)

            # Normalize coordinates to [-1, 1] range for grid_sample
            h_coords = h_coords.to(device=device, dtype=torch.float32)
            w_coords = w_coords.to(device=device, dtype=torch.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

            # Create sampling grid
            grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

            # Perform bicubic interpolation
            interpolated_embed_fp32 = F.grid_sample(
                pos_embed_2d,
                grid,
                mode="bicubic",
                align_corners=False,
                padding_mode="border",
            )

            # Reshape and convert back to original dtype
            adapted_pos_embed_fp32 = (
                interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
            )
            adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype).to(
                embeddings.device
            )

        # Add adapted position encoding to embeddings
        embeddings = embeddings + adapted_pos_embed
        return embeddings


class Glm4vVisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config: Glm4vVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert multimodal_config is not None, "multimodal_config must be provided"

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.out_hidden_size = vision_config.out_hidden_size

        self.patch_embed = Glm4vVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        norm_layer = partial(RMSNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
        )
        self.blocks = nn.ModuleList(
            [
                Glm4vVisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.out_hidden_size,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    multimodal_config=multimodal_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(depth)
            ]
        )
        self.merger = Glm4vPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.intermediate_size,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            bias=False,
            prefix=f"{prefix}.merger",
        )
        self.embeddings = Glm4vVisionEmbeddings(vision_config)

        self.post_conv_layernorm = RMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )
        self.downsample = Conv2dLayer(
            in_channels=vision_config.hidden_size,
            out_channels=vision_config.out_hidden_size,
            kernel_size=vision_config.spatial_merge_size,
            stride=vision_config.spatial_merge_size,
        )
        self.post_layernorm = RMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )

        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
            attn_backend_override=multimodal_config.mm_encoder_attn_backend,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(
        self, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()

        # Use pre-computed cos_sin_cache from RotaryEmbedding
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)
        return cos_combined, sin_combined, pos_ids

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor | None:
        max_seqlen = None
        if (
            self.attn_backend == AttentionBackendEnum.FLASH_ATTN
            or self.attn_backend == AttentionBackendEnum.ROCM_AITER_FA
        ):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        return max_seqlen

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
    ) -> torch.Tensor:
        if isinstance(grid_thw, list):
            grid_thw = torch.tensor(grid_thw, dtype=torch.int32)

        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)
        x = self.post_conv_layernorm(x)

        # compute position embedding
        rotary_pos_emb_cos, rotary_pos_emb_sin, image_type_ids = self.rot_pos_emb(
            grid_thw
        )
        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
        cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)

        # pre-compute max_seqlen for attn mask to reduce cuMemcpy operations
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        x = self.embeddings(
            x, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1]
        )

        # transformers
        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,
            )

        # adapter
        x = self.post_layernorm(x)

        x = x.view(-1, self.spatial_merge_size, self.spatial_merge_size, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x).view(-1, self.out_hidden_size)
        x = self.merger(x)

        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Glm4vProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": 1}

    def get_image_processor(self, **kwargs: object) -> Glm4vImageProcessor:
        return self.get_hf_processor(**kwargs).image_processor

    def get_video_processor(self, **kwargs: object) -> Glm4vVideoProcessor:
        return self.get_hf_processor(**kwargs).video_processor

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 16,
        do_resize: bool = True,
        max_image_pixels: int = 28 * 28 * 2 * 30000,
    ) -> tuple[ImageSize, int]:
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size
        if do_resize:
            resized_height, resized_width = smart_resize(
                num_frames=num_frames
                if num_frames > temporal_patch_size
                else temporal_patch_size,
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                max_pixels=max_image_pixels,
            )
            preprocessed_size = ImageSize(width=resized_width, height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width, height=image_height)

        # NOTE: Frames are padded to be divisible by `temporal_patch_size`
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L294
        padded_num_frames = num_frames + num_frames % temporal_patch_size

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        max_image_size, _ = self._get_vision_info(
            image_width=9999999, image_height=9999999
        )
        return max_image_size

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        _, num_image_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            max_image_pixels=28 * 28 * 2 * 6144,
        )
        return num_image_tokens

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
        )

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
    ) -> int:
        _, num_video_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=num_frames,
            max_image_pixels=28 * 28 * 2 * 30000,
        )
        return num_video_tokens

    def _get_max_video_frames(self, max_tokens: int) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        num_frames = 0

        while True:
            next_num_frames = num_frames + 1
            next_max_tokens = self.get_num_video_tokens(
                image_width=target_width,
                image_height=target_height,
                num_frames=next_num_frames,
            )
            if next_max_tokens > max_tokens or next_max_tokens == 0:
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
        max_total_frames = self._get_max_video_frames(seq_len - max_image_tokens)
        max_frames_per_video = min(
            max_total_frames // max(max_videos, 1), _MAX_FRAMES_PER_VIDEO
        )

        return max(max_frames_per_video, 1)

    def _get_video_second_idx_glm4v(
        self, metadata: dict[str, Any], total_frames: int
    ) -> list[int]:
        video_processor = self.get_video_processor()

        video_fps = metadata.get("fps", video_processor.fps)
        meta_frames = metadata.get("total_num_frames", total_frames)
        max_frame_idx = meta_frames - 1
        duration = metadata.get("duration", round(max_frame_idx / video_fps) + 1)
        do_sample_frames = metadata["do_sample_frames"]
        if not do_sample_frames:
            frame_indices = metadata["frames_indices"]
        else:
            if duration <= video_processor.max_duration:
                n = int(math.floor(duration * video_processor.fps))
                frame_indices = [
                    min(
                        max_frame_idx,
                        int(math.ceil(i * video_fps / video_processor.fps)),
                    )
                    for i in range(n)
                ]
            else:
                num_samples = int(video_processor.max_duration * video_processor.fps)
                if num_samples >= meta_frames:
                    frame_indices = list(range(meta_frames))
                else:
                    target_seconds = np.linspace(
                        0, duration, num_samples, endpoint=True
                    )
                    frame_indices = [
                        min(max_frame_idx, int(math.ceil(t * video_fps)))
                        for t in target_seconds
                    ]

        seen, uniq = set(), []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                uniq.append(idx)
        if len(uniq) & 1:
            uniq.append(uniq[-1])
        frame_indices = uniq

        full_second_idxs = [int(idx / video_fps) for idx in frame_indices]
        timestamps_list = full_second_idxs[::2]
        selected_timestamps = []
        for idx in range(0, len(timestamps_list)):
            selected_timestamps.append(timestamps_list[idx])
        return selected_timestamps

    def _get_video_second_idx_glm46v(
        self, metadata: dict[str, Any], total_frames: int
    ) -> list[int]:
        video_processor = self.get_video_processor()

        video_fps = metadata["fps"]
        meta_frames = metadata.get("total_num_frames", total_frames)
        max_frame_idx = meta_frames - 1
        duration = metadata.get("duration", round(max_frame_idx / video_fps) + 1)

        do_sample_frames = metadata.get("do_sample_frames", True)
        if not do_sample_frames:
            frame_indices = metadata["frames_indices"]
        else:
            DYNAMIC_FPS_THRES = {30: 3, 300: 1, 2400: 0.5}
            MAX_FRAME_COUNT_DYNAMIC = 640
            MAX_DURATION = 2400

            effective_duration = min(duration, MAX_DURATION)
            if effective_duration <= 30:
                target_fps = DYNAMIC_FPS_THRES[30]
            elif effective_duration <= 300:
                target_fps = DYNAMIC_FPS_THRES[300]
            else:
                target_fps = DYNAMIC_FPS_THRES[2400]

            temporal_patch_size = getattr(video_processor, "temporal_patch_size", 1)
            extract_t = int(effective_duration * target_fps * temporal_patch_size)
            extract_t = min(extract_t, MAX_FRAME_COUNT_DYNAMIC)

            duration_per_frame = 1 / video_fps
            timestamps = [i * duration_per_frame for i in range(meta_frames)]
            max_second = int(duration)

            if meta_frames < extract_t:
                frame_indices = np.linspace(
                    0, meta_frames - 1, extract_t, dtype=int
                ).tolist()
            else:
                frame_indices = []
                current_second = 0.0
                inv_fps = 1 / (temporal_patch_size * target_fps)
                for frame_index in range(meta_frames):
                    if timestamps[frame_index] >= current_second:
                        current_second += inv_fps
                        frame_indices.append(frame_index)
                        if current_second >= max_second:
                            break

            if len(frame_indices) < extract_t:
                if len(frame_indices) == 0:
                    start, end = 0, max(meta_frames - 1, 0)
                else:
                    start, end = frame_indices[0], frame_indices[-1]
                frame_indices = np.linspace(start, end, extract_t, dtype=int).tolist()
            elif len(frame_indices) > extract_t:
                frame_indices = np.linspace(
                    0, meta_frames - 1, extract_t, dtype=int
                ).tolist()

        seen, uniq = set(), []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                uniq.append(idx)

        if len(uniq) & 1:
            uniq.append(uniq[-1])

        frame_indices = uniq
        full_second_idxs = [int(idx / video_fps) for idx in frame_indices]
        timestamps_list = full_second_idxs[::2]
        selected_timestamps = []
        for idx in range(len(timestamps_list)):
            selected_timestamps.append(timestamps_list[idx])
        return selected_timestamps

    def _construct_video_placeholder(
        self,
        video_array: np.ndarray,
        metadata: dict[str, Any],
        grid_thw: torch.Tensor,
    ) -> str:
        hf_processor = self.get_hf_processor()
        tokenizer = self.get_tokenizer()
        image_processor = hf_processor.image_processor

        hf_config = self.get_hf_config()
        boi_token_id = hf_config.image_start_token_id
        eoi_token_id = hf_config.image_end_token_id
        bov_token_id = hf_config.video_start_token_id
        eov_token_id = hf_config.video_end_token_id
        merge_length = image_processor.merge_size**2

        assert isinstance(grid_thw, torch.Tensor)
        timestamps = (
            self._get_video_second_idx_glm4v(metadata, len(video_array))
            if isinstance(hf_processor, Glm4vProcessor)
            else self._get_video_second_idx_glm46v(metadata, len(video_array))
        )

        timestamp_format = (
            "{}" if isinstance(hf_processor, Glm4vProcessor) else "{:.1f} seconds"
        )
        frames_idx_token = [
            tokenizer.encode(timestamp_format.format(i), add_special_tokens=False)
            for i in timestamps
        ]
        T, H, W = grid_thw
        num_tokens_per_frame = int(H * W) // merge_length
        placeholder = []
        placeholder.append(bov_token_id)
        for frame_idx in frames_idx_token:
            placeholder.append(boi_token_id)
            placeholder.extend([hf_processor.video_token_id] * num_tokens_per_frame)
            placeholder.append(eoi_token_id)
            placeholder.extend(frame_idx)
        placeholder.append(eov_token_id)

        return placeholder


class Glm4vDummyInputsBuilder(BaseDummyInputsBuilder[Glm4vProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        hf_config = self.info.get_hf_config()
        hf_processor = self.info.get_hf_processor()
        tokenizer = self.info.get_tokenizer()

        image_token: str = hf_processor.image_token
        video_token_ids = [
            hf_config.video_start_token_id,
            hf_processor.video_token_id,
            hf_config.video_end_token_id,
        ]
        video_token = tokenizer.decode(video_token_ids)

        return image_token * num_images + video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )

        image_overrides = mm_options.get("image") if mm_options else None
        video_overrides = mm_options.get("video") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "video": self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
                overrides=video_overrides,
            ),
        }

    def _get_dummy_videos(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
        overrides: VideoDummyOptions | None = None,
    ) -> list[VideoItem]:
        if overrides:
            if overrides.num_frames:
                if overrides.num_frames > num_frames:
                    logger.warning(
                        "video.num_frames override (%d) exceeds model's "
                        "maximum number of frames (%d), will be ignored",
                        overrides.num_frames,
                        num_frames,
                    )
                num_frames = min(num_frames, overrides.num_frames)
            if overrides.width:
                if overrides.width > width:
                    logger.warning(
                        "video.width override (%d) exceeds model's "
                        "maximum width (%d), will be ignored",
                        overrides.width,
                        width,
                    )
                width = min(width, overrides.width)
            if overrides.height:
                if overrides.height > height:
                    logger.warning(
                        "video.height override (%d) exceeds model's "
                        "maximum height (%d), will be ignored",
                        overrides.height,
                        height,
                    )
                height = min(height, overrides.height)

        num_frames = max(num_frames, 2)  # GLM 4.6V requires 2 frames
        video = np.full((num_frames, width, height, 3), 255, dtype=np.uint8)
        video_items = []
        for i in range(num_videos):
            video_metadata = {
                "fps": 2.0,
                "duration": num_frames / 2.0,
                "total_num_frames": num_frames,
                "frames_indices": [i for i in range(num_frames)],
                "video_backend": "opencv",
                "do_sample_frames": False,
            }
            video_item = (video.copy(), video_metadata)
            video_items.append(video_item)

        return video_items


class Glm4vMultiModalProcessor(BaseMultiModalProcessor[Glm4vProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(video_needs_metadata=True)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        processor = self.info.get_hf_processor(**mm_kwargs)

        # GLM-4.1V use `image_token_id` as video placeholder, we need to
        # replace it with `video_token_id` for video processing. So we
        # separate video processing from image processing.
        if (
            "videos" in mm_data
            and isinstance(mm_data["videos"], list)
            and len(mm_data["videos"]) > 0
        ):
            video_grid_thw_lst = []
            pixel_values_videos_lst = []
            for item in mm_data.pop("videos", []):
                video_array, metadata = item

                # don't update mm_kwargs inplace
                video_mm_kwargs = dict(**mm_kwargs)
                video_mm_kwargs["do_sample_frames"] = metadata.get(
                    "do_sample_frames", True
                )

                video_mm_data = dict()
                video_mm_data["videos"] = [[video_array]]

                unuse_metadata = ["do_sample_frames"]
                video_mm_data["video_metadata"] = [
                    [
                        VideoMetadata(
                            **{
                                k: metadata[k]
                                for k in metadata
                                if k not in unuse_metadata
                            }
                        )
                    ]
                ]

                video_outputs = super()._call_hf_processor(
                    prompt="<|begin_of_video|><|video|><|end_of_video|>",
                    mm_data=video_mm_data,
                    mm_kwargs=video_mm_kwargs,
                    tok_kwargs=tok_kwargs,
                )
                input_ids = video_outputs.pop("input_ids")
                input_ids[input_ids == processor.image_token_id] = (
                    processor.video_token_id
                )
                video_placeholder = processor.tokenizer.batch_decode(input_ids)[0]
                prompt = prompt.replace(
                    "<|begin_of_video|><|video|><|end_of_video|>",
                    video_placeholder,
                    1,
                )

                video_grid_thw_lst.append(video_outputs["video_grid_thw"])
                pixel_values_videos_lst.append(video_outputs["pixel_values_videos"])
            video_outputs = dict(
                pixel_values_videos=torch.cat(pixel_values_videos_lst),
                video_grid_thw=torch.cat(video_grid_thw_lst),
            )
        else:
            video_outputs = dict()

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        combined_outputs = dict(
            processed_outputs,
            **video_outputs,
        )
        return BatchFeature(combined_outputs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _create_qwen2vl_field_factory(
            self.info.get_hf_config().vision_config.spatial_merge_size
        )(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)

        merge_length = image_processor.merge_size**2

        def get_image_replacement_glm4v(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [hf_processor.image_token_id] * num_tokens

        def get_video_replacement_glm4v(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            video, metadata = mm_items["video"][item_idx]
            placeholder = self.info._construct_video_placeholder(
                video, metadata, grid_thw
            )
            return PromptUpdateDetails.select_token_id(
                placeholder,
                embed_token_id=hf_processor.video_token_id,
            )

        return [
            PromptReplacement(
                modality="image",
                target=hf_processor.image_token,
                replacement=get_image_replacement_glm4v,
            ),
            PromptReplacement(
                modality="video",
                target="<|begin_of_video|><|video|><|end_of_video|>",
                replacement=get_video_replacement_glm4v,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Glm4vMultiModalProcessor,
    info=Glm4vProcessingInfo,
    dummy_inputs=Glm4vDummyInputsBuilder,
)
class Glm4vForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP, SupportsMRoPE
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_up_proj"],
    }

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
        }
    )

    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|begin_of_image|><|image|><|end_of_image|>"
        if modality.startswith("video"):
            return "<|begin_of_video|><|video|><|end_of_video|>"

        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm4vVisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-5),
                quant_config=quant_config,
                multimodal_config=multimodal_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        if config.model_type == "glm4v":
            architectures = ["Glm4ForCausalLM"]
        elif config.model_type == "glm4v_moe":
            architectures = ["Glm4MoeForCausalLM"]
        else:
            architectures = None

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=architectures,
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Glm4vImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return Glm4vImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return Glm4vImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> Glm4vVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return Glm4vVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        if video_embeds is not None:
            return Glm4vVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
            )

    def _process_image_input(
        self, image_input: Glm4vImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.visual, pixel_values, grid_thw.tolist(), rope_type="rope_3d"
                )
            else:
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(
        self, video_input: Glm4vVideoInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype
            )
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.visual,
                    pixel_values_videos,
                    grid_thw.tolist(),
                    rope_type="rope_3d",
                )
            else:
                video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += tuple(video_embeddings)
        return multimodal_embeddings

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {"image_grid_thw", "video_grid_thw"},
        )
        image_grid_thw = [item.tolist() for item in kwargs.get("image_grid_thw", [])]
        video_grid_thw = [item.tolist() for item in kwargs.get("video_grid_thw", [])]

        hf_config = self.config
        image_token_id = hf_config.image_token_id
        video_start_token_id = hf_config.video_start_token_id
        video_end_token_id = hf_config.video_end_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        llm_pos_ids_list: list = []

        if image_grid_thw or video_grid_thw:
            input_token_type: list[str] = []
            video_check_flg = False
            for token in input_tokens:
                if token == video_start_token_id:
                    video_check_flg = True
                elif token == video_end_token_id:
                    video_check_flg = False

                if (token == image_token_id) and (video_check_flg is False):
                    input_token_type.append("image")
                elif (token == image_token_id) and (video_check_flg is True):
                    input_token_type.append("video")
                else:
                    input_token_type.append("text")

            input_type_group: list[tuple[str, int, int]] = []
            for key, group_iter in itertools.groupby(
                enumerate(input_token_type), lambda x: x[1]
            ):
                group_list = list(group_iter)
                start_index = group_list[0][0]
                end_index = group_list[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            video_frame_num = 1
            mm_data_idx = 0
            for modality_type, start_idx, end_idx in input_type_group:
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                if modality_type == "image":
                    t, h, w = image_grid_thw[mm_data_idx]
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t,
                        h // spatial_merge_size,
                        w // spatial_merge_size,
                    )

                    t_index = (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + st_idx
                    )
                    mm_data_idx += 1

                elif modality_type == "video":
                    t, h, w = (
                        video_frame_num,
                        *image_grid_thw[mm_data_idx][1:],
                    )
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t,
                        h // spatial_merge_size,
                        w // spatial_merge_size,
                    )

                    for t_idx in range(llm_grid_t):
                        t_index = (
                            torch.tensor(t_idx)
                            .view(-1, 1)
                            .expand(-1, llm_grid_h * llm_grid_w)
                            .flatten()
                        )
                        h_index = (
                            torch.arange(llm_grid_h)
                            .view(1, -1, 1)
                            .expand(1, -1, llm_grid_w)
                            .flatten()
                        )
                        w_index = (
                            torch.arange(llm_grid_w)
                            .view(1, 1, -1)
                            .expand(1, llm_grid_h, -1)
                            .flatten()
                        )
                        llm_pos_ids_list.append(
                            torch.stack([t_index, h_index, w_index]) + st_idx
                        )

                    mm_data_idx += 1
                    video_frame_num += 1

                else:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    video_frame_num = 1

        else:
            text_len = len(input_tokens)
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1))

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        return llm_positions, mrope_position_delta

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for GLM-4V.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for GLM-4V
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Optional intermediate tensors for pipeline
                parallelism.
            inputs_embeds: Optional pre-computed input embeddings.
            **kwargs: Additional keyword arguments.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

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
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model.model",
            connector="visual.merger.",
            tower_model="visual.",
        )

    def get_num_mm_encoder_tokens(
        self,
        num_image_tokens: int,
    ) -> int:
        merge_size = self.config.vision_config.spatial_merge_size
        return num_image_tokens * (merge_size**2)

    def get_num_mm_connector_tokens(
        self,
        num_vision_tokens: int,
    ) -> int:
        merge_size = self.config.vision_config.spatial_merge_size
        return num_vision_tokens // (merge_size**2)


@MULTIMODAL_REGISTRY.register_processor(
    Glm4vMultiModalProcessor,
    info=Glm4vProcessingInfo,
    dummy_inputs=Glm4vDummyInputsBuilder,
)
class Glm4vMoeForConditionalGeneration(Glm4vForConditionalGeneration):
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
