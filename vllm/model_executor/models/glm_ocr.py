# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/Glm4v/modeling_Glm4v.py
# Copyright 2026 The ZhipuAI Team.
# Copyright 2026 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only GLM-OCR model compatible with HuggingFace weights."""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange

if TYPE_CHECKING:
    from transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrVisionConfig

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size, parallel_state
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb,
)
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration,
    Glm4vMultiModalProcessor,
    Glm4vPatchMerger,
    Glm4vProcessingInfo,
    Glm4vVisionBlock,
    Glm4vVisionMLP,
    Glm4vVisionPatchEmbed,
    Glm4vVisionTransformer,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .utils import (
    maybe_prefix,
)
from .vision import (
    get_vit_attn_backend,
    is_vit_use_data_parallel,
)

logger = init_logger(__name__)


class GlmOcrVisionMLP(Glm4vVisionMLP):
    pass


class GlmOcrVisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        use_data_parallel = is_vit_use_data_parallel()
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

        self.head_dim = embed_dim // num_heads

        self.q_norm = RMSNorm(self.head_dim, eps=1e-5)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-5)

        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
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
            bias=True,
            disable_tp=use_data_parallel,
        )

        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            scale=self.hidden_size_per_attention_head**-0.5,
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

        # RMSNorm on q, k
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(q_shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(k_shape)

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


class GlmOcrVisionBlock(Glm4vVisionBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            mlp_hidden_dim,
            norm_layer,
            quant_config,
            prefix,
        )
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = GlmOcrVisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = GlmOcrVisionMLP(
            dim,
            mlp_hidden_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )


class GlmOcrVisionPatchEmbed(Glm4vVisionPatchEmbed):
    pass


class GlmOcrPatchMerger(Glm4vPatchMerger):
    pass


class GlmOcrVisionTransformer(Glm4vVisionTransformer):
    def __init__(
        self,
        vision_config: "GlmOcrVisionConfig",
        norm_eps: float = 1e-5,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(vision_config, norm_eps, quant_config, prefix)

        del self.post_conv_layernorm
        del self.embeddings

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
                GlmOcrVisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(depth)
            ]
        )
        self.merger = GlmOcrPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.out_hidden_size * vision_config.in_channels,
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.merger",
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
        )

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


@MULTIMODAL_REGISTRY.register_processor(
    Glm4vMultiModalProcessor,
    info=Glm4vProcessingInfo,
    dummy_inputs=Glm4vDummyInputsBuilder,
)
class GlmOcrForConditionalGeneration(Glm4vForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = GlmOcrVisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-5),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )
