# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
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
"""Inference-only Qwen2.5-VL model compatible with HuggingFace weights."""

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache, partial
from typing import Annotated, Any, Literal, TypeAlias

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLVisionConfig,
)

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.layer import maybe_get_vit_flash_attn_backend
from vllm.attention.ops.vit_attn_wrappers import (
    vit_flash_attn_wrapper,
    vit_torch_sdpa_wrapper,
    vit_xformers_attn_wrapper,
)
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.vision import should_torch_compile_mm_vit
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.evs import (
    compute_mrope_for_media,
    compute_retained_tokens_count,
    compute_retention_mask,
    recompute_mrope_positions,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargs,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdate
from vllm.sequence import IntermediateTensors
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsMultiModalPruning,
    SupportsPP,
    SupportsQuant,
)
from .qwen2_vl import Qwen2VLDummyInputsBuilder as Qwen2_5_VLDummyInputsBuilder
from .qwen2_vl import (
    Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo,
    apply_rotary_pos_emb_vision,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    cast_overflow_tensors,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import (
    conv3d_to_linear_weight,
    get_vit_attn_backend,
    run_dp_sharded_mrope_vision_model,
)

logger = init_logger(__name__)

# === Vision Inputs === #


class Qwen2_5_VLImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - np: Number of patches
        - ni: Number of images
        - cps: Number of channels * patch_size * patch_size

    Historical context:
        - pixel_values shape: (num_patches, num_channels * patch_size *
          patch_size)
        - image_grid_thw shape: (num_images, 3) in (grid_t, grid_h, grid_w)
          format.
    """

    type: Literal["pixel_values"]

    pixel_values: Annotated[
        torch.Tensor,
        TensorShape("np", "cps"),
    ]

    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ni", 3),
    ]


class Qwen2_5_VLImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - nf: Number of image features
        - hs: Hidden size
        - ni: Number of images

    Historical context:
        - image_embeds shape: (num_image_features, hidden_size)
        - num_image_features varies based on the number and resolution of the
          images.
        - hidden_size must match the hidden size of language model backbone.
        - image_grid_thw shape: (num_images, 3) in (grid_t, grid_h, grid_w)
          format
    """

    type: Literal["image_embeds"]

    image_embeds: Annotated[
        torch.Tensor,
        TensorShape("nf", "hs"),
    ]

    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ni", 3),
    ]


Qwen2_5_VLImageInputs: TypeAlias = (
    Qwen2_5_VLImagePixelInputs | Qwen2_5_VLImageEmbeddingInputs
)


class Qwen2_5_VLVideoPixelInputs(TensorSchema):
    """
    Dimensions:
        - np: Number of patches
        - nv: Number of videos
        - ctps: Number of channels * temporal_patch_size * patch_size *
          patch_size

    Historical context:
        - pixel_values_videos shape: (num_patches, num_channels *
          temporal_patch_size * patch_size * patch_size)
        - video_grid_thw shape: (num_videos, 3) in (grid_t, grid_h, grid_w)
          format
        - second_per_grid_ts: The video time interval (in seconds) for each
          grid along the temporal dimension in the 3D position IDs. Returned
          when `videos` is not `None`.
    """

    type: Literal["pixel_values_videos"]

    pixel_values_videos: Annotated[
        torch.Tensor,
        TensorShape("np", "ctps"),
    ]

    video_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("nv", 3),
    ]

    second_per_grid_ts: Annotated[
        torch.Tensor | None,
        TensorShape("nv"),
    ]


class Qwen2_5_VLVideoEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - nf: Number of video features
        - hs: Hidden size
        - nv: Number of videos

    Historical context:
        - video_embeds shape: (num_video_features, hidden_size)
        - num_video_features varies based on the number and resolution of the
          videos.
        - hidden_size must match the hidden size of language model backbone.
        - video_grid_thw shape: (num_videos, 3) in (grid_t, grid_h, grid_w)
          format
    """

    type: Literal["video_embeds"]

    video_embeds: Annotated[
        torch.Tensor,
        TensorShape("nf", "hs"),
    ]

    video_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("nv", 3),
    ]


Qwen2_5_VLVideoInputs: TypeAlias = (
    Qwen2_5_VLVideoPixelInputs | Qwen2_5_VLVideoEmbeddingInputs
)

# === Vision Encoder === #


class Qwen2_5_VisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,  # [gate_proj, up_proj]
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
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x_down, _ = self.down_proj(x)
        return x_down


class Qwen2_5_VisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        attn_backend: AttentionBackendEnum = AttentionBackendEnum.TORCH_SDPA,
        use_upstream_fa: bool = False,
        attn_backend_override: AttentionBackendEnum | None = None,
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
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
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )

        self.proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )
        self.attn_backend = attn_backend
        self.use_upstream_fa = use_upstream_fa
        self.attn_backend, self.flash_attn_varlen_func = (
            maybe_get_vit_flash_attn_backend(
                self.attn_backend,
                self.use_upstream_fa,
                attn_backend_override=attn_backend_override,
            )
        )
        # On ROCm with FLASH_ATTN backend, upstream flash_attn is used
        from vllm.platforms import current_platform

        if (
            current_platform.is_rocm()
            and self.attn_backend == AttentionBackendEnum.FLASH_ATTN
        ):
            self.use_upstream_fa = True
        if current_platform.is_xpu():
            self.use_upstream_fa = False
        self.is_flash_attn_backend = self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }

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
        rotary_pos_emb: torch.Tensor,
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
        seqlens: torch.Tensor,  # Only used for xFormers
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        q, k, v = (einops.rearrange(x, "s b ... -> b s ...") for x in (q, k, v))
        if rotary_pos_emb is not None:
            # [2 * b, s, heads, head_dim]
            qk_concat = torch.cat([q, k], dim=0)
            qk_rotated = apply_rotary_pos_emb_vision(qk_concat, rotary_pos_emb)
            q, k = torch.chunk(qk_rotated, 2, dim=0)

        if self.is_flash_attn_backend:
            context_layer = vit_flash_attn_wrapper(
                q,
                k,
                v,
                cu_seqlens,
                max_seqlen,
                batch_size,
                self.attn_backend == AttentionBackendEnum.ROCM_AITER_FA,
                self.use_upstream_fa,
            )
        elif self.attn_backend == AttentionBackendEnum.TORCH_SDPA:
            # Execute attention entry by entry for speed & less VRAM.
            from vllm.platforms import current_platform

            # Never remove the next contiguous logic
            # Without it, hallucinations occur with the backend
            if current_platform.is_rocm():
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()
            context_layer = vit_torch_sdpa_wrapper(
                q,
                k,
                v,
                cu_seqlens,
            )
        elif self.attn_backend == AttentionBackendEnum.XFORMERS:
            context_layer = vit_xformers_attn_wrapper(q, k, v, seqlens)

        output, _ = self.proj(context_layer)
        return output


@support_torch_compile(
    dynamic_arg_dims={
        "x": 0,
        "cu_seqlens": 0,
        "rotary_pos_emb": 0,
        "seqlens": 0,
    },
    mark_unbacked_dims={"seqlens": 0},
    enable_if=should_torch_compile_mm_vit,
)
class Qwen2_5_VisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        attn_backend: AttentionBackendEnum = AttentionBackendEnum.TORCH_SDPA,
        use_upstream_fa: bool = False,
        attn_backend_override: AttentionBackendEnum | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Qwen2_5_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_data_parallel=use_data_parallel,
            attn_backend=attn_backend,
            use_upstream_fa=use_upstream_fa,
            attn_backend_override=attn_backend_override,
        )
        self.mlp = Qwen2_5_VisionMLP(
            dim,
            mlp_hidden_dim,
            act_fn=act_fn,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
        seqlens: torch.Tensor,  # Only used for xFormers
    ) -> torch.Tensor:
        x_attn = self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
            seqlens=seqlens,
        )
        x_fused_norm, residual = self.norm2(x, residual=x_attn)
        x = residual + self.mlp(x_fused_norm)
        return x


@support_torch_compile(
    dynamic_arg_dims={
        "x": 0,
    },
    enable_if=should_torch_compile_mm_vit,
)
class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = ReplicatedLinear(
            in_channels * math.prod(kernel_size),
            hidden_size,
            bias=False,
            return_bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


@support_torch_compile(
    dynamic_arg_dims={
        "x": 0,
    },
    enable_if=should_torch_compile_mm_vit,
)
class Qwen2_5_VisionPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.ln_q = norm_layer(context_dim)

        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.0",
                return_bias=False,
                disable_tp=use_data_parallel,
            ),
            nn.GELU(),
            RowParallelLinear(
                self.hidden_size,
                d_model,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.2",
                return_bias=False,
                disable_tp=use_data_parallel,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)
        out = self.mlp(x)
        return out


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (
                self.theta
                ** (
                    torch.arange(
                        0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device
                    )
                    / self.dim
                )
            )
            seq = torch.arange(
                seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class Qwen2_5_VisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        attn_backend_override: AttentionBackendEnum | None = None,
    ) -> None:
        super().__init__()

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.use_data_parallel = use_data_parallel
        self.out_hidden_size = vision_config.out_hidden_size

        # args for get_window_index_thw
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2
        # TODO[@lucaskabela]: Investigate fixing this usage
        # see https://github.com/vllm-project/vllm/issues/27044
        # DO NOT MOVE THIS IMPORT
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("Qwen2_5_VisionPatchEmbed"):
            self.patch_embed = Qwen2_5_VisionPatchEmbed(
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                in_channels=in_channels,
                hidden_size=self.hidden_size,
            )

        norm_layer = partial(RMSNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        use_upstream_fa = False
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
            attn_backend_override=attn_backend_override,
        )

        self.attn_backend, self.flash_attn_varlen_func = (
            maybe_get_vit_flash_attn_backend(
                self.attn_backend,
                use_upstream_fa,
                attn_backend_override=attn_backend_override,
            )
        )

        if self.attn_backend not in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.XFORMERS,
            AttentionBackendEnum.ROCM_AITER_FA,
        }:
            raise RuntimeError(
                f"Qwen2.5-VL does not support {self.attn_backend} backend now."
            )

        with set_model_tag("Qwen2_5_VisionBlock"):
            self.blocks = nn.ModuleList(
                [
                    Qwen2_5_VisionBlock(
                        dim=self.hidden_size,
                        num_heads=self.num_heads,
                        mlp_hidden_dim=vision_config.intermediate_size,
                        act_fn=get_act_and_mul_fn(vision_config.hidden_act),
                        norm_layer=norm_layer,
                        quant_config=quant_config,
                        prefix=f"{prefix}.blocks.{layer_idx}",
                        use_data_parallel=use_data_parallel,
                        attn_backend=self.attn_backend,
                        use_upstream_fa=use_upstream_fa,
                        attn_backend_override=attn_backend_override,
                    )
                    for layer_idx in range(depth)
                ]
            )

        with set_model_tag("Qwen2_5_VisionPatchMerger"):
            self.merger = Qwen2_5_VisionPatchMerger(
                d_model=vision_config.out_hidden_size,
                context_dim=self.hidden_size,
                norm_layer=norm_layer,
                spatial_merge_size=self.spatial_merge_size,
                quant_config=quant_config,
                prefix=f"{prefix}.merger",
                use_data_parallel=use_data_parallel,
            )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rotary_pos_emb_thw(self, t, h, w):
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
        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
        max_size = max(h, w)
        rotary_pos_emb_full = self.rotary_pos_emb(max_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            rotary_pos_emb.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )

        return rotary_pos_emb

    def get_window_index_thw(self, grid_t, grid_h, grid_w):
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )

        llm_grid_h = grid_h // self.spatial_merge_size
        llm_grid_w = grid_w // self.spatial_merge_size
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w
        )
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit
        cu_seqlens_tmp = cu_seqlens_tmp.to(dtype=torch.int32)
        cu_seqlens_tmp = torch.unique_consecutive(cu_seqlens_tmp)

        return index_new, cu_seqlens_tmp

    @lru_cache(maxsize=1024)  # noqa: B019
    def get_rope_by_thw(self, t, h, w):
        window_index_thw, cu_seqlens_window_thw = self.get_window_index_thw(t, h, w)
        rotary_pos_emb_thw = self.rotary_pos_emb_thw(t, h, w)
        rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :]
        rotary_pos_emb_thw = rotary_pos_emb_thw.flatten(start_dim=0, end_dim=1)
        cu_seqlens_thw = torch.repeat_interleave(
            torch.tensor([h * w], dtype=torch.int32), t
        )
        return (
            rotary_pos_emb_thw,
            window_index_thw,
            cu_seqlens_window_thw,
            cu_seqlens_thw,
        )

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_seqlen = torch.zeros([], device=cu_seqlens.device)
        seqlens = torch.zeros(1, device=cu_seqlens.device)
        if self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        elif self.attn_backend == AttentionBackendEnum.XFORMERS:
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        return max_seqlen, seqlens

    @staticmethod
    def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
        # building the inverse permutation in O(n) time
        inv = torch.empty_like(perm, pin_memory=is_pin_memory_available())
        inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
        return inv

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        # patchify
        seq_len, _ = x.size()
        rotary_pos_emb = []
        window_index: list = []
        cu_window_seqlens: list = [torch.tensor([0], dtype=torch.int32)]
        cu_seqlens: list = []

        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        window_index_id = 0
        cu_window_seqlens_last = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            (
                rotary_pos_emb_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = self.get_rope_by_thw(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += t * llm_h * llm_w

            cu_seqlens_window_thw = cu_seqlens_window_thw + cu_window_seqlens_last
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            rotary_pos_emb.append(rotary_pos_emb_thw)

            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb = torch.cat(rotary_pos_emb)
        window_index = torch.cat(window_index)
        # compute reverse indices
        reverse_indices = self.invert_permutation(window_index)
        cu_window_seqlens = torch.cat(cu_window_seqlens)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_seqlens = torch.cat(cu_seqlens)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        # transformers
        # pre-compute seqlens for window/full attn to reduce cuMemcpy operations
        max_seqlen_full, seqlens_full = self.compute_attn_mask_seqlen(cu_seqlens)
        max_seqlen_window, seqlens_window = self.compute_attn_mask_seqlen(
            cu_window_seqlens
        )

        cu_seqlens = cu_seqlens.to(device=self.device, non_blocking=True)
        cu_window_seqlens = cu_window_seqlens.to(device=self.device, non_blocking=True)
        rotary_pos_emb = rotary_pos_emb.to(device=self.device, non_blocking=True)
        window_index = window_index.to(device=hidden_states.device, non_blocking=True)
        reverse_indices = reverse_indices.to(
            device=hidden_states.device, non_blocking=True
        )

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        hidden_states = hidden_states.unsqueeze(1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen_full
                seqlens_now = seqlens_full
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_seqlen_window
                seqlens_now = seqlens_window

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen_now,
                seqlens=seqlens_now,
            )

        # For Qwen2.5-VL-3B, float16 will overflow at last block
        # for long visual tokens sequences.
        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        # adapter
        hidden_states = self.merger(hidden_states)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
            ("mlp.gate_up_proj.", "mlp.gate_proj.", 0),
            ("mlp.gate_up_proj.", "mlp.up_proj.", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name.endswith("patch_embed.proj.weight"):
                loaded_weight = conv3d_to_linear_weight(loaded_weight)

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


class Qwen2_5_VLProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen2_5_VLConfig)

    def get_hf_processor(self, **kwargs: object) -> Qwen2_5_VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen2_5_VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


class Qwen2_5_VLMultiModalProcessor(Qwen2VLMultiModalProcessor):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            **super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs),
            second_per_grid_ts=MultiModalFieldConfig.batched("video"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        placeholder = {
            "image": vocab[hf_processor.image_token],
            "video": vocab[hf_processor.video_token],
        }

        merge_length = image_processor.merge_size**2

        def get_replacement_qwen2vl(item_idx: int, modality: str):
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length

            # EVS-specific code
            video_pruning_rate = self.info.ctx.get_mm_config().video_pruning_rate
            if (
                modality == "video"
                and video_pruning_rate is not None
                and video_pruning_rate > 0.0
            ):
                T, H, W = map(int, grid_thw)
                tokens_per_frame = (H // image_processor.merge_size) * (
                    W // image_processor.merge_size
                )
                num_tokens = compute_retained_tokens_count(
                    tokens_per_frame,
                    T,
                    video_pruning_rate,
                )
            # End of EVS-specific code

            return [placeholder[modality]] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement_qwen2vl, modality=modality),
            )
            for modality in ("image", "video")
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class Qwen2_5_VLForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
    SupportsEagle3,
    SupportsMultiModalPruning,
    SupportsMRoPE,
):
    merge_by_field_config = True
    multimodal_cpu_fields = {"image_grid_thw", "video_grid_thw"}

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            # mapping for original checkpoint
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    supports_encoder_tp_data = True

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {"image_grid_thw", "video_grid_thw", "second_per_grid_ts"},
        )
        image_grid_thw = [item.tolist() for item in kwargs.get("image_grid_thw", [])]
        video_grid_thw = [item.tolist() for item in kwargs.get("video_grid_thw", [])]
        second_per_grid_ts = kwargs.get("second_per_grid_ts", [])

        hf_config = self.config
        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(hf_config.vision_config, "tokens_per_second", 1.0)

        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id
        ).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            video_second_per_grid_t = 0.0
            if remain_images > 0:
                try:
                    ed_image = input_tokens.index(image_token_id, st)
                except ValueError:
                    ed_image = len(input_tokens) + 1
            else:
                ed_image = len(input_tokens) + 1
            if remain_videos > 0:
                try:
                    ed_video = input_tokens.index(video_token_id, st)
                except ValueError:
                    ed_video = len(input_tokens) + 1
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = image_grid_thw[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grid_thw[video_index]
                video_second_per_grid_t = 1.0
                if second_per_grid_ts:
                    video_second_per_grid_t = second_per_grid_ts[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

            t_index = (
                (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    * video_second_per_grid_t
                    * tokens_per_second
                )
                .long()
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
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()

        return llm_positions, mrope_position_delta

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"

        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen2_5_VLConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.config = config
        self.vllm_config = vllm_config
        self.multimodal_config = multimodal_config
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        if multimodal_config.get_limit_per_prompt(
            "image"
        ) or multimodal_config.get_limit_per_prompt("video"):
            attn_backend_override = (
                multimodal_config.mm_encoder_attn_backend
                if multimodal_config is not None
                else None
            )
            self.visual = Qwen2_5_VisionTransformer(
                vision_config=config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                use_data_parallel=self.use_data_parallel,
                attn_backend_override=attn_backend_override,
            )
        else:
            self.visual = None

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.language_model.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )

        if video_embeds is not None:
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
            )

    def _process_image_input(
        self, image_input: Qwen2_5_VLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"]
            with set_forward_context(None, self.vllm_config):
                if self.use_data_parallel:
                    return run_dp_sharded_mrope_vision_model(
                        self.visual, pixel_values, grid_thw_list, rope_type="rope_3d"
                    )
                else:
                    image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _postprocess_image_embeds_evs(
        self,
        image_embeds_split: tuple[torch.Tensor, ...],
        image_input: Qwen2_5_VLImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        """
        Append mrope positions for each for images.
        This is necessary to recover correct mrope
        positions after video pruning

        Args:
            image_embeds_split: Tuple of image embeddings for
                each image item.
            image_input: Image input data.

        Returns:
            Tuple of image embeddings for each image item.
            Resulting embeddings will have extra 4 channels for
            computed mrope positions.
        """
        merge_size = self.visual.spatial_merge_size
        grid_thw = image_input["image_grid_thw"]
        grid_thw_list = grid_thw.tolist()
        image_embeds_out = []
        for emb, size in zip(image_embeds_split, grid_thw_list):
            positions = compute_mrope_for_media(size, merge_size).to(emb.device)
            emb = torch.cat([emb, positions], dim=1)
            image_embeds_out.append(emb)
        image_embeds_split = image_embeds_out
        return tuple(image_embeds_split)

    def _process_video_input(
        self, video_input: Qwen2_5_VLVideoInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"]
            with set_forward_context(None, self.vllm_config):
                if self.use_data_parallel:
                    return run_dp_sharded_mrope_vision_model(
                        self.visual,
                        pixel_values_videos,
                        grid_thw_list,
                        rope_type="rope_3d",
                    )
                else:
                    video_embeds = self.visual(
                        pixel_values_videos, grid_thw=grid_thw_list
                    )

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)

    def _postprocess_video_embeds_evs(
        self,
        video_embeds_split: tuple[torch.Tensor, ...],
        video_input: Qwen2_5_VLVideoInputs,
    ) -> tuple[torch.Tensor, ...]:
        """
        Prunes video embeddings via Efficient Video Sampling (EVS)
        and then appends mrope positions for each retained embeddings

        Args:
            video_embeds_split: Tuple of video embeddings for each video item.
            video_input: Video input data.

        Returns:
            Tuple of video embeddings for each video item.
            Resulting embeddings will have extra 4 channels for
            computed mrope positions.
        """
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()
        merge_size = self.visual.spatial_merge_size

        # Cast to long to match the original code
        # https://github.com/huggingface/transformers/blob/41980ce93e775f6c88500c51c8db7946fc6a2add/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py#L491 # noqa
        second_per_grid_ts = video_input["second_per_grid_ts"].long()
        tokens_per_second = self.config.vision_config.tokens_per_second

        video_embeds_out = []
        for emb, size, video_second_per_grid_t in zip(
            video_embeds_split, grid_thw_list, second_per_grid_ts
        ):
            # For each video, we compute retention mask using EVS
            retention_mask = compute_retention_mask(
                emb,
                size,
                spatial_merge_size=self.visual.spatial_merge_size,
                q=self.video_pruning_rate,
            )
            positions = compute_mrope_for_media(
                size,
                merge_size,
                tokens_per_second=tokens_per_second,
                video_second_per_grid=video_second_per_grid_t.item(),
            ).to(emb.device)

            emb = emb[retention_mask]
            positions = positions[retention_mask]
            emb = torch.cat([emb, positions], dim=1)
            video_embeds_out.append(emb)
        return tuple(video_embeds_out)

    def recompute_mrope_positions(
        self,
        input_ids: list[int],
        multimodal_embeddings: tuple[torch.Tensor, ...],
        mrope_positions: torch.LongTensor,
        num_computed_tokens: int,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, int]:
        """
        Update part of input mrope positions (starting with
        num_computed_tokens index). Original mrope_positions are computed
        for unpruned sequence and becomes incorrect once pruning occurs,
        so once we prune media tokens we should reflect this in the
        mrope_positions before we feed it to LLM.

        Args:
            input_ids: (N,) All input tokens of the prompt (Containing
                entire sequence).
            multimodal_embeddings: Tuple of multimodal embeddings.
            mrope_positions: Existing mrope positions (3, N) for entire
                sequence
            num_computed_tokens: A number of computed tokens so far.

        Returns:
            Tuple of (multimodal_embeddings, mrope_positions,
                mrope_position_delta).
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        # Device
        device = (
            multimodal_embeddings[0].device
            if len(multimodal_embeddings)
            else mrope_positions.device
        )

        # Tensors
        input_ids_t = torch.as_tensor(input_ids, device=device, dtype=torch.long)

        mm_embeddings_out = [mm[:, :-4] for mm in multimodal_embeddings]
        mm_embeddings_pos = [
            mm[:, -4:].permute(1, 0).long() for mm in multimodal_embeddings
        ]

        positions, mrope_positions_delta = recompute_mrope_positions(
            input_ids_t,
            mm_embeddings_pos,
            mrope_positions,
            num_computed_tokens,
            vision_start_token_id,
            image_token_id,
            video_token_id,
        )

        return tuple(mm_embeddings_out), positions, mrope_positions_delta

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

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                if self.is_multimodal_pruning_enabled:
                    image_embeddings = self._postprocess_image_embeds_evs(
                        image_embeddings, multimodal_input
                    )
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                if self.is_multimodal_pruning_enabled:
                    video_embeddings = self._postprocess_video_embeds_evs(
                        video_embeddings, multimodal_input
                    )
                multimodal_embeddings += tuple(video_embeddings)
        return multimodal_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen2.5-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch. **NOTE**: If mrope is enabled (default setting for
                Qwen2.5-VL opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
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
