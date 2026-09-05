# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2026 The Cohere Team.
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
"""Inference-only Cohere Compass model compatible with HuggingFace weights."""

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import lru_cache, partial
from itertools import islice
from typing import Annotated, Any, Literal, TypeAlias

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature
from transformers.models.cohere_compass import (
    CohereCompassConfig,
    CohereCompassProcessor,
)
from transformers.models.cohere_compass.configuration_cohere_compass import (
    CohereCompassVisionConfig,
)
from transformers.models.cohere_compass.image_processing_cohere_compass import (
    CohereCompassImageProcessor,
)
from transformers.models.cohere_compass.image_processing_cohere_compass import (
    smart_resize as image_smart_resize,
)

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    parallel_state,
)
from vllm.distributed import utils as dist_utils
from vllm.inputs import ModalityData, MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    ImageItem,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ImageSize,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.video_prune.evs import (
    compute_mrope_for_media,
    recompute_mrope_positions,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import round_up
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphReplayBuffers

from ...utils.torch_utils import async_tensor_h2d
from .commandr import CohereForCausalLM, CohereMLP, LayerNorm
from .interfaces import (
    MultiModalEmbeddings,
    SupportsEagle,
    SupportsEagle3,
    SupportsEncoderCudaGraph,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsMultiModalPruning,
    SupportsPP,
    _require_is_multimodal,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from .vision import (
    get_fp8_padded_hidden_size,
    get_vit_attn_backend,
    is_vit_use_data_parallel,
    run_dp_sharded_mrope_vision_model,
)

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Triton kernel: fused bilinear position-embedding interpolation
# ---------------------------------------------------------------------------
# Replaces many small eager-mode CUDA kernels with a single launch.
# The spatial-merge reorder is baked into the index math so the output
# is ready to be added to the patch embeddings directly.
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _bilinear_pos_embed_kernel(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused bilinear pos-embed interpolation with spatial-merge reorder."""
        pid = tl.program_id(0)
        total_spatial = H * W
        spatial_idx = pid % total_spatial

        num_blocks_w = W // M_SIZE
        block_idx = spatial_idx // (M_SIZE * M_SIZE)
        local_idx = spatial_idx % (M_SIZE * M_SIZE)
        br = block_idx // num_blocks_w
        bc = block_idx % num_blocks_w
        lr = local_idx // M_SIZE
        lc = local_idx % M_SIZE
        row = br * M_SIZE + lr
        col = bc * M_SIZE + lc

        h_frac = row.to(tl.float32) * h_scale
        w_frac = col.to(tl.float32) * w_scale

        hf = tl.math.floor(h_frac).to(tl.int32)
        wf = tl.math.floor(w_frac).to(tl.int32)
        hc = tl.minimum(hf + 1, NUM_GRID - 1)
        wc = tl.minimum(wf + 1, NUM_GRID - 1)

        dh = h_frac - hf.to(tl.float32)
        dw = w_frac - wf.to(tl.float32)
        w11 = dh * dw
        w10 = dh - w11
        w01 = dw - w11
        w00 = 1.0 - dh - w01

        off00 = (hf * NUM_GRID + wf) * HIDDEN_DIM
        off01 = (hf * NUM_GRID + wc) * HIDDEN_DIM
        off10 = (hc * NUM_GRID + wf) * HIDDEN_DIM
        off11 = (hc * NUM_GRID + wc) * HIDDEN_DIM
        out_off = pid * HIDDEN_DIM

        # Cast weights to output dtype so the multiply-accumulate stays
        # in the same precision as the native PyTorch implementation.
        out_dtype = output_ptr.dtype.element_ty
        w00_c = w00.to(out_dtype)
        w01_c = w01.to(out_dtype)
        w10_c = w10.to(out_dtype)
        w11_c = w11.to(out_dtype)

        for d in tl.range(0, HIDDEN_DIM, BLOCK_D):
            cols = d + tl.arange(0, BLOCK_D)
            mask = cols < HIDDEN_DIM

            e00 = tl.load(embed_ptr + off00 + cols, mask=mask)
            e01 = tl.load(embed_ptr + off01 + cols, mask=mask)
            e10 = tl.load(embed_ptr + off10 + cols, mask=mask)
            e11 = tl.load(embed_ptr + off11 + cols, mask=mask)

            val = w00_c * e00 + w01_c * e01 + w10_c * e10 + w11_c * e11

            tl.store(output_ptr + out_off + cols, val, mask=mask)

    def triton_pos_embed_interpolate(
        embed_weight: torch.Tensor,
        t: int,
        h: int,
        w: int,
        num_grid_per_side: int,
        m_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Launch the fused Triton kernel for one (t,h,w) grid.

        Returns a tensor of shape ``(t * h * w, hidden_dim)`` with the
        bilinearly-interpolated position embeddings in spatial-merge order.
        """
        assert h % m_size == 0 and w % m_size == 0, (
            f"h={h} and w={w} must be divisible by m_size={m_size}"
        )
        hidden_dim = embed_weight.shape[1]
        total_out = t * h * w
        output = torch.empty(
            total_out,
            hidden_dim,
            device=embed_weight.device,
            dtype=dtype,
        )

        h_scale = float(num_grid_per_side - 1) / float(h - 1) if h > 1 else 0.0
        w_scale = float(num_grid_per_side - 1) / float(w - 1) if w > 1 else 0.0

        BLOCK_D = triton.next_power_of_2(hidden_dim)

        _bilinear_pos_embed_kernel[(total_out,)](
            embed_weight,
            output,
            h,
            w,
            h_scale,
            w_scale,
            num_grid_per_side,
            m_size,
            hidden_dim,
            BLOCK_D,
        )
        return output


def pos_embed_interpolate_native(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Eager PyTorch bilinear position-embedding interpolation.

    Returns a tensor of shape ``(t * h * w, hidden_dim)`` with the
    bilinearly-interpolated position embeddings in spatial-merge order.
    """
    assert h % m_size == 0 and w % m_size == 0, (
        f"h={h} and w={w} must be divisible by m_size={m_size}"
    )
    hidden_dim = embed_weight.shape[1]
    device = embed_weight.device

    h_idxs = torch.linspace(
        0,
        num_grid_per_side - 1,
        h,
        dtype=torch.float32,
        device=device,
    )
    w_idxs = torch.linspace(
        0,
        num_grid_per_side - 1,
        w,
        dtype=torch.float32,
        device=device,
    )

    h_floor = h_idxs.to(torch.long)
    w_floor = w_idxs.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

    dh = h_idxs - h_floor
    dw = w_idxs - w_floor

    dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
    h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
    h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1 - dh_grid - w01

    h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
    w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
    h_grid_idx = h_grid * num_grid_per_side

    indices = (h_grid_idx + w_grid).reshape(4, -1)
    weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
    weights = weights.to(dtype=dtype)

    embeds = embed_weight[indices]
    embeds *= weights
    combined = embeds.sum(dim=0)

    combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)
    combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
    repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
    return repeated.to(dtype=dtype)


class Cohere_VLImagePixelInputs(TensorSchema):
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


class Cohere_VLImageEmbeddingInputs(TensorSchema):
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


Cohere_VLImageInputs: TypeAlias = (
    Cohere_VLImagePixelInputs | Cohere_VLImageEmbeddingInputs
)


class Cohere_VisionAttention(nn.Module):
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

        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            scale=self.hidden_size_per_attention_head**-0.5,
            prefix=f"{prefix}.attn",
        )

        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
        # Only used for FlashInfer CuDNN backend.
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)
        seq_len, batch_size, _ = x.shape

        qkv = einops.rearrange(
            x,
            "s b (three head head_dim) -> b s three head head_dim",
            three=3,
            head=self.num_attention_heads_per_partition,
        )

        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            qk, v = qkv[:, :, :2], qkv[:, :, 2]

            qk_reshaped = einops.rearrange(
                qk, "b s two head head_dim -> (two b) s head head_dim", two=2
            )
            qk_reshaped = qk_reshaped.contiguous()
            qk_rotated = self.apply_rotary_emb(
                qk_reshaped,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
            )
            qk_rotated = qk_rotated.view(
                2,
                batch_size,
                seq_len,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            q, k = qk_rotated.unbind(dim=0)
        else:
            q, k, v = qkv.unbind(dim=2)

        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

        context_layer = einops.rearrange(
            context_layer, "b s h d -> s b (h d)", b=batch_size
        ).contiguous()

        output, _ = self.proj(context_layer)
        return output


class Cohere_VisionPatchEmbed(nn.Module):
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


class Cohere_VisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.linear_fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc1",
            disable_tp=use_data_parallel,
        )
        self.linear_fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc2",
            disable_tp=use_data_parallel,
        )
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        mlp_output = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return mlp_output


class Cohere_VisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Cohere_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = Cohere_VisionMLP(
            dim,
            mlp_hidden_dim,
            act_fn=act_fn,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
        sequence_lengths: torch.Tensor,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

        x = x + self.mlp(self.norm2(x))
        return x


class Cohere_VisionPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.hidden_size = context_dim * (spatial_merge_size**2)

        self.use_postshuffle_norm = use_postshuffle_norm
        if self.use_postshuffle_norm:
            context_dim = self.hidden_size

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(context_dim)
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc1",
            disable_tp=use_data_parallel,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(
            self.hidden_size,
            d_model,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc2",
            disable_tp=use_data_parallel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)

        x_parallel, _ = self.linear_fc1(x)
        x_parallel = self.act_fn(x_parallel)
        out, _ = self.linear_fc2(x_parallel)
        return out


class Cohere_VisionTransformer(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q.": (".qkv.", "q"),
            ".k.": (".qkv.", "k"),
            ".v.": (".qkv.", "v"),
        }
    )

    def __init__(
        self,
        vision_config: CohereCompassVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.deepstack_visual_indexes = (
            vision_config.deepstack_visual_indexes
            if hasattr(vision_config, "deepstack_visual_indexes")
            else []
        )
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)

        use_data_parallel = is_vit_use_data_parallel()
        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )

        # NOTE: This is used for creating empty tensor for all_gather for
        # DP ViT. Here out_hidden_size is enlarged due to deepstack
        self.out_hidden_size = vision_config.out_hidden_size * (
            1 + len(self.deepstack_visual_indexes)
        )

        self.patch_embed = Cohere_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )

        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads

        # FP8 attention: Q/K/V become independent contiguous tensors
        # after quantization, so cu_seqlens uses uniform stride (no 3x V).
        self.fp8_padded_hidden_size = get_fp8_padded_hidden_size(
            self.num_heads, head_dim
        )

        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
        )

        self.merger = Cohere_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
                Cohere_VisionPatchMerger(
                    d_model=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.deepstack_merger_list.{layer_idx}",
                )
                for layer_idx in range(len(self.deepstack_visual_indexes))
            ]
        )

        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
        )

        self.blocks = nn.ModuleList(
            [
                Cohere_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(vision_config.depth)
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size
        hpos_ids = hpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        wpos_ids = wpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()

        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))

    def rot_pos_emb(self, grid_thw: list[list[int]]):
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)
        pos_ids = [
            self.rot_pos_ids(h, w, self.spatial_merge_size)
            if t == 1
            else self.rot_pos_ids(h, w, self.spatial_merge_size).repeat(t, 1)
            for t, h, w in grid_thw
        ]
        pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)

        # Use pre-computed cos_sin_cache from RotaryEmbedding
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)

        return cos_combined, sin_combined

    def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
        interpolate_fn = (
            triton_pos_embed_interpolate if HAS_TRITON else pos_embed_interpolate_native
        )
        outputs = []
        for t, h, w in grid_thw:
            outputs.append(
                interpolate_fn(
                    self.pos_embed.weight,
                    t,
                    h,
                    w,
                    self.num_grid_per_side,
                    self.spatial_merge_size,
                    self.dtype,
                )
            )
        return torch.cat(outputs, dim=0)

    def prepare_encoder_metadata(
        self,
        grid_thw_list: list[list[int]],
        *,
        max_batch_size: int | None = None,
        max_frames_per_batch: int | None = None,
        max_seqlen_override: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute encoder metadata from grid_thw_list.

        Shared by the eager forward path, CUDA graph capture, and
        CUDA graph replay to avoid duplicated implementation.

        Args:
            grid_thw_list: Grid configurations as list of [t, h, w].
            max_batch_size: If set, pad cu_seqlens to this size
                (needed for CUDA graph capture/replay).
            max_frames_per_batch: If set, overrides max_batch_size for
                cu_seqlens padding. For video inputs each item contributes
                T attention sequences (frames); this sizes the buffer to
                the total frame budget so video replays never overflow.
            max_seqlen_override: If set, use this value for max_seqlen
                instead of computing from cu_seqlens (needed for CUDA
                graph capture to cover worst-case replay scenarios).
            device: Device to place tensors on. Defaults to self.device.
        """
        if device is None:
            device = self.device

        metadata: dict[str, torch.Tensor | None] = {}

        # Positional embeddings
        metadata["pos_embeds"] = self.fast_pos_embed_interpolate(grid_thw_list)
        rotary_cos, rotary_sin = self.rot_pos_emb(grid_thw_list)
        metadata["rotary_pos_emb_cos"] = rotary_cos
        metadata["rotary_pos_emb_sin"] = rotary_sin

        # cu_seqlens from grid_thw
        grid_thw_np = np.array(grid_thw_list, dtype=np.int32)
        patches_per_frame = grid_thw_np[:, 1] * grid_thw_np[:, 2]
        cu_seqlens = np.repeat(patches_per_frame, grid_thw_np[:, 0]).cumsum(
            dtype=np.int32
        )
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])

        # Pad cu_seqlens to the required number of sequences.
        # For videos each item contributes T frames = T attention sequences,
        # so the total can exceed max_batch_size. max_frames_per_batch
        # overrides the pad target when set.
        pad_to = (
            max_frames_per_batch if max_frames_per_batch is not None else max_batch_size
        )
        if pad_to is not None:
            num_seqs = len(cu_seqlens) - 1
            if num_seqs < pad_to:
                cu_seqlens = np.concatenate(
                    [
                        cu_seqlens,
                        np.full(
                            pad_to - num_seqs,
                            cu_seqlens[-1],
                            dtype=np.int32,
                        ),
                    ]
                )

        # sequence_lengths (backend-specific)
        metadata["sequence_lengths"] = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu_seqlens, device
        )

        # max_seqlen
        if max_seqlen_override is not None:
            max_seqlen_val = max_seqlen_override
        else:
            max_seqlen_val = MMEncoderAttention.compute_max_seqlen(
                self.attn_backend, cu_seqlens
            )
        # Keep max_seqlen on CPU: attention wrappers call .item() on it,
        # and having it on GPU would capture a wasteful D2H copy in CUDA
        # graphs without changing behavior (the scalar is baked at capture).
        metadata["max_seqlen"] = torch.tensor(max_seqlen_val, dtype=torch.int32)

        # Recompute cu_seqlens (backend-specific transformation)
        metadata["cu_seqlens"] = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu_seqlens,
            self.hidden_size,
            self.tp_size,
            device,
            fp8_padded_hidden_size=self.fp8_padded_hidden_size,
        )

        return metadata

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
        *,
        encoder_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        hidden_states = self.patch_embed(hidden_states)

        if encoder_metadata is None:
            if isinstance(grid_thw, list):
                grid_thw_list = grid_thw
            else:
                grid_thw_list = grid_thw.tolist()
            encoder_metadata = self.prepare_encoder_metadata(grid_thw_list)

        pos_embeds = encoder_metadata["pos_embeds"]
        hidden_states = hidden_states + pos_embeds
        hidden_states = hidden_states.unsqueeze(1)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=encoder_metadata["cu_seqlens"],
                rotary_pos_emb_cos=encoder_metadata["rotary_pos_emb_cos"],
                rotary_pos_emb_sin=encoder_metadata["rotary_pos_emb_sin"],
                max_seqlen=encoder_metadata["max_seqlen"],
                sequence_lengths=encoder_metadata.get("sequence_lengths"),
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        hidden_states = torch.cat(
            [hidden_states] + deepstack_feature_lists, dim=1
        )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


def _create_cohere_compass_field_factory(
    spatial_merge_size: int,
) -> Callable[
    [Mapping[str, torch.Tensor]],
    Mapping[str, MultiModalFieldConfig],
]:
    def _field_config(
        hf_inputs: Mapping[str, torch.Tensor],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_pixel_grid_sizes = image_grid_thw.prod(-1)
        image_embed_grid_sizes = (
            image_pixel_grid_sizes // spatial_merge_size // spatial_merge_size
        )

        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes(
                "image", image_pixel_grid_sizes
            ),
            "image_embeds": MultiModalFieldConfig.flat_from_sizes(
                "image", image_embed_grid_sizes
            ),
            "image_grid_thw": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        }

    return _field_config


class CohereCompassMultiModalDataParser(MultiModalDataParser):
    def __init__(self, spatial_merge_size: int, *args, **kwargs):
        self._spatial_merge_size = spatial_merge_size
        super().__init__(*args, **kwargs)

    def _parse_image_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[ImageItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields={"image_embeds", "image_grid_thw"},
                fields_factory=_create_cohere_compass_field_factory(
                    self._spatial_merge_size
                ),
            )

        return super()._parse_image_data(data)


class CohereCompassProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        max_image_tokens = self.get_max_image_tokens()
        return {"image": max_image_tokens}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: CohereCompassImageProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> int:
        _, num_image_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=1,
            image_processor=image_processor,
            mm_kwargs=mm_kwargs,
        )
        return num_image_tokens

    def get_image_size_with_most_features(
        self, max_pixels: int | None = None
    ) -> ImageSize:
        # NOTE: Simply processing a huge size with _get_vision_info might not give a
        # size that maximizes the number of features, i.e., the number of (merged)
        # patches. This is because the number of patches limits the allowed aspect
        # ratios. For example, suppose the maximum number of patches is 1280. A square
        # image cannot be broken down into 1280 patches, so feeding a giant square image
        # into _get_vision_info will not yield a size that maximizes the number of
        # patches. Therefore, we directly factorize the maximum number of patches into
        # height and width. The tricky part is to avoid extreme aspect ratios (>200).
        # If we can't find a suitable aspect ratio, we decrease the number of
        # patches and retry. This is safe because the processor does not accept extreme
        # aspect ratios, so there is no valid post-resize image with the number of
        # patches that yields extreme aspect ratios.

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size

        if max_pixels is None:
            image_processor = self.get_image_processor()

            mm_kwargs = self.ctx.get_merged_mm_kwargs({})
            size = image_processor.size
            if override_size := mm_kwargs.get("size"):
                size = size | override_size
            if (override_min_pixels := mm_kwargs.get("min_pixels")) is not None:
                size = size | {"shortest_edge": override_min_pixels}
            if (override_max_pixels := mm_kwargs.get("max_pixels")) is not None:
                size = size | {"longest_edge": override_max_pixels}

            max_pixels = size["longest_edge"]

        unit = patch_size * merge_size
        max_seq_len = max_pixels // (unit * unit)

        def closest_factor_pair(n: int) -> tuple[int, int]:
            # left <= right
            for d in range(math.isqrt(n), 0, -1):
                if n % d == 0:
                    return d, n // d
            return 1, n

        height_factor, width_factor = 1, max_seq_len
        for seq_len in range(max_seq_len, 0, -1):
            height_factor, width_factor = closest_factor_pair(seq_len)
            if width_factor / height_factor <= 200:
                break

        return ImageSize(width=unit * width_factor, height=unit * height_factor)

    def get_max_image_tokens(self) -> int:
        image_processor = self.get_image_processor()
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            image_processor=image_processor,
            mm_kwargs={},
        )

    def get_hf_config(self) -> CohereCompassConfig:
        return self.ctx.get_hf_config(CohereCompassConfig)

    def get_hf_processor(self, **kwargs: object) -> CohereCompassProcessor:
        return self.ctx.get_hf_processor(
            CohereCompassProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_image_processor(self, **kwargs: object) -> CohereCompassImageProcessor:
        return self.get_hf_processor(**kwargs).image_processor

    def get_data_parser(self):
        return CohereCompassMultiModalDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 2,
        do_resize: bool = True,
        image_processor: CohereCompassImageProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> tuple[ImageSize, int]:
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        mm_kwargs = self.ctx.get_merged_mm_kwargs(mm_kwargs)
        size = image_processor.size
        if override_size := mm_kwargs.get("size"):
            size = size | override_size
        if (override_min_pixels := mm_kwargs.get("min_pixels")) is not None:
            size = size | {"shortest_edge": override_min_pixels}
        if (override_max_pixels := mm_kwargs.get("max_pixels")) is not None:
            size = size | {"longest_edge": override_max_pixels}

        if do_resize:
            smart_resize = image_smart_resize
            extra_kwargs: dict[str, Any] = {}

            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=size["shortest_edge"],
                max_pixels=size["longest_edge"],
                **extra_kwargs,
            )
            preprocessed_size = ImageSize(width=resized_width, height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width, height=image_height)

        padded_num_frames = round_up(num_frames, temporal_patch_size)

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens


class CohereCompassDummyInputsBuilder(
    BaseDummyInputsBuilder[CohereCompassProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return "<|VISION_START|><|IMAGE_PAD|><|VISION_END|>" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        image_overrides = mm_options.get("image")

        target_image_width, target_image_height = (
            self.info.get_image_size_with_most_features()
        )

        return {
            "image": self._get_dummy_images(
                width=target_image_width,
                height=target_image_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
        }


class CohereCompassMultiModalProcessor(
    BaseMultiModalProcessor[CohereCompassProcessingInfo]
):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _create_cohere_compass_field_factory(
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

        def get_image_replacement_cohere_compass(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [hf_processor.image_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[hf_processor.image_token_id],
                replacement=get_image_replacement_cohere_compass,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    CohereCompassMultiModalProcessor,
    info=CohereCompassProcessingInfo,
    dummy_inputs=CohereCompassDummyInputsBuilder,
)
class CohereCompassForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsEncoderCudaGraph,
    SupportsLoRA,
    SupportsPP,
    SupportsMRoPE,
    SupportsEagle,
    SupportsEagle3,
    SupportsMultiModalPruning,
):
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
        "qkv": ["qkv"],  # For vision tower's already-packed QKV
    }

    supports_encoder_tp_data = True

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|VISION_START|><|IMAGE_PAD|><|VISION_END|>"
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        self.multimodal_config = multimodal_config
        assert multimodal_config is not None
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        self.use_deepstack = bool(config.vision_config.deepstack_visual_indexes)
        self.deepstack_num_level = len(config.vision_config.deepstack_visual_indexes)
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image"}):
            self.visual = Cohere_VisionTransformer(
                config.vision_config,
                norm_eps=1e-6,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )
            if self.use_deepstack:
                self.deepstack_input_embeds = [
                    torch.zeros(
                        vllm_config.scheduler_config.max_num_batched_tokens,
                        config.text_config.hidden_size,
                    )
                    for _ in range(self.deepstack_num_level)
                ]
                self.deepstack_input_embeds_num_tokens = 0

        with self._mark_language_model(vllm_config):
            self.language_model = CohereCompassForCausalLM(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        if not get_pp_group().is_first_rank and self.use_deepstack:
            assert self.language_model.model.start_layer >= self.deepstack_num_level

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def iter_mm_grid_thw(
        self, mm_features: list[MultiModalFeatureSpec]
    ) -> Iterator[tuple[int, int, int, int, float]]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            if mm_feature.modality != "image":
                raise ValueError(f"Unsupported modality: {mm_feature.modality}")

            offset = mm_feature.mm_position.offset
            feature_data = mm_feature.data
            assert feature_data is not None
            grid_item = feature_data.get("image_grid_thw")
            assert grid_item is not None
            grid_data = grid_item.data
            assert isinstance(grid_data, torch.Tensor)
            t, h, w = grid_data.tolist()
            assert t == 1, f"Image must have 1 frame, got {t}"
            yield offset, 1, h // spatial_merge_size, w // spatial_merge_size, 1.0

    def _get_deepstack_input_embeds(
        self,
        num_tokens: int,
    ) -> IntermediateTensors | None:
        if not getattr(self, "deepstack_input_embeds", None):
            return None  # If vision tower is skipped
        if num_tokens > self.deepstack_input_embeds[0].size(0):
            self._resize_deepstack_input_embeds(num_tokens)

        # get deepstack_input_embeds from buffer, and clear the buffer
        return IntermediateTensors(
            {
                f"deepstack_input_embeds_{idx}": self.deepstack_input_embeds[idx][
                    :num_tokens
                ]
                for idx in range(self.deepstack_num_level)
            }
        )

    def _resize_deepstack_input_embeds(self, num_tokens: int) -> None:
        self.deepstack_input_embeds = [
            torch.zeros(
                num_tokens,
                self.config.text_config.hidden_size,
                device=self.deepstack_input_embeds[0].device,
                dtype=self.deepstack_input_embeds[0].dtype,
            )
            for _ in range(self.deepstack_num_level)
        ]

    def _set_deepstack_input_embeds(self, deepstack_input_embeds: torch.Tensor) -> None:
        if not getattr(self, "deepstack_input_embeds", None):
            return

        # set deepstack_input_embeds to buffer
        num_tokens = deepstack_input_embeds.size(1)
        if num_tokens > self.deepstack_input_embeds[0].size(0):
            self._resize_deepstack_input_embeds(num_tokens)
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][:num_tokens].copy_(
                deepstack_input_embeds[idx]
            )
        self.deepstack_input_embeds_num_tokens = num_tokens

    def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
        if not getattr(self, "deepstack_input_embeds", None):
            return
        if getattr(self, "deepstack_input_embeds_num_tokens", 0) == 0:
            return

        # clear deepstack_input_embeds in buffer
        if num_tokens > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_tokens].zero_()
            self.deepstack_input_embeds_num_tokens = 0

    # -- SupportsEncoderCudaGraph protocol methods --

    def get_encoder_cudagraph_config(self):
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphConfig,
        )

        # When EVS pruning is enabled, embed_multimodal post-processes both
        # image and video embeddings (mrope positions are appended for image,
        # prune+append for video). The encoder CUDA graph path bypasses that
        # post-process, producing inconsistent embedding formats vs eager. So
        # disable CUDA graph for all modalities when pruning is on.
        modalities = [] if self.is_multimodal_pruning_enabled else ["image"]

        # Compute max_frames_per_video for budget sizing.
        max_frames = 1

        return EncoderCudaGraphConfig(
            modalities=modalities,
            buffer_keys=[
                "pixel_values",
                "pos_embeds",
                "rotary_pos_emb_cos",
                "rotary_pos_emb_sin",
                "cu_seqlens",
                "max_seqlen",
                "sequence_lengths",
            ],
            out_hidden_size=self.visual.out_hidden_size,
            max_frames_per_video=max_frames,
        )

    def get_input_modality(
        self,
        mm_kwargs: dict[str, Any],
    ) -> str:
        if "image_grid_thw" in mm_kwargs:
            return "image"
        raise AssertionError("This line should be unreachable.")

    def get_encoder_cudagraph_budget_range(
        self,
        vllm_config,
    ) -> tuple[int, int]:
        # Min: estimated smallest possible encoder input.
        # 224x224 image → 16x16 patches (patch_size=14)
        #                 spatial_merge_size=2 → 8x8 = 64 tokens
        min_budget = 64
        # Max: capped by max_num_batched_tokens
        # TODO(shen-shanshan): the max_budget auto-infer needs to be optimized later.
        max_budget = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.model_config.max_model_len,
        )
        return (min_budget, max_budget)

    def _get_pixel_values_by_modality(
        self,
        mm_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        modality = self.get_input_modality(mm_kwargs)
        if modality == "image":
            return mm_kwargs["pixel_values"]
        raise AssertionError("This line should be unreachable.")

    def _get_grid_thw_by_modality(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[list[int]]:
        grid_thw_key = f"{self.get_input_modality(mm_kwargs)}_grid_thw"
        grid_thw = mm_kwargs[grid_thw_key]
        if not isinstance(grid_thw, list):
            grid_thw = grid_thw.tolist()
        return grid_thw

    def get_encoder_cudagraph_item_specs(
        self,
        mm_kwargs: dict[str, Any],
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec

        m = self.visual.spatial_merge_size
        grid_thw = self._get_grid_thw_by_modality(mm_kwargs)
        return [
            EncoderItemSpec(
                input_size=t * h * w,
                output_tokens=t * (h // m) * (w // m),
            )
            for t, h, w in grid_thw
        ]

    def select_encoder_cudagraph_items(
        self,
        mm_kwargs: dict[str, Any],
        indices: list[int],
    ) -> dict[str, Any]:
        grid_thw = self._get_grid_thw_by_modality(mm_kwargs)
        pixel_values = self._get_pixel_values_by_modality(mm_kwargs)

        if len(indices) == 0:
            return {
                "pixel_values": pixel_values[:0],
                "image_grid_thw": [],
            }

        # Compute cumulative patch offsets for slicing pixel_values
        patches_per_item = [t * h * w for t, h, w in grid_thw]
        cum_patches = [0]
        for p in patches_per_item:
            cum_patches.append(cum_patches[-1] + p)

        selected_pv = torch.cat(
            [pixel_values[cum_patches[i] : cum_patches[i + 1]] for i in indices]
        )
        selected_grid = [grid_thw[i] for i in indices]

        return {
            "pixel_values": selected_pv,
            "image_grid_thw": selected_grid,
        }

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphCaptureInputs,
        )

        spatial_merge_size = self.visual.spatial_merge_size
        # Ceil so the buffer fits the worst case of one item using the full
        # budget. Floor under-allocates when budget is not a multiple of
        # max_batch_size.
        per_mm_item_output = (token_budget + max_batch_size - 1) // max_batch_size

        # Image-format grid_config (T=1).
        grid_config = [
            [1, spatial_merge_size, per_mm_item_output * spatial_merge_size]
            for _ in range(max_batch_size)
        ]

        # Create dummy pixel_values
        patch_embed = self.visual.patch_embed
        in_channels = patch_embed.proj.in_channels
        patch_size = patch_embed.patch_size
        temporal_patch_size = patch_embed.temporal_patch_size
        total_patches = sum(t * h * w for t, h, w in grid_config)
        flattened_patch_size = (
            in_channels * temporal_patch_size * patch_size * patch_size
        )
        dummy_pixel_values = torch.randn(
            total_patches, flattened_patch_size, device=device, dtype=dtype
        )

        # Override max_seqlen with a safe upper bound for capture.
        # max_seqlen.item() gets baked into the CUDA graph (not replayed),
        # so the capture value must cover any replay scenario.
        # Worst case: 1 item consuming the full budget ->
        # seq_len = token_budget * spatial_merge_size^2.
        metadata = self.visual.prepare_encoder_metadata(
            grid_config,
            max_batch_size=max_batch_size,
            max_frames_per_batch=max_frames_per_batch,
            max_seqlen_override=token_budget * (spatial_merge_size**2),
            device=device,
        )

        # Just use image-modality dummy input_buffer for capturing
        values = metadata | {
            "pixel_values": dummy_pixel_values,
        }

        return EncoderCudaGraphCaptureInputs(
            values=values,
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        modality = self.get_input_modality(mm_kwargs)
        grid_thw_list = self._get_grid_thw_by_modality(mm_kwargs)

        if modality == "image":
            metadata = self.visual.prepare_encoder_metadata(
                grid_thw_list,
                max_batch_size=max_batch_size,
            )
        else:
            raise AssertionError("This line should be unreachable.")

        values = metadata | {
            "pixel_values": self._get_pixel_values_by_modality(mm_kwargs),
        }
        return EncoderCudaGraphReplayBuffers(values=values)

    def encoder_cudagraph_forward(
        self,
        values: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        pixel_values = values.pop("pixel_values")
        metadata = values
        return self.visual(pixel_values, None, encoder_metadata=metadata)

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        pixel_values = self._get_pixel_values_by_modality(mm_kwargs)
        grid_thw = self._get_grid_thw_by_modality(mm_kwargs)
        return self.visual(pixel_values, grid_thw)

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Cohere_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return Cohere_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return Cohere_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )
        raise AssertionError("image input must contain pixels or embeddings")

    def _process_image_input(
        self, image_input: Cohere_VLImageInputs
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

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _postprocess_image_embeds_evs(
        self,
        image_embeds_split: tuple[torch.Tensor, ...],
        image_input: Cohere_VLImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        """
        Append mrope positions for each for images.
        This is necessary to recover correct mrope positions

        Args:
            image_embeds_split: Tuple of image embeddings for
                each image item.
            image_input: Image input data.

        Returns:
            Tuple of image embeddings for each image item.
            Resulting embeddings will have extra 5 channels for
            computed mrope positions, consistent with video embeddings.
        """
        if self.is_multimodal_pruning_enabled:
            merge_size = self.visual.spatial_merge_size
            grid_thw = image_input["image_grid_thw"]
            grid_thw_list = grid_thw.tolist()
            image_embeds_out = []
            for emb, size in zip(image_embeds_split, grid_thw_list):
                positions = compute_mrope_for_media(size, merge_size).to(
                    emb.device, non_blocking=True
                )
                positions = torch.cat(
                    [
                        positions,
                        torch.zeros_like(
                            positions[:, 0:1]
                        ),  # Dummy extra fifth channel
                    ],
                    dim=1,
                )
                emb = torch.cat([emb, positions], dim=1)
                image_embeds_out.append(emb)
            image_embeds_split = tuple(image_embeds_out)
        return image_embeds_split

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
        return mm_input_by_modality

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        llm_pos_ids_list: list[np.ndarray] = []
        st = 0

        for (
            offset,
            llm_grid_t,
            llm_grid_h,
            llm_grid_w,
            _,
        ) in self.iter_mm_grid_thw(mm_features):
            text_len = offset - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )
            grid_indices = np.indices((llm_grid_t, llm_grid_h, llm_grid_w)).reshape(
                3, -1
            )
            llm_pos_ids_list.append(grid_indices + text_len + st_idx)
            st = offset + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        return torch.from_numpy(llm_positions), mrope_position_delta

    def recompute_mrope_positions(
        self,
        input_ids: list[int],
        multimodal_embeddings: MultiModalEmbeddings,
        mrope_positions: torch.LongTensor,
        num_computed_tokens: int,
    ) -> tuple[MultiModalEmbeddings, torch.Tensor, int]:
        """
        Update part of input mrope positions (starting with
        num_computed_tokens index). Original mrope_positions are computed
        for unpruned sequence and becomes incorrect once pruning occurs,
        so once we prune media tokens we should reflect this in the
        mrope_positions before we feed it to LLM.

        Args:
            input_ids: (N,) All input tokens of the prompt containing
                entire sequence.
            multimodal_embeddings: Tuple of multimodal embeddings that
                fits into the prefill chunk that is being processed.
            mrope_positions: Existing mrope positions (3, N) for entire
                sequence
            num_computed_tokens: A number of computed tokens so far.

        Returns:
            Tuple of (multimodal_embeddings, mrope_positions,
                mrope_position_delta).
        """
        return self._recompute_mrope_positions(
            input_ids=input_ids,
            multimodal_embeddings=multimodal_embeddings,
            mrope_positions=mrope_positions,
            num_computed_tokens=num_computed_tokens,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
        )

    @staticmethod
    def _recompute_mrope_positions(
        input_ids: list[int],
        multimodal_embeddings: MultiModalEmbeddings,
        mrope_positions: torch.LongTensor,
        num_computed_tokens: int,
        vision_start_token_id: int,
        image_token_id: int,
        video_token_id: int,
    ) -> tuple[MultiModalEmbeddings, torch.Tensor, int]:
        # Device
        device = (
            multimodal_embeddings[0].device
            if len(multimodal_embeddings)
            else mrope_positions.device
        )

        # Tensors
        input_ids_t = async_tensor_h2d(input_ids, device=device, dtype=torch.long)

        mm_embeddings_out = []
        mm_embeddings_pos = []
        # Strip position information from embeddings (last 5 channels)
        # For Cohere VL, handle potentially empty frames (from unpacking)
        for mm in multimodal_embeddings:
            if mm.shape[0] > 0:  # Only process non-empty frames
                mm_embeddings_out.append(mm[:, :-5])
                mm_embeddings_pos.append(mm[:, -5:].permute(1, 0).long())
            else:
                # Empty frame - keep as is
                mm_embeddings_out.append(mm)
                # Create empty position tensor with correct shape
                mm_embeddings_pos.append(
                    torch.empty(5, 0, device=device, dtype=torch.long)
                )

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

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image).
        multimodal_embeddings: list[torch.Tensor] = []

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                image_embeddings = self._postprocess_image_embeds_evs(
                    image_embeddings, multimodal_input
                )
                multimodal_embeddings.extend(image_embeddings)

        embeddings_tuple = tuple(multimodal_embeddings)
        return embeddings_tuple

    def _compute_deepstack_embeds(
        self,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
        is_multimodal: torch.Tensor,
    ) -> tuple[torch.Tensor, MultiModalEmbeddings]:
        visual_lens = [len(x) for x in multimodal_embeddings]
        multimodal_embeddings_cat = torch.cat(multimodal_embeddings, dim=0)

        (
            multimodal_embeddings_main,
            multimodal_embeddings_multiscale,
        ) = torch.split(
            multimodal_embeddings_cat,
            [self.visual_dim, self.multiscale_dim],
            dim=-1,
        )

        multimodal_embeddings = torch.split(
            multimodal_embeddings_main, visual_lens, dim=0
        )
        multimodal_embeddings_multiscale = torch.split(
            multimodal_embeddings_multiscale, visual_lens, dim=0
        )

        deepstack_input_embeds = inputs_embeds.new_zeros(
            inputs_embeds.size(0), self.deepstack_num_level * inputs_embeds.size(1)
        )

        deepstack_input_embeds = _merge_multimodal_embeddings(
            inputs_embeds=deepstack_input_embeds,
            multimodal_embeddings=multimodal_embeddings_multiscale,
            is_multimodal=is_multimodal,
        )
        deepstack_input_embeds = deepstack_input_embeds.view(
            inputs_embeds.shape[0], self.deepstack_num_level, self.visual_dim
        )
        deepstack_input_embeds = deepstack_input_embeds.permute(1, 0, 2)

        return deepstack_input_embeds, multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        if self.use_deepstack:
            (
                deepstack_input_embeds,
                multimodal_embeddings,
            ) = self._compute_deepstack_embeds(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
        else:
            deepstack_input_embeds = None

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        if deepstack_input_embeds is not None:
            self._set_deepstack_input_embeds(deepstack_input_embeds)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Cohere Compass.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Cohere Compass
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from previous pipeline
                stages.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments including:
                - pixel_values: Pixel values to be fed to a model.
                    `None` if no images are passed.
                - image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    LLM. `None` if no images are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        if inputs_embeds is not None and get_pp_group().is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )
        else:
            deepstack_input_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            # args for deepstack
            deepstack_input_embeds=deepstack_input_embeds,
        )

        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))

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
            language_model="language_model",
            connector=["visual.merger", "visual.deepstack_merger_list"],
            tower_model="visual.",
        )

    def get_num_mm_encoder_tokens(
        self,
        num_image_tokens: int,
    ) -> int:
        hf_config = self.config
        vision_config = hf_config.vision_config
        merge_size = vision_config.spatial_merge_size

        return num_image_tokens * merge_size**2

    def get_num_mm_connector_tokens(
        self,
        num_vision_tokens: int,
    ) -> int:
        hf_config = self.config
        vision_config = hf_config.vision_config
        merge_size = vision_config.spatial_merge_size
        return num_vision_tokens // merge_size**2


@lru_cache
def _cached_tensor(x, device) -> torch.Tensor:
    return torch.tensor(x, device=device)


class CohereCompassAttention(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        max_position_embeddings: int | None = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]

        self.head_dim = config.head_dim
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        rope_parameters = config.rope_parameters.get(layer_type)
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                max_position=max_position_embeddings or config.max_position_embeddings,
                rope_parameters=rope_parameters,
                is_neox_style=True,
            )
            if rope_parameters is not None
            else None
        )
        sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class CohereCompassDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        max_position_embeddings: int | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = CohereCompassAttention(
            config,
            cache_config,
            quant_config=quant_config,
            max_position_embeddings=max_position_embeddings,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = CohereMLP(
            config,
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = LayerNorm(
            param_shape=config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states_attention = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states_mlp = self.mlp(hidden_states)
        hidden_states = residual + hidden_states_attention + hidden_states_mlp
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "deepstack_input_embeds": 0,
    }
)
class CohereCompassTextModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: CohereCompassDecoderLayer(
                config,
                cache_config,
                quant_config=quant_config,
                max_position_embeddings=vllm_config.model_config.max_model_len,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = LayerNorm(
            param_shape=config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                hidden_states = (
                    hidden_states
                    + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class CohereCompassForCausalLM(CohereForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        assert config.tie_word_embeddings

        self.config = config
        self.quant_config = vllm_config.quant_config
        logit_scale = 1.0 if config.logit_scale is None else config.logit_scale
        self.logits_processor = LogitsProcessor(config.vocab_size, scale=logit_scale)
        self.model = CohereCompassTextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )


__all__ = ["CohereCompassForConditionalGeneration"]
