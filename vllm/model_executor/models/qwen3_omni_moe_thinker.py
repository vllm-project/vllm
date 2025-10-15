# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
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
"""Inference-only Qwen3-Omni-Moe model (thinker part)."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeThinkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.attention.backends.registry import _Backend
from vllm.attention.layer import check_upstream_fa_availability
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioFeatureInputs,
    Qwen2AudioProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)

# yapf conflicts with isort for this block
# yapf: disable
from .qwen2_5_omni_thinker import (
    Qwen2_5OmniConditionalGenerationMixin,
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo,
)

# yapf: enable
from .qwen2_5_vl import (
    Qwen2_5_VisionAttention,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLProcessingInfo,
)
from .qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from .vision import get_llm_pos_ids_for_vision, get_vit_attn_backend

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None

logger = init_logger(__name__)


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor):
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    )
    return feat_lengths, output_lengths


class Qwen3_VisionPatchEmbed(nn.Module):
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
        self.proj = nn.Conv3d(
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


class Qwen3_VisionMLP(nn.Module):
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
        self.linear_fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc1",
        )
        self.linear_fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc2",
        )
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        mlp_output = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return mlp_output


class Qwen3_VisionBlock(nn.Module):
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
        self.attn = Qwen2_5_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = Qwen3_VisionMLP(
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
        rotary_pos_emb: torch.Tensor,
        max_seqlen: int | None = None,  # Only used for Flash Attention
        seqlens: list[int] | None = None,  # Only used for xFormers
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


class Qwen3_VisionPatchMerger(nn.Module):
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
        self.hidden_size = context_dim * (spatial_merge_size**2)

        self.use_postshuffle_norm = use_postshuffle_norm
        if self.use_postshuffle_norm:
            context_dim = self.hidden_size

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.ln_q = norm_layer(
            self.hidden_size if use_postshuffle_norm else context_dim
        )
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp.0",
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    d_model,
                    bias=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp.2",
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.ln_q(x.view(-1, self.hidden_size))
        else:
            x = self.ln_q(x).view(-1, self.hidden_size)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class Qwen3Omni_VisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.image_size = vision_config.image_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.num_grid_per_side = self.image_size // self.patch_size
        self.apply_vit_abs_pos_embed = vision_config.apply_vit_abs_pos_embed
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes

        self.patch_embed = Qwen3_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )

        # vit pos embeding, TODO: spatial_patch_size vs patch_size
        if self.apply_vit_abs_pos_embed:
            self.pos_embed = nn.Embedding(self.num_grid_per_side**2, self.hidden_size)
        else:
            self.pos_embed = nn.Parameter(
                torch.empty([1, self.num_grid_per_side**2, self.hidden_size])
            )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen3_VisionBlock(
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
        self.merger = Qwen3_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )
        if self.deepstack_visual_indexes is not None:
            self.merger_list = nn.ModuleList(
                [
                    Qwen3_VisionPatchMerger(
                        d_model=vision_config.out_hidden_size,
                        context_dim=self.hidden_size,
                        spatial_merge_size=self.spatial_merge_size,
                        use_postshuffle_norm=True,
                        norm_layer=norm_layer,
                        quant_config=quant_config,
                        prefix=f"{prefix}.merger_list.{layer_idx}",
                    )
                    for layer_idx in range(len(self.deepstack_visual_indexes))
                ]
            )

        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim, dtype=torch.get_default_dtype()
        )
        if self.attn_backend != _Backend.FLASH_ATTN and check_upstream_fa_availability(
            torch.get_default_dtype()
        ):
            self.attn_backend = _Backend.FLASH_ATTN

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
        num_grid_per_side = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim

        outputs = []
        for t, h, w in grid_thw:
            h_idxs = torch.linspace(
                0, num_grid_per_side - 1, h, dtype=torch.float32, device=self.device
            )
            w_idxs = torch.linspace(
                0, num_grid_per_side - 1, w, dtype=torch.float32, device=self.device
            )

            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # Create meshgrid view for all h, w vars
            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")
            h_floor_grid_idx = h_floor_grid * num_grid_per_side
            h_ceil_grid_idx = h_ceil_grid * num_grid_per_side

            # original computation of weights
            # w00 = (1 - dh_grid) * (1 - dw_grid)
            # w01 = (1 - dh_grid) * dw_grid
            # w10 = dh_grid * (1 - dw_grid)
            # w11 = dh_grid * dw_grid
            # we reuse w11 here to avoid duplicate
            # dh_grid * dw_grid computation
            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - dw_grid + w11

            idx00 = h_floor_grid_idx + w_floor_grid
            idx01 = h_floor_grid_idx + w_ceil_grid
            idx10 = h_ceil_grid_idx + w_floor_grid
            idx11 = h_ceil_grid_idx + w_ceil_grid

            indices = torch.stack([idx00, idx01, idx10, idx11], dim=0).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
            weights = weights.to(dtype=self.dtype, device=self.device)

            embeds = self.pos_embed(indices)
            weighted_embeds = embeds * weights
            p0, p1, p2, p3 = weighted_embeds.unbind(dim=0)
            combined = p0 + p1 + p2 + p3

            combined = combined.view(h * w, hidden_dim)
            repeated = combined.unsqueeze(0).expand(t, -1, -1).contiguous()
            repeated = repeated.view(
                t, h // m_size, m_size, w // m_size, m_size, hidden_dim
            )
            repeated = repeated.permute(0, 1, 3, 2, 4, 5).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> tuple[int | None, list[int] | None]:
        max_seqlen, seqlens = None, None
        if self.attn_backend == _Backend.FLASH_ATTN:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        elif self.attn_backend == _Backend.XFORMERS:
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return max_seqlen, seqlens

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        if self.apply_vit_abs_pos_embed:
            pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
            hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        hidden_states = hidden_states.unsqueeze(1)
        rotary_pos_emb = rotary_pos_emb.to(hidden_states.device)
        max_seqlen, seqlens = self.compute_attn_mask_seqlen(cu_seqlens)

        hidden_states_list = []
        deepstack_visual_indexes = self.deepstack_visual_indexes

        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen,
                seqlens=seqlens,
            )
            if (
                deepstack_visual_indexes is not None
                and layer_num in deepstack_visual_indexes
            ):
                hidden_states_list.append(hidden_states)

        hidden_states = self.merger(hidden_states)

        # processing deepstack
        if deepstack_visual_indexes is not None:
            processed_hidden_states_list = [hidden_states]
            for idx, x in enumerate(hidden_states_list):
                x = self.merger_list[idx](x)
                processed_hidden_states_list.append(x)
            # we cat the original visual features and deepstack features
            # along the feature dim
            hidden_states = torch.cat(
                processed_hidden_states_list, dim=1
            )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
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


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "deepstack_input_embeds": 0,
    }
)
class Qwen3MoeLLMModel(Qwen3MoeModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        self.deepstack_multiscale_layer_start = 1

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer]
        ):
            layer_idx = layer_idx + self.start_layer

            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

            if deepstack_input_embeds is not None and layer_idx in range(
                0, len(deepstack_input_embeds)
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


class Qwen3MoeLLMForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3MoeForCausalLM, self).__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3MoeLLMModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=quant_config
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )


class Qwen3OmniMoeThinkerProcessingInfo(
    Qwen2AudioProcessingInfo, Qwen2_5_VLProcessingInfo
):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3OmniMoeConfig).thinker_config

    def get_hf_processor(self, **kwargs: object) -> Qwen3OmniMoeProcessor:
        processor = self.ctx.get_hf_processor(
            Qwen3OmniMoeProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )
        if not hasattr(processor, "audio_token"):
            processor.audio_token = "<|audio_pad|>"
        if not hasattr(processor, "image_token"):
            processor.image_token = "<|image_pad|>"
        if not hasattr(processor, "video_token"):
            processor.video_token = "<|video_pad|>"
        return processor

    def get_feature_extractor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None, "image": None, "video": None}


Qwen3OmniMoeThinkerDummyInputsBuilder = Qwen2_5OmniThinkerDummyInputsBuilder


class Qwen3OmniMoeThinkerMultiModalProcessor(
    Qwen2_5OmniThinkerMultiModalProcessor,
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np.ndarray:
            length = x.shape[-1]
            if length % hop_length != 0:
                pad_length = hop_length - (length % hop_length)
                x = np.pad(x, (0, pad_length), mode="constant", constant_values=0)
            return x

        # NOTE: WhisperFeatureExtractor cannot handle empty list of audios
        if audios:
            # NOTE: Qwen3-Omni processor accept "audio"
            # To make sure the cache works with padding=True, we pre-padded
            # the audio to multiple of hop_length.
            hop_length = self.info.get_feature_extractor().hop_length
            mm_data["audio"] = [
                pad_to_hop_length(audio, hop_length)
                if isinstance(audio, np.ndarray)
                else (pad_to_hop_length(audio[0], hop_length), audio[1])
                for audio in audios
            ]
            mm_kwargs = dict(
                **mm_kwargs,
            )

        hf_inputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        if (
            "audio_feature_lengths" in hf_inputs
            and "feature_attention_mask" in hf_inputs
            and (audios := mm_data.get("audio", []))
        ):
            hop_length = self.info.get_feature_extractor().hop_length
            audio_num_frames = []
            for _, audio in enumerate(audios):
                audio_length = len(audio[0]) if isinstance(audio, tuple) else len(audio)
                num_frame = (
                    (audio_length // hop_length)
                    if audio_length % hop_length == 0
                    else (audio_length // hop_length - 1)
                )
                audio_num_frames.append(num_frame)
            hf_inputs["feature_attention_mask"] = [
                torch.ones(num_frame) for num_frame in audio_num_frames
            ]
            hf_inputs["audio_feature_lengths"] = torch.tensor(audio_num_frames)
        return hf_inputs

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
        """
        Qwen3-Omni reimplements this function to handle `use_audio_in_video`.
        """
        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

        use_audio_in_video = False
        if "video" in mm_kwargs:
            for item in mm_kwargs["video"]:
                if item and item["use_audio_in_video"].data:
                    use_audio_in_video = True
                else:
                    use_audio_in_video = False

        if use_audio_in_video and "video" in mm_item_counts:
            assert "audio" in mm_item_counts
            mm_item_counts["audio"] -= mm_item_counts["video"]

        # Special case with `use_audio_in_video=True`
        if use_audio_in_video:
            if is_update_applied:
                prompt_ids = self._get_raw_input_ids(prompt_ids, use_audio_in_video)
            (
                prompt_ids,
                mm_placeholders,
            ) = self._apply_prompt_updates(
                prompt_ids,
                mm_prompt_updates,
            )
            self._validate_mm_placeholders(mm_placeholders, mm_item_counts)
        # normal case with `use_audio_in_video=False`
        elif is_update_applied:
            mm_placeholders = self._find_mm_placeholders(
                prompt_ids,
                mm_prompt_updates,
            )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
            )
        else:
            prompt_ids, mm_placeholders = self._apply_prompt_updates(
                prompt_ids,
                mm_prompt_updates,
            )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
            )

        return prompt_ids, mm_placeholders

    def get_updates_use_audio_in_video(
        self,
        thinker_config: PretrainedConfig,
        audio_len: int,
        video_grid_thw: list[int] | torch.Tensor,
        video_second_per_grid_t: float,
    ) -> list[int]:
        shift = 0
        audio_token_id = thinker_config.audio_token_id
        video_token_id = thinker_config.video_token_id
        audio_start_token_id = thinker_config.audio_start_token_id
        audio_end_token_id = thinker_config.audio_end_token_id
        spatial_merge_size = thinker_config.vision_config.spatial_merge_size
        position_id_per_seconds = thinker_config.position_id_per_seconds
        audio_token_indices = np.arange(next(iter([audio_len])))
        curr_video_grid_thw = next(iter([video_grid_thw]))
        height = curr_video_grid_thw[1] // spatial_merge_size
        width = curr_video_grid_thw[2] // spatial_merge_size
        video_token_indices = np.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
        video_token_indices = np.broadcast_to(
            video_token_indices, (video_token_indices.shape[0], height, width)
        ).reshape(-1)
        video_token_indices = (
            (video_token_indices + shift)
            * next(iter([video_second_per_grid_t]))
            * position_id_per_seconds
        )
        video_data_index, audio_data_index = 0, 0
        updates = [audio_start_token_id]
        while video_data_index < len(video_token_indices) and audio_data_index < len(
            audio_token_indices
        ):
            if (
                video_token_indices[video_data_index]
                <= audio_token_indices[audio_data_index]
            ):
                updates += [video_token_id]
                video_data_index += 1
            else:
                updates += [audio_token_id]
                audio_data_index += 1
        if video_data_index < len(video_token_indices):
            updates += [video_token_id] * (len(video_token_indices) - video_data_index)
        if audio_data_index < len(audio_token_indices):
            updates += [audio_token_id] * (len(audio_token_indices) - audio_data_index)
        updates += [audio_end_token_id]
        return updates

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        vocab = tokenizer.get_vocab()

        audio_token = processor.audio_token
        image_token = processor.image_token
        video_token = processor.video_token
        audio_token_id = vocab[audio_token]
        image_token_id = vocab[image_token]
        video_token_id = vocab[video_token]

        out_mm_data = out_mm_kwargs.get_data()
        audio_feature_lengths = out_mm_data.get("audio_feature_lengths")
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        if audio_feature_lengths is None and feature_attention_mask is None:
            audio_output_lengths = []
        elif audio_feature_lengths is not None:
            _, audio_output_lens = _get_feat_extract_output_lengths(
                audio_feature_lengths
            )
            audio_output_lengths = audio_output_lens.tolist()
        elif feature_attention_mask is not None:
            assert isinstance(feature_attention_mask, torch.Tensor)
            _, audio_output_lens = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            audio_output_lengths = audio_output_lens.tolist()

        # number of audios read from video.
        audio_in_video_item_idx = 0
        audio_item_idx = 0

        def get_replacement_qwen2_audio(item_idx: int):
            nonlocal audio_item_idx
            item_idx += audio_in_video_item_idx

            audio_item_idx += 1

            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio = audios.get(item_idx)
                raise ValueError(
                    f"The audio {audio} (len={len(audio)}) is too short "
                    "to be represented inside the model"
                )

            return [audio_token_id] * num_features

        def get_replacement_qwen2_vision(item_idx: int, modality: str):
            grid_thw = out_mm_data[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)
            merge_length = image_processor.merge_size**2

            token_id = image_token_id if modality == "image" else video_token_id
            return [token_id] * (int(grid_thw.prod()) // merge_length)

        use_audio_in_video = hf_processor_mm_kwargs.get("use_audio_in_video", False)
        thinker_config = self.info.get_hf_config()

        def get_replacement_qwen2_use_audio_in_video(item_idx: int):
            nonlocal audio_in_video_item_idx
            audio_num_features = audio_output_lengths[audio_item_idx + item_idx]
            video_grid_thw = out_mm_data["video_grid_thw"][item_idx]

            audio_in_video_item_idx += 1

            second_per_grid_ts = hf_processor_mm_kwargs.get("second_per_grid_ts", None)
            if second_per_grid_ts:
                video_second_per_grid_t = second_per_grid_ts[item_idx]
            else:
                video_second_per_grid_t = 1.0

            return self.get_updates_use_audio_in_video(
                thinker_config=thinker_config,
                audio_len=audio_num_features,
                video_grid_thw=video_grid_thw,
                video_second_per_grid_t=video_second_per_grid_t,
            )

        video_replacement_fn = (
            get_replacement_qwen2_use_audio_in_video
            if use_audio_in_video
            else partial(get_replacement_qwen2_vision, modality="video")
        )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_qwen2_audio,
            ),
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=partial(get_replacement_qwen2_vision, modality="image"),
            ),
            PromptReplacement(
                modality="video",
                target=video_token,
                replacement=video_replacement_fn,
            ),
        ]

    def _validate_mm_placeholders(
        self,
        mm_placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
        mm_item_counts: Mapping[str, int],
    ) -> None:
        BaseMultiModalProcessor[
            Qwen2_5OmniThinkerProcessingInfo
        ]._validate_mm_placeholders(self, mm_placeholders, mm_item_counts)

    def _get_raw_input_ids(
        self,
        token_ids: list[int],
        use_audio_in_video: bool = False,
    ) -> list[int]:
        tokenizer = self.info.get_tokenizer()
        vision_bos_token = tokenizer.encode(tokenizer.vision_bos_token)[0]
        vision_eos_token = tokenizer.encode(tokenizer.vision_eos_token)[0]
        audio_bos_token = tokenizer.encode(tokenizer.audio_bos_token)[0]
        audio_eos_token = tokenizer.encode(tokenizer.audio_eos_token)[0]
        audio_token = tokenizer.encode("<|audio_pad|>")[0]
        image_token = tokenizer.encode("<|image_pad|>")[0]
        video_token = tokenizer.encode("<|video_pad|>")[0]

        result = token_ids[:]
        if use_audio_in_video:
            while True:
                start = None
                for i in range(len(result) - 1):
                    if result[i : i + 2] == [vision_bos_token, audio_bos_token]:
                        start = i
                        break
                if start is not None:
                    end = None
                    for i in range(start + 2, len(result) - 1):
                        if result[i : i + 2] == [audio_eos_token, vision_eos_token]:
                            end = i
                            break
                    if end is not None:
                        result = (
                            result[:start]
                            + [vision_bos_token, video_token, vision_eos_token]
                            + result[end + 2 :]
                        )
                else:
                    break

        for mm_token in [audio_token, image_token, video_token]:
            compressed = []
            for x in result:
                if x != mm_token or (not compressed or compressed[-1] != mm_token):
                    compressed.append(x)
            result = compressed

        return result


class Qwen3OmniMoeConditionalGenerationMixin(Qwen2_5OmniConditionalGenerationMixin):
    def _validate_and_reshape_mm_tensor(
        self, mm_input: object, name: str, dim: int = 0
    ) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
        if name == "feature_attention_mask":
            dim = -1
        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input), dim=dim)
        else:
            if isinstance(mm_input[0], list):
                return torch.concat(
                    [torch.concat(mm_input[i], dim=dim) for i in range(len(mm_input))],
                    dim=dim,
                )
            else:
                return torch.concat(mm_input, dim=dim)

    def _process_audio_input(
        self,
        audio_input: Qwen2AudioFeatureInputs,
        audio_hashes: list[str] = None,
        cached_audio_features: torch.Tensor = None,
    ) -> torch.Tensor:
        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]

        if input_features.ndim == 3:
            assert input_features.shape[0] == 1
            input_features = input_features.squeeze(0)

        if not isinstance(audio_feature_lengths, torch.Tensor):
            audio_feature_lengths = torch.cat(audio_feature_lengths)
        if audio_feature_lengths.ndim == 2:
            audio_feature_lengths = audio_feature_lengths.reshape(-1)

        audio_feat_lengths, audio_output_lengths = _get_feat_extract_output_lengths(
            audio_feature_lengths
        )

        audio_outputs = self.audio_tower(
            input_features.to(self.audio_tower.dtype),
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_feat_lengths,
        )
        audio_features = audio_outputs.last_hidden_state
        return audio_features.split(audio_output_lengths.tolist())


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    Qwen3OmniMoeConditionalGenerationMixin,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "thinker.lm_head.": "language_model.lm_head.",
            "thinker.model.": "language_model.model.",
            "thinker.": "",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"

        raise ValueError("Only image, video or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        thinker_config: Qwen3OmniMoeThinkerConfig = (
            vllm_config.model_config.hf_config.thinker_config
        )
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = thinker_config
        self.multimodal_config = multimodal_config

        # force "use_flash_attention_2=True" to audio tower to align
        # the results.
        if flash_attn is not None:
            audio_config = thinker_config.audio_config
            audio_config._attn_implementation_autoset = True
            audio_config._attn_implementation = "flash_attention_2"
        else:
            logger.warning(
                "flash_attn is not available, the model may not yield the "
                "exactly same result as the transformers implementation "
                "in the audio tower part."
            )

        self.audio_tower = Qwen3OmniMoeAudioEncoder(thinker_config.audio_config)

        self.visual = Qwen3Omni_VisionTransformer(
            vision_config=thinker_config.vision_config,
            norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )
        self.quant_config = quant_config

        self.language_model = Qwen3MoeLLMForCausalLM(
            vllm_config=vllm_config.with_hf_config(
                thinker_config.text_config, architectures=["Qwen3MoeForCausalLM"]
            ),
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        self.use_deepstack = hasattr(
            thinker_config.vision_config, "deepstack_visual_indexes"
        )
        self.deepstack_num_level = (
            len(thinker_config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        # register buffer for deepstack
        self.deepstack_input_embeds = (
            [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    thinker_config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
            if self.use_deepstack
            else None
        )
        self.visual_dim = thinker_config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

    def _get_deepstack_input_embeds(self, num_tokens: int) -> IntermediateTensors:
        # get deepstack_input_embeds from buffer, and clear the buffer
        return IntermediateTensors(
            {
                f"deepstack_input_embeds_{idx}": self.deepstack_input_embeds[idx][
                    :num_tokens
                ]
                for idx in range(self.deepstack_num_level)
            }
        )

    def _set_deepstack_input_embeds(self, deepstack_input_embeds: torch.Tensor) -> None:
        # set deepstack_input_embeds to buffer
        num_tokens = deepstack_input_embeds.size(1)
        if num_tokens > self.deepstack_input_embeds[0].size(0):
            self.deepstack_input_embeds = [
                torch.zeros(
                    num_tokens,
                    self.config.text_config.hidden_size,
                    device=self.deepstack_input_embeds[0].device,
                    dtype=self.deepstack_input_embeds[0].dtype,
                )
                for _ in range(self.deepstack_num_level)
            ]
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][:num_tokens].copy_(
                deepstack_input_embeds[idx]
            )

    def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
        # clear deepstack_input_embeds in buffer
        if num_tokens > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_tokens].zero_()

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
            if (
                input_key in ("input_audio_features")
                and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(
                    **kwargs
                )
        return mm_input_by_modality

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> MultiModalEmbeddings | None:
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
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += tuple(video_embeddings)
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                multimodal_embeddings += tuple(audio_embeddings)
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self._get_text_embeddings(
            input_ids,
            self.language_model.get_input_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        deepstack_input_embeds = None
        # TODO (ywang96): support overlapping modalitiy embeddings so that
        # `use_audio_in_video` will work on V1.
        # split the feat dim to obtain multi-scale visual feature
        has_vision_embeddings = [
            embeddings.shape[-1] != self.config.text_config.hidden_size
            for embeddings in multimodal_embeddings
        ]
        if self.visual.deepstack_visual_indexes is not None and any(
            has_vision_embeddings
        ):
            multiscale_len = len(self.visual.deepstack_visual_indexes)
            multimodal_embeddings_multiscale = []
            is_vision = torch.zeros_like(is_multimodal)
            mm_positions = torch.nonzero(is_multimodal, as_tuple=True)[0]
            mm_position_idx = 0
            for index, embeddings in enumerate(multimodal_embeddings):
                num_tokens = embeddings.shape[0]
                current_positions = mm_positions[
                    mm_position_idx : mm_position_idx + num_tokens
                ]

                # Vision embeddings
                if embeddings.shape[-1] != self.config.text_config.hidden_size:
                    visual_dim = embeddings.shape[-1] // (multiscale_len + 1)
                    multi_dim = visual_dim * multiscale_len
                    embeddings_main, embeddings_multiscale = torch.split(
                        embeddings, [visual_dim, multi_dim], dim=-1
                    )
                    multimodal_embeddings[index] = embeddings_main
                    multimodal_embeddings_multiscale.append(embeddings_multiscale)
                    is_vision[current_positions] = True

                # Audio embeddings
                else:
                    is_vision[current_positions] = False

                mm_position_idx += num_tokens

            deepstack_input_embeds = inputs_embeds.new_zeros(
                inputs_embeds.size(0), multiscale_len * inputs_embeds.size(1)
            )
            deepstack_input_embeds = _merge_multimodal_embeddings(
                inputs_embeds=deepstack_input_embeds,
                multimodal_embeddings=multimodal_embeddings_multiscale,
                is_multimodal=is_vision,
            )
            deepstack_input_embeds = (
                deepstack_input_embeds.view(
                    inputs_embeds.shape[0], multiscale_len, visual_dim
                )
                .permute(1, 0, 2)
                .contiguous()
            )
            self._set_deepstack_input_embeds(deepstack_input_embeds)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        if (
            self.use_deepstack
            and inputs_embeds is not None
            and get_pp_group().is_first_rank
        ):
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )
        else:
            deepstack_input_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
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
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["talker.", "code2wav."],
        )
        loaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        return loaded_weights

    @classmethod
    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: list[list[int]] | torch.Tensor | None,
        video_grid_thw: list[list[int]] | torch.Tensor | None,
        second_per_grid_ts: list[float] | None = None,
        context_len: int = 0,
        seq_len: int | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        config = hf_config.thinker_config
        if isinstance(image_grid_thw, list):
            image_grid_thw = torch.tensor(image_grid_thw)
        if isinstance(video_grid_thw, list):
            video_grid_thw = torch.tensor(video_grid_thw)
        input_ids = torch.tensor(input_tokens)
        if input_ids is None or input_ids.ndim != 1:
            raise ValueError("_omni3_get_input_positions_tensor expects 1D input_ids")

        seq_len = input_ids.shape[0]
        if audio_feature_lengths is not None and not isinstance(
            audio_feature_lengths, torch.Tensor
        ):
            audio_feature_lengths = torch.as_tensor(
                audio_feature_lengths, dtype=torch.long
            )
        if second_per_grid_ts is None:
            if video_grid_thw is not None and video_grid_thw.numel() > 0:
                second_per_grids = torch.ones(
                    video_grid_thw.shape[0], dtype=torch.float32
                )
            else:
                second_per_grids = torch.tensor([], dtype=torch.float32)
        else:
            second_per_grids = torch.tensor(second_per_grid_ts, dtype=torch.float32)

        spatial_merge_size = config.vision_config.spatial_merge_size
        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        audio_token_id = config.audio_token_id
        vision_start_token_id = config.vision_start_token_id
        audio_start_token_id = config.audio_start_token_id
        position_id_per_seconds = config.position_id_per_seconds

        vision_start_indices = torch.argwhere(
            input_ids == vision_start_token_id
        ).squeeze(1)
        if vision_start_indices.numel() > 0:
            vision_tokens = input_ids[vision_start_indices + 1]
        else:
            vision_tokens = input_ids.new_empty((0,), dtype=input_ids.dtype)
        audio_nums = torch.sum(input_ids == audio_start_token_id)
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (
            (vision_tokens == audio_start_token_id).sum()
            if use_audio_in_video
            else (vision_tokens == video_token_id).sum()
        )

        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0
        image_idx = 0
        video_idx = 0
        audio_idx = 0
        remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums  # noqa: E501
        multimodal_nums = (
            image_nums + audio_nums
            if use_audio_in_video
            else image_nums + video_nums + audio_nums
        )  # noqa: E501

        for _ in range(multimodal_nums):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                remain_videos > 0 or remain_images > 0
            ):
                ed_vision_start = input_tokens.index(vision_start_token_id, st)
            else:
                ed_vision_start = len(input_tokens) + 1
            if audio_token_id in input_tokens and remain_audios > 0:
                ed_audio_start = input_tokens.index(audio_start_token_id, st)
            else:
                ed_audio_start = len(input_tokens) + 1
            min_ed = min(ed_vision_start, ed_audio_start)

            if min_ed == ed_audio_start:
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, dtype=torch.long)
                        .view(1, -1)
                        .expand(3, -1)
                        + st_idx
                    )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                bos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(bos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                _, audio_len = _get_feat_extract_output_lengths(
                    audio_feature_lengths[audio_idx]
                )
                llm_pos_ids = (
                    torch.arange(audio_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                llm_pos_ids_list.append(llm_pos_ids)
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                eos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(eos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st += text_len + bos_len + audio_len + eos_len
                audio_idx += 1
                remain_audios -= 1
            elif (
                min_ed == ed_vision_start
                and input_ids[ed_vision_start + 1] == image_token_id
            ):
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, dtype=torch.long)
                        .view(1, -1)
                        .expand(3, -1)
                        + st_idx
                    )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                bos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(bos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                grid_t = image_grid_thw[image_idx][0]
                grid_hs = image_grid_thw[:, 1]
                grid_ws = image_grid_thw[:, 2]
                t_index = torch.arange(grid_t) * position_id_per_seconds
                llm_pos_ids = get_llm_pos_ids_for_vision(
                    st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                )
                image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                llm_pos_ids_list.append(llm_pos_ids)
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                eos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(eos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st += text_len + bos_len + image_len + eos_len
                image_idx += 1
                remain_images -= 1
            elif (
                min_ed == ed_vision_start
                and input_ids[ed_vision_start + 1] == video_token_id
                and not use_audio_in_video
            ):
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, dtype=torch.long)
                        .view(1, -1)
                        .expand(3, -1)
                        + st_idx
                    )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                bos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(bos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                grid_t = video_grid_thw[video_idx][0]
                grid_hs = video_grid_thw[:, 1]
                grid_ws = video_grid_thw[:, 2]
                t_index = (
                    torch.arange(grid_t)
                    * float(second_per_grids[video_idx].item())
                    * position_id_per_seconds
                )
                llm_pos_ids = get_llm_pos_ids_for_vision(
                    st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                )
                video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                llm_pos_ids_list.append(llm_pos_ids)
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                eos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(eos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st += text_len + bos_len + video_len + eos_len
                video_idx += 1
                remain_videos -= 1
            elif (
                min_ed == ed_vision_start
                and ed_vision_start + 1 == ed_audio_start
                and use_audio_in_video
            ):
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, dtype=torch.long)
                        .view(1, -1)
                        .expand(3, -1)
                        + st_idx
                    )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                bos_len = 1
                bos_block = (
                    torch.arange(bos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                llm_pos_ids_list.append(bos_block)
                llm_pos_ids_list.append(bos_block)
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                _, audio_len = _get_feat_extract_output_lengths(
                    audio_feature_lengths[audio_idx]
                )
                audio_llm_pos_ids = (
                    torch.arange(audio_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                grid_t = video_grid_thw[video_idx][0]
                grid_hs = video_grid_thw[:, 1]
                grid_ws = video_grid_thw[:, 2]
                t_index = (
                    torch.arange(grid_t)
                    * float(second_per_grids[video_idx].item())
                    * position_id_per_seconds
                )
                video_llm_pos_ids = get_llm_pos_ids_for_vision(
                    st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                )
                video_data_index, audio_data_index = 0, 0
                while (
                    video_data_index < video_llm_pos_ids.shape[-1]
                    and audio_data_index < audio_llm_pos_ids.shape[-1]
                ):
                    if (
                        video_llm_pos_ids[0][video_data_index]
                        <= audio_llm_pos_ids[0][audio_data_index]
                    ):
                        llm_pos_ids_list.append(
                            video_llm_pos_ids[
                                :, video_data_index : video_data_index + 1
                            ]
                        )
                        video_data_index += 1
                    else:
                        llm_pos_ids_list.append(
                            audio_llm_pos_ids[
                                :, audio_data_index : audio_data_index + 1
                            ]
                        )
                        audio_data_index += 1
                if video_data_index < video_llm_pos_ids.shape[-1]:
                    llm_pos_ids_list.append(
                        video_llm_pos_ids[
                            :, video_data_index : video_llm_pos_ids.shape[-1]
                        ]
                    )
                if audio_data_index < audio_llm_pos_ids.shape[-1]:
                    llm_pos_ids_list.append(
                        audio_llm_pos_ids[
                            :, audio_data_index : audio_llm_pos_ids.shape[-1]
                        ]
                    )
                video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                eos_len = 1
                eos_block = (
                    torch.arange(eos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                llm_pos_ids_list.append(eos_block)
                llm_pos_ids_list.append(eos_block)
                st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2  # noqa: E501
                audio_idx += 1
                video_idx += 1
                remain_videos -= 1
                remain_audios -= 1

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1)
                + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if llm_positions.shape[1] != seq_len:
            raise RuntimeError("Position ids length mismatch with input ids length")

        mrope_position_delta = llm_positions.max() + 1 - seq_len
        return llm_positions, mrope_position_delta
