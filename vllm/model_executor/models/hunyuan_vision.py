# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# coding=utf-8
# Copyright 2025 The HunYuan team.
# Copyright 2025 The vLLM team.
# Copyright 2025 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only HunYuan-VL model compatible with HuggingFace weights."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Annotated, Any, Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature

from vllm.config import MultiModalConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    ImageItem,
    ModalityData,
    MultiModalDataDict,
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
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.hunyuan_vl import (
    HunYuanVLConfig,
    HunYuanVLVisionConfig,
)
from vllm.transformers_utils.processors.hunyuan_vl import HunYuanVLProcessor
from vllm.transformers_utils.processors.hunyuan_vl_image import smart_resize
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
    SupportsXDRoPE,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

logger = init_logger(__name__)

# === Vision Inputs === #


class HunYuanVLImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - np: Number of patches
        - ni: Number of images
        - cps: Number of channels * patch_size * patch_size
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


class HunYuanVLImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - nf: Number of image features
        - hs: Hidden size
        - ni: Number of images
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


HunYuanVLImageInputs: TypeAlias = (
    HunYuanVLImagePixelInputs | HunYuanVLImageEmbeddingInputs
)

# === Vision Encoder === #


class HunYuanVisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.dense_h_to_4h",
            disable_tp=use_data_parallel,
        )
        self.dense_4h_to_h = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.dense_4h_to_h",
            disable_tp=use_data_parallel,
        )
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        x_up, _ = self.dense_h_to_4h(x)
        x_down, _ = self.dense_4h_to_h(self.act_fn(x_up))
        return x_down


class HunYuanVisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
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
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )

        self.o_proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=use_data_parallel,
        )

        self.scale = self.hidden_size_per_attention_head**-0.5
        self.attn = MMEncoderAttention(
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            self.scale,
            prefix=f"{prefix}.attn",
            multimodal_config=multimodal_config,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        out = self.attn(q, k, v)
        output, _ = self.o_proj(out)
        return output


class HunYuanVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.input_layernorm = norm_layer(dim)
        self.post_attention_layernorm = norm_layer(dim)
        self.self_attn = HunYuanVisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.self_attn",
            use_data_parallel=use_data_parallel,
        )
        self.mlp = HunYuanVisionMLP(
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
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class HunYuanVisionPatchEmbed(nn.Module):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.spatial_merge_size = config.spatial_merge_size
        self.interpolate_mode = config.interpolate_mode

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.max_num_patches = (config.max_image_size // self.patch_size) ** 2

        self.num_positions = self.max_num_patches + 1
        self.position_edge = int(self.num_positions**0.5)
        # first token is cls token, skip it
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        self.patch_pos_embed = None

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor:
        num_patches = pixel_values.size(0)
        pixel_values = pixel_values.reshape(
            num_patches, self.num_channels, self.patch_size, self.patch_size
        )

        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.squeeze(-1).squeeze(-1).unsqueeze(0)

        if self.patch_pos_embed is None:
            patch_pos_shape = (
                1,
                self.position_edge,
                self.position_edge,
                self.embed_dim,
            )
            self.patch_pos_embed = (
                self.position_embedding.weight[1:, :]
                .reshape(patch_pos_shape)
                .permute(0, 3, 1, 2)
                .float()
            )

        patch_pos_embed_list = []
        for grid in grid_thw:
            _, h0, w0 = grid
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            h0, w0 = h0 + 0.1, w0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                self.patch_pos_embed,
                scale_factor=(h0 / self.position_edge, w0 / self.position_edge),
                mode=self.interpolate_mode,
                align_corners=False,
            )

            patch_pos_embed = (
                patch_pos_embed.reshape(self.embed_dim, -1)
                .transpose(0, 1)
                .unsqueeze(0)
                .to(patch_embeds.dtype)
            )
            patch_pos_embed_list.append(patch_pos_embed)

        patch_pos_embed = torch.cat(patch_pos_embed_list, dim=1)
        embeddings = patch_embeds + patch_pos_embed

        return embeddings


class HunYuanVisionPatchMerger(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_merge_size=2,
        rms_norm_eps=1e-5,
        prefix="",
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        embed_std = out_channels**-0.5

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * 2,
                kernel_size=spatial_merge_size,
                stride=spatial_merge_size,
            ),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=1),
        )
        self.mlp = nn.Linear(in_channels * 4, out_channels)

        self.image_newline = nn.Parameter(torch.randn(in_channels * 4) * embed_std)
        self.image_begin = nn.Parameter(torch.randn(out_channels) * embed_std)
        self.image_end = nn.Parameter(torch.randn(out_channels) * embed_std)
        self.image_sep = nn.Parameter(torch.randn(out_channels) * embed_std)

        self.before_rms = RMSNorm(in_channels, eps=rms_norm_eps)
        self.after_rms = RMSNorm(out_channels, eps=rms_norm_eps)

    def forward(self, x, size=(16, 16)):
        x = self.before_rms(x)

        h, w = size
        dtype = x.dtype
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, h, w)

        x = self.proj(x)  # b,c,h,w
        b, c, h, w = x.shape
        x = torch.cat(
            [x, self.image_newline.reshape(1, c, 1, 1).expand(b, c, h, 1).to(dtype)],
            dim=-1,
        )
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        x = self.mlp(x)

        begin = self.image_begin.reshape(1, 1, -1).expand(b, 1, x.shape[-1]).to(dtype)
        end = self.image_end.reshape(1, 1, -1).expand(b, 1, x.shape[-1]).to(dtype)
        x = torch.cat([begin, x, end], dim=1)

        return self.after_rms(x)


class HunYuanVisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config: HunYuanVLVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        multimodal_config: MultiModalConfig | None = None,
        attn_backend_override: AttentionBackendEnum | None = None,
    ) -> None:
        super().__init__()

        num_hidden_layers = vision_config.num_hidden_layers
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_attention_heads
        self.spatial_merge_size = vision_config.spatial_merge_size

        from vllm.compilation.backends import set_model_tag

        with set_model_tag("HunYuanVisionPatchEmbed"):
            self.embeddings = HunYuanVisionPatchEmbed(vision_config)

        norm_layer = partial(nn.LayerNorm, eps=vision_config.rms_norm_eps)

        with set_model_tag("HunYuanVisionBlock"):
            self.layers = nn.ModuleList(
                [
                    HunYuanVisionBlock(
                        dim=vision_config.hidden_size,
                        num_heads=vision_config.num_attention_heads,
                        mlp_hidden_dim=vision_config.intermediate_size,
                        act_fn=get_act_fn(vision_config.hidden_act),
                        norm_layer=norm_layer,
                        quant_config=quant_config,
                        multimodal_config=multimodal_config,
                        prefix=f"{prefix}.layers.{layer_idx}",
                        use_data_parallel=use_data_parallel,
                    )
                    for layer_idx in range(num_hidden_layers)
                ]
            )

        with set_model_tag("HunYuanVisionPatchMerger"):
            self.perceive = HunYuanVisionPatchMerger(
                vision_config.hidden_size,
                vision_config.out_hidden_size,
                spatial_merge_size=vision_config.spatial_merge_size,
                rms_norm_eps=vision_config.rms_norm_eps,
                prefix=f"{prefix}.perceive",
            )

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embeddings.patch_embedding.weight.device

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        # patchify
        seq_len = x.size(0)
        cu_seqlens: list = [0]

        hidden_states = x.to(device=self.device, dtype=self.dtype)
        # embeddings = patch_embeds + patch_pos_embed
        hidden_states = self.embeddings(hidden_states, grid_thw)

        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            cu_seqlens.append(h * w)

        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)

        cu_seqlens = cu_seqlens.to(device=self.device, non_blocking=True)

        hidden_states = hidden_states.reshape(seq_len, -1)
        hidden_states = hidden_states.unsqueeze(0)

        # build per-image lengths once
        split_lengths = [int(h) * int(w) for (_, h, w) in grid_thw]
        for layer in self.layers:
            # hidden_states: (1, T_total, D)
            parts = hidden_states.split(split_lengths, dim=1)  # list of (1, L_i, D)
            parts = [layer(p) for p in parts]
            hidden_states = torch.cat(parts, dim=1)

        # adapter
        split_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        split_items = hidden_states.split(split_lengths, dim=1)
        image_embeds_list = []
        for grid, split_item in zip(grid_thw, split_items):
            image_embeds_list.append(
                self.perceive(split_item.contiguous(), size=grid[1:]).squeeze(0)
            )

        return image_embeds_list

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv", ".q_proj", "q"),
            (".qkv", ".k_proj", "k"),
            (".qkv", ".v_proj", "v"),
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


def _hunyuan_vl_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
    image_grid_sizes = image_grid_thw.prod(-1)
    return dict(
        pixel_values=MultiModalFieldConfig.flat_from_sizes("image", image_grid_sizes),
        image_embeds=MultiModalFieldConfig.flat_from_sizes("image", image_grid_sizes),
        image_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
    )


class HunYuanVLMultiModalDataParser(MultiModalDataParser):
    def _parse_image_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[ImageItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields={"image_embeds", "image_grid_thw"},
                fields_factory=_hunyuan_vl_field_config,
            )

        return super()._parse_image_data(data)


class HunYuanVLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(HunYuanVLConfig)

    def get_hf_processor(
        self,
        **kwargs: object,
    ) -> HunYuanVLProcessor:
        return self.ctx.get_hf_processor(
            HunYuanVLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_image_processor(
        self,
        **kwargs: object,
    ) -> HunYuanVLProcessor:
        return self.get_hf_processor(**kwargs).image_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        max_image_tokens = self.get_max_image_tokens()
        # TODO: support video
        max_video_tokens = 0
        return {"image": max_image_tokens, "video": max_video_tokens}

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
        image_processor: HunYuanVLProcessor | None,
    ) -> tuple[ImageSize, int]:
        if image_processor is None:
            image_processor = self.get_image_processor()

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        spatial_merge_size = vision_config.spatial_merge_size

        if do_resize:
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * spatial_merge_size,
                min_pixels=image_processor.min_pixels,
                max_pixels=image_processor.max_pixels,
            )
            preprocessed_size = ImageSize(width=resized_width, height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width, height=image_height)

        grid_t = 1
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_vision_tokens = (
            grid_t * grid_h // spatial_merge_size * (grid_w // spatial_merge_size + 1)
            + 2
        )

        return preprocessed_size, num_vision_tokens

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: HunYuanVLProcessor | None,
    ) -> int:
        _, num_image_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            image_processor=image_processor,
        )
        return num_image_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        max_image_size, _ = self._get_vision_info(
            image_width=512,
            image_height=8192,
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


class HunYuanVLDummyInputsBuilder(BaseDummyInputsBuilder[HunYuanVLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        hf_processor = self.info.get_hf_processor()
        image_token: str = hf_processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 1)

        target_width, target_height = self.info.get_image_size_with_most_features()

        return {
            "image": self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images
            ),
        }


class HunYuanVLMultiModalProcessor(BaseMultiModalProcessor[HunYuanVLProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return HunYuanVLMultiModalDataParser()

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        return self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=prompt, **mm_data),
            dict(**mm_kwargs, **tok_kwargs),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)

        placeholder = {
            "image": hf_processor.image_token_id,
        }

        merge_size = image_processor.merge_size

        def get_replacement_hunyuan_vl(item_idx: int, modality: str):
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            _, grid_h, grid_w = grid_thw
            num_tokens = (int(grid_h) // merge_size) * (
                int(grid_w) // merge_size + 1
            ) + 2
            return [placeholder[modality]] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement_hunyuan_vl, modality=modality),
            )
            for modality in ("image",)
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _hunyuan_vl_field_config(hf_inputs)


@MULTIMODAL_REGISTRY.register_processor(
    HunYuanVLMultiModalProcessor,
    info=HunYuanVLProcessingInfo,
    dummy_inputs=HunYuanVLDummyInputsBuilder,
)
class HunYuanVLForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
    SupportsXDRoPE,
):
    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "vit.vit.": "visual.",
            "vit.": "visual.",
            "model.": "language_model.model.",
        }
    )

    supports_encoder_tp_data = True

    def get_xdrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> torch.Tensor:
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {"image_grid_thw"},
        )
        image_grid_thw = [item.tolist() for item in kwargs.get("image_grid_thw", [])]

        hf_config = self.config
        image_start_token_id = hf_config.image_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        xd_num = len(hf_config.rope_scaling["xdrope_section"])

        input_tokens_tensor = torch.tensor(input_tokens)
        image_start_indices = torch.argwhere(
            input_tokens_tensor == image_start_token_id
        ).squeeze(1)

        p_index = torch.arange(len(input_tokens_tensor))
        w_index = torch.arange(len(input_tokens_tensor))
        h_index = torch.arange(len(input_tokens_tensor))
        t_index = torch.arange(len(input_tokens_tensor))
        for image_index in range(len(image_start_indices)):
            # +1 : first image_token, +2: for xdrope positions
            pos = image_start_indices[image_index] + 2
            t, h, w = image_grid_thw[image_index]
            _, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )

            token_num = (llm_grid_w + 1) * llm_grid_h
            w_index[pos : pos + token_num].copy_(
                torch.arange(0, llm_grid_w + 1)
                .reshape(1, -1)
                .expand(llm_grid_h, -1)
                .reshape(-1)
            )
            h_index[pos : pos + token_num].copy_(
                torch.arange(0, llm_grid_h)
                .reshape(-1, 1)
                .expand(-1, llm_grid_w + 1)
                .reshape(-1)
            )
            t_index[pos : pos + token_num] = image_index

        if xd_num == 4:
            llm_positions = torch.stack([p_index, w_index, h_index, t_index])
        elif xd_num == 3:
            llm_positions = torch.stack([w_index, h_index, t_index])

        return llm_positions

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<｜hy_place▁holder▁no▁100｜><｜hy_place▁holder▁no▁102｜><｜hy_place▁holder▁no▁101｜>"  # noqa: E501

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: HunYuanVLConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        with self._mark_tower_model(vllm_config, {"image"}):
            attn_backend_override = (
                multimodal_config.mm_encoder_attn_backend
                if multimodal_config is not None
                else None
            )
            self.visual = HunYuanVisionTransformer(
                config.vision_config,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                multimodal_config=multimodal_config,
                attn_backend_override=attn_backend_override,
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model.model"),
                architectures=[
                    "HunYuanDenseV1ForCausalLM",
                    "HunYuanMoEV1ForCausalLM",
                ],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> HunYuanVLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        # TODO: refine
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)
        if len(pixel_values.shape) == 3:
            last_dim = pixel_values.shape[-1]
            pixel_values = pixel_values.reshape(-1, last_dim)
            image_grid_thw = image_grid_thw.reshape(-1, 3)

        if pixel_values is not None:
            return HunYuanVLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return HunYuanVLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _process_image_input(
        self, image_input: HunYuanVLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"]

            # TODO: use_data_parallel (split image_embeds in visual)
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)

        return image_embeds

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
        return mm_input_by_modality

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
                multimodal_embeddings += tuple(image_embeddings)
        return multimodal_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model(
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
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model.model",
            connector="visual.perceive",
            tower_model="visual",
        )
