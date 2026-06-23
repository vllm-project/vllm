# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Unlimited-OCR model compatible with HuggingFace weights."""

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
from transformers import CLIPVisionConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.multimodal.parse import ImageSize
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekVLV2Config
from vllm.transformers_utils.processors.deepseek_ocr import DeepseekOCRProcessor

from ...transformers_utils.processors.deepseek_ocr import count_tiles
from .deepencoder import DeepCLIPVisionTransformer, build_sam_vit_b
from .deepseek_ocr2 import (
    DeepseekOCR2ForCausalLM,
    DeepseekOCR2MultiModalProcessor,
    DeepseekOCR2ProcessingInfo,
)
from .deepseek_vl2 import MlpProjector

_IMAGE_TOKEN = "<image>"
_UNLIMITED_IMAGE_SIZE = 640
_UNLIMITED_BASE_SIZE = 1024


class UnlimitedOCRProcessingInfo(DeepseekOCR2ProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        processor_config = dict(
            image_size=_UNLIMITED_IMAGE_SIZE,
            base_size=_UNLIMITED_BASE_SIZE,
            crop_mode=True,
            strategy="v1",
        )
        return self.ctx.get_hf_processor(
            DeepseekOCRProcessor,
            **{**processor_config, **kwargs},
        )

    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, cropping: bool = True
    ) -> int:
        image_size = _UNLIMITED_IMAGE_SIZE
        base_size = _UNLIMITED_BASE_SIZE
        patch_size = 16
        downsample_ratio = 4

        if cropping and (image_width > image_size or image_height > image_size):
            num_width_tiles, num_height_tiles = count_tiles(
                image_width, image_height, image_size=image_size
            )
        else:
            num_width_tiles = num_height_tiles = 1

        global_side = math.ceil((base_size // patch_size) / downsample_ratio)
        local_side = math.ceil((image_size // patch_size) / downsample_ratio)

        global_tokens = global_side * (global_side + 1)
        local_tokens = 0
        if num_width_tiles > 1 or num_height_tiles > 1:
            local_tokens = (
                (local_side * num_width_tiles + 1) * local_side * num_height_tiles
            )
        return global_tokens + local_tokens + 1

    def get_image_size_with_most_features(self) -> ImageSize:
        return ImageSize(
            width=_UNLIMITED_IMAGE_SIZE * 2,
            height=_UNLIMITED_IMAGE_SIZE * 2,
        )


class UnlimitedOCRDummyInputsBuilder(
    BaseDummyInputsBuilder[UnlimitedOCRProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return self.info.get_hf_processor().image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        max_image_size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=max_image_size.width,
                height=max_image_size.height,
                num_images=num_images,
            )
        }


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekOCR2MultiModalProcessor,
    info=UnlimitedOCRProcessingInfo,
    dummy_inputs=UnlimitedOCRDummyInputsBuilder,
)
class UnlimitedOCRForCausalLM(DeepseekOCR2ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        compat_config = DeepseekVLV2Config(**config.to_dict())
        self.config = config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.vision_config = compat_config.vision_config
        self.projector_config = compat_config.projector_config
        self.text_config = compat_config.text_config
        self.text_config.architectures = ["DeepseekV2ForCausalLM"]
        self.text_config.model_type = "deepseek_v2"

        model_config = vllm_config.model_config
        tokenizer = cached_tokenizer_from_config(model_config)
        self.image_token_id = tokenizer.vocab[_IMAGE_TOKEN]

        with self._mark_tower_model(vllm_config, "image"):
            self.sam_model = build_sam_vit_b()
            clip_vision_config = CLIPVisionConfig(
                hidden_size=1024,
                intermediate_size=4096,
                num_attention_heads=16,
                num_hidden_layers=24,
                image_size=224,
                patch_size=14,
                projection_dim=512,
                layer_norm_eps=1e-5,
            )
            self.vision_model = DeepCLIPVisionTransformer(
                config=clip_vision_config,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "vision_model"),
            )
            self.projector = MlpProjector(self.projector_config)
            self.tile_tag = config.tile_tag
            self.global_view_pos = config.global_view_pos

            n_embed = self.projector_config.n_embed
            embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
            if self.tile_tag != "2D":
                raise ValueError(
                    f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
                )
            self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=self.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _encode_global_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        sam_features = self.sam_model(image_tensor)
        clip_features = self.vision_model(image_tensor, sam_features)
        features = torch.cat(
            [clip_features[:, 1:], sam_features.flatten(2).permute(0, 2, 1)],
            dim=-1,
        )
        features = self.projector(features)
        _, hw, dim = features.shape
        side = int(hw**0.5)
        features = features.view(side, side, dim)
        features = torch.cat(
            [features, self.image_newline[None, None, :].expand(side, 1, dim)],
            dim=1,
        )
        return features.view(-1, dim)

    def _encode_local_features(
        self, patches: torch.Tensor, crop_shape: torch.Tensor
    ) -> torch.Tensor | None:
        if torch.sum(patches).item() == 0:
            return None

        sam_features = self.sam_model(patches)
        clip_features = self.vision_model(patches, sam_features)
        features = torch.cat(
            [clip_features[:, 1:], sam_features.flatten(2).permute(0, 2, 1)],
            dim=-1,
        )
        features = self.projector(features)
        _, hw, dim = features.shape
        side = int(hw**0.5)
        width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]
        features = features.view(
            height_crop_num, width_crop_num, side, side, dim
        ).permute(0, 2, 1, 3, 4)
        features = features.reshape(height_crop_num * side, width_crop_num * side, dim)
        features = torch.cat(
            [
                features,
                self.image_newline[None, None, :].expand(
                    height_crop_num * side, 1, dim
                ),
            ],
            dim=1,
        )
        return features.view(-1, dim)

    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
    ) -> NestedTensors:
        images_in_this_batch = []

        is_tiled = (images_spatial_crop[:, 0] > 1) | (images_spatial_crop[:, 1] > 1)
        patches_per_image = torch.where(is_tiled, images_spatial_crop.prod(dim=-1), 0)
        images_crop = images_crop.split(patches_per_image.tolist())
        for jdx in range(images_spatial_crop.size(0)):
            patches = images_crop[jdx]
            image_ori = pixel_values[[jdx]]

            global_features = self._encode_global_features(image_ori)
            local_features = self._encode_local_features(
                patches, images_spatial_crop[jdx]
            )

            if local_features is not None:
                combined = torch.cat(
                    [local_features, global_features, self.view_seperator[None, :]],
                    dim=0,
                )
            else:
                combined = torch.cat(
                    [global_features, self.view_seperator[None, :]], dim=0
                )

            images_in_this_batch.append(combined)

        return images_in_this_batch

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="projector",
            tower_model=["sam_model", "vision_model"],
        )
