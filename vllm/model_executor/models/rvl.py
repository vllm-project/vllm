# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Mapping
from typing import Optional
import torch
import torch.nn as nn
from transformers.activations import GELUActivation
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageSize
from vllm.multimodal.inputs import MultiModalDataDict
from .utils import WeightsMapper
from .llava_onevision import LlavaOnevisionForConditionalGeneration
from .llava_next import LlavaNextProcessingInfo, LlavaDummyInputsBuilder, LlavaNextMultiModalProcessor


class RVLProcessingInfo(LlavaNextProcessingInfo):
    
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(**kwargs)

    def _get_num_unpadded_features(
        self,
        *,
        original_height: int,
        original_width: int,
        npatches: int,
        num_patch_height: int,
        num_patch_width: int,
    ) -> tuple[int, int]:
        current_height = npatches * num_patch_height
        current_width = npatches * num_patch_width

        aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        if aspect_ratio > current_aspect_ratio:
            new_height = int(
                round(original_height * (current_width / original_width), 7))
            padding = (current_height - new_height) // 2
            current_height = current_height - (2 * padding)
        else:
            new_width = int(
                round(original_width * (current_height / original_height), 7))
            padding = (current_width - new_width) // 2
            current_width = current_width - (2 * padding)

        unpadded_features = current_height * current_width
        newline_features = current_height

        return (unpadded_features, newline_features)



class RVLDummyInputsBuilder(
        LlavaDummyInputsBuilder[RVLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        image_token = '<image>'
        
        return image_token * num_images 

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()


        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
        }


class RVLMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_norm = torch.nn.LayerNorm(config.vision_config.hidden_size, eps=1e-06)
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = GELUActivation()
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_feature: torch.Tensor) -> torch.Tensor:
        image_feature = self.pre_norm(image_feature)
        hidden_states = self.linear_1(image_feature)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states



@MULTIMODAL_REGISTRY.register_processor(
    LlavaNextMultiModalProcessor,
    info=RVLProcessingInfo,
    dummy_inputs=RVLDummyInputsBuilder)
class RForConditionalGeneration(LlavaOnevisionForConditionalGeneration):

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "model.image_newline": "image_newline",
            "lm_head.": "language_model.lm_head.",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self.multi_modal_projector = RVLMultiModalProjector(config)
 