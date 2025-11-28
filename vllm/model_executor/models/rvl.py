# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping

import torch
import torch.nn as nn
from transformers.activations import GELUActivation

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict

from .llava_next import (
    LlavaDummyInputsBuilder,
    LlavaNextMultiModalProcessor,
    LlavaNextProcessingInfo,
)
from .llava_onevision import LlavaOnevisionForConditionalGeneration
from .utils import WeightsMapper


class RVLProcessingInfo(LlavaNextProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(**kwargs)


class RVLDummyInputsBuilder(LlavaDummyInputsBuilder[RVLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        image_token = "<image>"

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()

        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
        }


class RVLMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=1e-06)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=True,
        )
        self.act = GELUActivation()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=True,
        )

    def forward(self, image_feature: torch.Tensor) -> torch.Tensor:
        image_feature = self.pre_norm(image_feature)
        hidden_states = self.linear_1(image_feature)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    LlavaNextMultiModalProcessor,
    info=RVLProcessingInfo,
    dummy_inputs=RVLDummyInputsBuilder,
)
class RForConditionalGeneration(LlavaOnevisionForConditionalGeneration):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers
            # v4.52
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "model.image_newline": "image_newline",
            "lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self.multi_modal_projector = RVLMultiModalProjector(config)
