# adapted from https://huggingface.co/nvidia/NVLM-D-72B/blob/main/modeling_nvlm_d.py
# --------------------------------------------------------
# NVLM-D
# Copyright (c) 2024 NVIDIA
# Licensed under Apache 2.0 License [see LICENSE for details]
# --------------------------------------------------------
from typing import Optional

import torch.nn as nn
from transformers import PretrainedConfig

from vllm.inputs import INPUT_REGISTRY
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from .intern_vit import InternVisionModel
from .internvl import (InternVLChatModel, InternVLInputPipeline,
                       get_max_internvl_image_tokens)

IMG_START = '<|vision_start|>'
IMG_END = '<|vision_end|>'
IMG_CONTEXT = '<|vision_pad|>'


class NVLMInputPipeline(InternVLInputPipeline):

    def _create_image_prompt(self, feature_size: int, num_patches: int) -> str:
        tile_pos_identifiers = ([f"<tile_{i}>"
                                 for i in range(1, num_patches)] +
                                ["<tile_global_thumbnail>"])
        context_size = feature_size // num_patches

        return '<Image>' + ''.join(
            tile_pos_identifier + self.img_context_token * context_size
            for tile_pos_identifier in tile_pos_identifiers) + '</Image>'


input_pipeline = NVLMInputPipeline(IMG_START, IMG_END, IMG_CONTEXT)


@MULTIMODAL_REGISTRY.register_image_input_mapper(input_pipeline.input_mapper)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_internvl_image_tokens)
@INPUT_REGISTRY.register_dummy_data(input_pipeline.dummy_data)
@INPUT_REGISTRY.register_input_processor(input_pipeline.input_processor)
class NVLM_D_Model(InternVLChatModel):

    def _init_mlp1(self, config: PretrainedConfig) -> nn.Sequential:
        vit_hidden_size = config.vision_config.hidden_size
        llm_intermediate_size = config.text_config.intermediate_size
        llm_hidden_size = config.text_config.hidden_size

        return nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio)**2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)**2,
                      llm_intermediate_size,
                      bias=False),
            nn.GELU(),
            nn.Linear(llm_intermediate_size, llm_hidden_size, bias=False),
        )

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        *,
        is_mono: bool,
        prefix: str,
    ):
        if not is_mono:
            vision_feature_layer = config.select_layer
            if vision_feature_layer < 0:
                num_hidden_layers = config.vision_config.num_hidden_layers \
                    + vision_feature_layer + 1
            else:
                num_hidden_layers = vision_feature_layer + 1

            # We added additional dummy heads to the original num of heads to
            # make the number of heads divisible by 8.
            return InternVisionModel(
                config.vision_config,
                quant_config=quant_config,
                num_hidden_layers_override=num_hidden_layers,
                num_dummy_heads=7,
                prefix=prefix,
            )
        else:
            msg = "Monolith mode is not applicable to NVLM_D"
            raise NotImplementedError(msg)
