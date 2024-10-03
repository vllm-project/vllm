# adapted from https://huggingface.co/nvidia/NVLM-D-72B/blob/main/modeling_nvlm_d.py
# --------------------------------------------------------
# NVLM-D
# Copyright (c) 2024 NVIDIA
# Licensed under Apache 2.0 License [see LICENSE for details]
# --------------------------------------------------------
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.inputs import INPUT_REGISTRY
from vllm.multimodal import MULTIMODAL_REGISTRY


from .internvl import (InternVLChatModel, dummy_data_for_internvl,
                       input_mapper_for_internvl, input_processor_for_internvl,
                       get_max_internvl_image_tokens)


@MULTIMODAL_REGISTRY.register_image_input_mapper(input_mapper_for_internvl)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_internvl_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_internvl)
@INPUT_REGISTRY.register_input_processor(input_processor_for_internvl)
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
