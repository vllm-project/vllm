# SPDX-License-Identifier: Apache-2.0
# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/configuration_kimi_vl.py
from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig

from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekV2Config
from vllm.transformers_utils.configs.moonvit import MoonViTConfig


class KimiVLConfig(PretrainedConfig):
    model_type = "kimi_vl"

    def __init__(self,
                 vision_config: Optional[Union[dict, MoonViTConfig]] = None,
                 text_config: Optional[Union[dict, DeepseekV2Config]] = None,
                 ignore_index: int = -100,
                 media_placeholder_token_id: int = 163605,
                 pad_token_id: int = 0,
                 **kwargs):
        if vision_config is None:
            vision_config = MoonViTConfig()
        elif isinstance(vision_config, dict):
            vision_config = MoonViTConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = DeepseekV2Config()
        elif isinstance(text_config, dict):
            text_config = DeepseekV2Config(**text_config)
        self.text_config = text_config

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id

        super().__init__(pad_token_id=pad_token_id, **kwargs)
