# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 XLANG Lab, The University of Hong Kong

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2 import Qwen2Config
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)


class OpenCUAConfig(PretrainedConfig):
    """OpenCUA-7B model configuration.
    
    OpenCUA is based on Qwen2.5-VL but uses 1D-RoPE instead of M-RoPE
    for the vision encoder.
    """

    model_type = "opencua"

    def __init__(
        self,
        vision_config: dict | Qwen2_5_VLVisionConfig | None = None,
        text_config: dict | Qwen2Config | None = None,
        ignore_index: int = -100,
        image_token_id: int = 151664,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151647,
        pad_token_id: int = 0,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = Qwen2_5_VLVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = Qwen2_5_VLVisionConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = Qwen2Config()
        elif isinstance(text_config, dict):
            text_config = Qwen2Config(**text_config)
        self.text_config = text_config

        self.ignore_index = ignore_index
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id

        super().__init__(pad_token_id=pad_token_id, **kwargs)

