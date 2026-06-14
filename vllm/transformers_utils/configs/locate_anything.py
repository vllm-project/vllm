# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://huggingface.co/nvidia/LocateAnything-3B/blob/main/configuration_locateanything.py
from transformers import Qwen2Config
from transformers.configuration_utils import PretrainedConfig

from vllm.transformers_utils.configs.moonvit import MoonViTConfig


class LocateAnythingConfig(PretrainedConfig):
    model_type = "locateanything"

    def __init__(
        self,
        vision_config: dict | MoonViTConfig | None = None,
        text_config: dict | Qwen2Config | None = None,
        image_token_index: int = 151665,
        box_start_token_id: int = 151668,
        box_end_token_id: int = 151669,
        ref_start_token_id: int = 151672,
        ref_end_token_id: int = 151673,
        coord_start_token_id: int = 151677,
        coord_end_token_id: int = 152677,
        none_token_id: int = 4064,
        mlp_connector_layers: int = 2,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = MoonViTConfig()
        elif isinstance(vision_config, dict):
            vision_config = MoonViTConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = Qwen2Config()
        elif isinstance(text_config, dict):
            text_config = Qwen2Config(**text_config)
        self.text_config = text_config

        self.image_token_index = image_token_index
        self.box_start_token_id = box_start_token_id
        self.box_end_token_id = box_end_token_id
        self.ref_start_token_id = ref_start_token_id
        self.ref_end_token_id = ref_end_token_id
        self.coord_start_token_id = coord_start_token_id
        self.coord_end_token_id = coord_end_token_id
        self.none_token_id = none_token_id
        self.mlp_connector_layers = mlp_connector_layers

        super().__init__(**kwargs)
