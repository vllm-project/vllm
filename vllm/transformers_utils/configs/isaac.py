# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from transformers import Qwen3Config
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig


class PixelShuffleSiglip2VisionConfig(Siglip2VisionConfig):
    """Vision configuration for Isaac with Pixel Shuffle support.

    Extends Siglip2VisionConfig with additional fields for pixel shuffle.
    """

    model_type = "pixel_shuffle_siglip2"
    base_config_key = "vision_config"

    def __init__(
        self,
        pixel_shuffle_scale_factor: int = 1,
        num_patches: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Add our custom fields
        self.pixel_shuffle_scale_factor = pixel_shuffle_scale_factor
        self.num_patches = num_patches


class IsaacConfig(Qwen3Config):
    """Configuration class for Isaac multimodal model."""

    model_type = "isaac"
    sub_configs = {
        "vision_config": PixelShuffleSiglip2VisionConfig,
        "text_config": Qwen3Config,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        vision_patch_size: int = 16,
        vision_max_num_patches: int = 256,
        vision_min_num_patches: int | None = None,
        pixel_shuffle_scale: int = 1,
        max_sequence_length: int = 16384,
        vision_token: str = "<image>",
        vision_attn_implementation: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(text_config, dict):
            # from HF config
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init text config.
            self.text_config = self.sub_configs["text_config"](**kwargs)
        else:
            # from Qwen3Config
            self.text_config = text_config

        # EventStreamProcessor parameters (for backward compatibility)
        self.video_patch_size = vision_patch_size
        self.vision_max_num_patches = vision_max_num_patches
        self.vision_min_num_patches = vision_min_num_patches
        self.pixel_shuffle_scale = pixel_shuffle_scale

        # Processing parameters
        self.max_sequence_length = max_sequence_length
        self.vision_token = vision_token

        # Handle vision config - PixelShuffleSiglip2VisionConfig instance
        if isinstance(vision_config, dict):
            self.vision_config = PixelShuffleSiglip2VisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = PixelShuffleSiglip2VisionConfig()
        else:
            self.vision_config = vision_config

        # Ensure compatibility with pretrained checkpoints
        self.vision_config.pixel_shuffle_scale_factor = getattr(
            self.vision_config,
            "pixel_shuffle_scale_factor",
            pixel_shuffle_scale,
        )
        self.vision_config.num_patches = getattr(
            self.vision_config,
            "num_patches",
            vision_max_num_patches,
        )
        self.vision_attn_implementation = vision_attn_implementation


__all__ = [
    "IsaacConfig",
    "PixelShuffleSiglip2VisionConfig",
]
