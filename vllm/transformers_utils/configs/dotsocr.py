# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2 import Qwen2Config


class DotsVisionConfig(PretrainedConfig):
    model_type: str = "dots_vit"

    def __init__(
        self,
        embed_dim: int = 1536,  # vision encoder embed size
        hidden_size: int = 1536,  # after merger hidden size
        intermediate_size: int = 4224,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        rms_norm_eps: float = 1e-5,
        use_bias: bool = False,
        attn_implementation="flash_attention_2",
        initializer_range=0.02,
        init_merger_std=0.02,
        is_causal=False,  # ve causal forward
        post_norm=True,
        gradient_checkpointing=False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.rms_norm_eps = rms_norm_eps
        self.use_bias = use_bias
        self.attn_implementation = attn_implementation
        self.initializer_range = initializer_range
        self.init_merger_std = init_merger_std
        self.is_causal = is_causal
        self.post_norm = post_norm
        self.gradient_checkpointing = gradient_checkpointing


class DotsOCRConfig(Qwen2Config):
    model_type = "dots_ocr"

    def __init__(
        self,
        image_token_id=151665,
        video_token_id=151656,
        vision_config: dict | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_config = DotsVisionConfig(**(vision_config or {}))

    def save_pretrained(self, save_directory, **kwargs):
        self._auto_class = None
        super().save_pretrained(save_directory, **kwargs)
