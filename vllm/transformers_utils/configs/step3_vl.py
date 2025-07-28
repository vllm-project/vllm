# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional, Union

from transformers.configuration_utils import PretrainedConfig


class Step3VisionEncoderConfig(PretrainedConfig):
    model_type = "step3_vision_encoder"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=728,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        super().__init__(**kwargs)


class Step3TextConfig(PretrainedConfig):
    model_type = "step3_text"

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 13312,
        num_attention_heads: int = 40,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 48,
        max_seq_len: int = 4096,
        vocab_size: int = 65536,
        rms_norm_eps: float = 1e-5,
        moe_intermediate_size: int = 10240,
        moe_num_experts: int = 16,
        moe_top_k: int = 4,
        rope_theta: float = 500000,
        rope_scaling: Optional[dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embedding: int = 16384,
        share_expert_dim: Optional[int] = None,
        share_q_dim: Optional[int] = None,
        norm_expert_weight: bool = True,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim
        self.max_position_embedding = max_position_embedding
        if share_expert_dim is None:
            self.share_expert_dim = self.moe_intermediate_size * self.moe_top_k
        else:
            self.share_expert_dim = share_expert_dim
        self.share_q_dim = share_q_dim
        self.norm_expert_weight = norm_expert_weight

        super().__init__(**kwargs)


class Step3VLConfig(PretrainedConfig):
    model_type = "step3_vl"

    def __init__(
        self,
        vision_config: Optional[Union[dict, Step3VisionEncoderConfig]] = None,
        text_config: Optional[Union[dict, Step3TextConfig]] = None,
        image_token_id: int = 128001,
        understand_projector_stride: int = 1,
        projector_bias: bool = True,
        **kwargs,
    ) -> None:
        if vision_config is None:
            vision_config = Step3VisionEncoderConfig()
        elif isinstance(vision_config, dict):
            vision_config = Step3VisionEncoderConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = Step3TextConfig()
        elif isinstance(text_config, dict):
            text_config = Step3TextConfig(**text_config)
        self.text_config = text_config

        self.image_token_id = image_token_id
        self.understand_projector_stride = understand_projector_stride
        self.projector_bias = projector_bias

        super().__init__(**kwargs)
