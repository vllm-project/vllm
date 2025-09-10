# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional, Union

from transformers.configuration_utils import PretrainedConfig


class Step3VisionEncoderConfig(PretrainedConfig):
    model_type = "step3_vision_encoder"

    def __init__(
        self,
        hidden_size=1792,
        intermediate_size=3072,
        output_hidden_size=4096,
        num_hidden_layers=63,
        num_attention_heads=16,
        num_channels=3,
        image_size=728,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.output_hidden_size = output_hidden_size
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
    architectures = ["Step3TextForCausalLM"]

    def __init__(
        self,
        hidden_size: int = 7168,
        intermediate_size: int = 18432,
        num_attention_heads: int = 64,
        num_attention_groups: int = 1,
        num_hidden_layers: int = 61,
        max_seq_len: int = 65536,
        vocab_size: int = 128815,
        rms_norm_eps: float = 1e-5,
        moe_intermediate_size: int = 5120,
        moe_num_experts: int = 48,
        moe_top_k: int = 3,
        rope_theta: float = 500000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embedding: int = 65536,
        share_expert_dim: int = 5120,
        share_q_dim: int = 2048,
        head_dim: int = 256,
        norm_expert_weight: bool = False,
        moe_layers_enum: tuple[int,
                               ...] = (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                       15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                       25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                       35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                                       45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                                       55, 56, 57, 58, 59),
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
        self.max_position_embedding = max_position_embedding
        self.share_expert_dim = share_expert_dim
        self.share_q_dim = share_q_dim
        self.head_dim = head_dim
        self.norm_expert_weight = norm_expert_weight
        self.moe_layers_enum = moe_layers_enum

        super().__init__(**kwargs)


class Step3VLConfig(PretrainedConfig):
    model_type = "step3_vl"

    def __init__(
        self,
        vision_config: Optional[Union[dict, Step3VisionEncoderConfig]] = None,
        text_config: Optional[Union[dict, Step3TextConfig]] = None,
        understand_projector_stride: int = 1,
        projector_bias: bool = True,
        image_token_id: int = 128001,
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

        self.understand_projector_stride = understand_projector_stride
        self.projector_bias = projector_bias
        self.hidden_size = text_config.hidden_size
        self.image_token_id = image_token_id

        super().__init__(**kwargs)
