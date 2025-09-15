# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PretrainedConfig


class GptOssConfig(PretrainedConfig):
    model_type = "gpt_oss"

    def __init__(self,
                 num_hidden_layers: int = 36,
                 num_experts: int = 128,
                 num_experts_per_token: int = 4,
                 vocab_size: int = 201088,
                 hidden_size: int = 2880,
                 intermediate_size: int = 2880,
                 head_dim: int = 64,
                 num_attention_heads: int = 64,
                 num_key_value_heads: int = 8,
                 sliding_window: int = 128,
                 rope_theta: float = 150000.0,
                 rope_scaling_factor: float = 32.0,
                 rope_ntk_alpha: float = 1.0,
                 rope_ntk_beta: float = 32.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_ntk_alpha = rope_ntk_alpha
        self.rope_ntk_beta = rope_ntk_beta
