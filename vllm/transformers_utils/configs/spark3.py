# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers.configuration_utils import PretrainedConfig


class Spark3Config(PretrainedConfig):
    model_type = "spark3"

    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6656,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        headwise_attn_output_gate: bool = True,
        sliding_window: int = 512,
        vocab_size: int = 131072,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 8192,
        rope_parameters: dict[str, Any] | None = None,
        layer_types: list[str] | None = None,
        tie_word_embeddings: bool | None = False,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.sliding_window = sliding_window
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings

        if layer_types is not None:
            layer_types = layer_types[: self.num_hidden_layers]
        else:
            layer_types = [
                "sliding_attention" if bool((i + 1) % 4) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        self.layer_types = layer_types

        if rope_parameters is not None:
            self.rope_parameters = rope_parameters
        else:
            self.rope_parameters = {
                "full_attention": {
                    "rope_theta": 5000000,
                    "partial_rotary_factor": 0.25,
                },
                "sliding_attention": {
                    "rope_theta": 10000,
                    "partial_rotary_factor": 1.0,
                },
            }

        super().__init__(**kwargs)
