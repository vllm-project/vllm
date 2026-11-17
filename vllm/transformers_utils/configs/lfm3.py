# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers.configuration_utils import PreTrainedConfig


class Lfm3Config(PreTrainedConfig):
    model_type = "lfm3"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"rms_norm_eps": "norm_eps"}
    default_theta = 1_000_000.0
    _rope_type_labels = ("full_attention", "sliding_attention")

    def __init__(
        self,
        vocab_size: int = 128_000,
        hidden_size: int = 2048,
        intermediate_size: int = 7680,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 128_000,
        initializer_range: float = 0.02,
        norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = 124_893,
        bos_token_id: int | None = 124_894,
        eos_token_id: int | list[int] | None = 124_900,
        tie_word_embeddings: bool = True,
        rope_parameters: dict[str, Any] | None = None,
        conv_bias: bool = False,
        conv_L_cache: int = 7,
        layer_types: list[str] | None = None,
        head_dim: int = 128,
        sliding_window: int | None = 256,
        use_sandwich_norm: bool = True,
        use_attention_gate: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.conv_bias = conv_bias
        self.conv_L_cache = conv_L_cache
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.use_sandwich_norm = use_sandwich_norm
        self.use_attention_gate = use_attention_gate
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_types = layer_types or [
            "full_attention"
            if i % 4 == 2
            else "sliding_attention"
            if i % 4 == 0 and i > 0
            else "conv"
            for i in range(num_hidden_layers)
        ]
        if rope_parameters is None:
            rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.5,
                },
            }
        for params in rope_parameters.values():
            # The released hub configs say `default`, but their remote code spreads
            # the partial frequencies over the whole head, which is `proportional`.
            if (
                isinstance(params, dict)
                and params.get("rope_type", "default") == "default"
                and params.get("partial_rotary_factor", 1.0) < 1.0
            ):
                params["rope_type"] = "proportional"
        self.rope_parameters = rope_parameters
        # Legacy key the hub remote code still accepts
        tie_word_embeddings = kwargs.pop("tie_embedding", tie_word_embeddings)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
