# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers.configuration_utils import PretrainedConfig


class AfmoeConfig(PretrainedConfig):
    model_type = "afmoe"

    def __init__(
        self,
        vocab_size: int = 200_192,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        moe_intermediate_size: int = 1408,
        num_hidden_layers: int = 32,
        num_dense_layers: int = 1,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        num_experts: int = 64,
        num_experts_per_tok: int = 6,
        num_shared_experts: int = 2,
        num_expert_groups: int = 1,
        num_limited_groups: int = 1,
        score_func: str = "sigmoid",
        route_norm: bool = True,
        route_scale: float = 1.0,
        global_attn_every_n_layers: int = 4,
        sliding_window: int = 2048,
        layer_types: list[str] | None = None,
        attention_dropout: float = 0.0,
        mup_enabled: bool = False,
        n_group: int = 1,
        topk_group: int = 1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_dense_layers = num_dense_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.num_expert_groups = num_expert_groups
        self.num_limited_groups = num_limited_groups
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale

        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.attention_dropout = attention_dropout

        self.mup_enabled = mup_enabled
        self.n_group = n_group
        self.topk_group = topk_group

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["AfmoeConfig"]
