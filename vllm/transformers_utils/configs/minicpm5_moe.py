# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2025-2026 SGLang Team

# Adapted for vLLM from the SGLang MiniCPM5 configuration.
"""Native config for the MiniCPM5 MoE model (model_type="minicpm5_moe")."""

from transformers import PreTrainedConfig


class MiniCPM5MoEConfig(PreTrainedConfig):
    model_type = "minicpm5_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Defaults follow the MiniCPM5-15B-A2.5B checkpoint config.
    def __init__(
        self,
        vocab_size: int = 130560,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 2,
        head_dim: int | None = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 524288,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = 1,
        bos_token_id: int | None = 0,
        eos_token_id: int | list[int] | tuple[int, ...] | None = (1, 130073),
        tie_word_embeddings: bool = False,
        rope_theta: float = 5000000.0,
        rope_parameters: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        first_k_dense_replace: int = 1,
        n_routed_experts: int = 160,
        n_shared_experts: int = 1,
        routed_scaling_factor: float = 3.66,
        topk_method: str = "greedy",
        num_experts_per_tok: int | None = 16,
        moe_intermediate_size: int = 512,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        use_gated_attention: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = dict(rope_parameters or {})
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.first_k_dense_replace = first_k_dense_replace
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.use_gated_attention = use_gated_attention

        if isinstance(eos_token_id, tuple):
            eos_token_id = list(eos_token_id)

        # Resolve precedence before Transformers normalizes RoPE in place.
        legacy_rope = kwargs.pop("rope_scaling", None)
        if rope_parameters is None and legacy_rope is not None:
            kwargs["rope_scaling"] = dict(legacy_rope)

        super().__init__(
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.rope_parameters = dict(self.rope_parameters)
        self.rope_parameters.pop("type", None)
        self.rope_theta = self.rope_parameters["rope_theta"]
