# coding=utf-8
# Copyright 2025 MiniMax and The vLLM team.
# Licensed under the Apache License, Version 2.0
"""MiniMax model configuration"""

from transformers import PretrainedConfig


class MiniMaxConfig(PretrainedConfig):
    """Configuration class for MiniMax-M2 model.

    MiniMax-M2 is a Mixture-of-Experts (MoE) model with 230B total parameters
    and 10B active parameters, designed for coding and agentic tasks.
    """
    model_type = "minimax"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=200064,
        hidden_size=6400,
        intermediate_size=17920,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=100,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        rotary_dim=100,
        attention_bias=False,
        attention_dropout=0.0,
        # MoE parameters
        num_experts=32,
        num_experts_per_tok=2,
        num_shared_experts=0,
        expert_intermediate_size=2560,
        moe_layer_freq=1,  # MoE layer frequency
        first_k_dense_replace=0,  # First k layers use dense MLP
        norm_topk_prob=True,
        # Sliding window
        sliding_window=None,
        sliding_window_pattern=None,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rotary_dim = rotary_dim
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # MoE parameters
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.expert_intermediate_size = expert_intermediate_size
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        # Sliding window
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
