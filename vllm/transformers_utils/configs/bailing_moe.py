# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/inclusionAI/Ling/blob/master/models/configuration_bailing_moe.py
from transformers.configuration_utils import PretrainedConfig


class BailingMoeConfig(PretrainedConfig):
    model_type = "bailing_moe"

    def __init__(
        self,
        vocab_size=30592,
        hidden_size=1024,
        intermediate_size=None,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=0,
        hidden_act="silu",
        use_qkv_bias=False,  # bailing only
        use_bias=True,  # bailing only
        rms_norm_eps=1e-05,
        norm_head=False,  # bailing only
        tie_word_embeddings=False,  # PretrainedConfig key, 
                                    # here change default value.
        embedding_dropout=0.1,
        attention_dropout=0.1,
        output_dropout=0.1,
        initializer_range=0.02,
        max_position_embeddings=16384,
        rope_theta=10000.0,
        use_cache=True,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        rope_scaling=None,
        pad_token_id=126081,
        num_experts=16,
        num_shared_experts=0,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        moe_intermediate_size=None,
        first_k_dense_replace=0,
        head_dim=None,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.norm_head = norm_head
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.head_dim = (
            head_dim 
            if head_dim is not None 
            else self.hidden_size // self.num_attention_heads
        )
        self.rope_scaling = rope_scaling

        # MoE configs
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace

        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)