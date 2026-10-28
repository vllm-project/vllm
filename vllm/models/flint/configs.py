# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint model config."""

from typing import Any

from transformers.configuration_utils import PreTrainedConfig


class FlintConfig(PreTrainedConfig):
    """Decoder-only MoE transformer with hybrid sliding/full attention.

    Sliding layers use RoPE; full-attention layers listed in
    ``no_rope_layer_indices`` use no positional encoding. Every attention head
    has a learned sink logit. ``sliding_window`` counts the current token.
    """

    model_type = "flint"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 150144,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 49,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_parameters: dict[str, Any] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_sliding_window: bool = True,
        sliding_window: int | None = 513,
        layer_types: list[str] | None = None,
        no_rope_layer_indices: list[int] | None = None,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 1024,
        num_experts_per_tok: int = 6,
        num_experts: int = 256,
        norm_topk_prob: bool = True,
        router_score_function: str = "sigmoid",
        mlp_only_layers: list[int] | None = None,
        **kwargs,
    ):
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
        self.rope_parameters = rope_parameters or {
            "rope_type": "default",
            "rope_theta": rope_theta,
        }
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.layer_types = layer_types or ["full_attention"] * num_hidden_layers
        if len(self.layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_hidden_layers is {num_hidden_layers}"
            )
        self.no_rope_layer_indices = no_rope_layer_indices or []
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.router_score_function = router_score_function
        self.mlp_only_layers = mlp_only_layers or []
        # The reference implementation computes the LM head in fp32.
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["FlintConfig"]
