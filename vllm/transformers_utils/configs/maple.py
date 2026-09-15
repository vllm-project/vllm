# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers.configuration_utils import PretrainedConfig


class MapleConfig(PretrainedConfig):
    """Config for DeepGrove Maple (e.g. `deepgrove/maple-preview`).

    Defaults mirror the `maple-preview` checkpoint: a 24-layer 256-expert MoE
    with a 3:1 ratio of sliding-window to global attention layers, per-head
    QK norm, half-width RoPE on the sliding layers and NoPE on the global ones.
    """

    model_type = "maple"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        hidden_act: str = "silu",
        use_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        partial_rotary_factor: float = 0.5,
        sliding_window: int = 512,
        layer_types: list[str] | None = None,
        nope_on_global_attention: bool = True,
        use_qk_norm: bool = True,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        num_shared_experts: int = 0,
        moe_intermediate_size: int = 512,
        norm_topk_prob: bool = True,
        moe_router_enable_expert_bias: bool = False,
        # The reference MapleMLP hardcodes the SwiGLU clamp bound; expose it so
        # future checkpoints can retune it without a new config class.
        swiglu_limit: float = 7.0,
        output_router_logits: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.use_bias = use_bias
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.sliding_window = sliding_window
        self.layer_types = layer_types or [
            "full_attention" if (i + 1) % 4 == 0 else "sliding_attention"
            for i in range(num_hidden_layers)
        ]
        self.nope_on_global_attention = nope_on_global_attention
        self.use_qk_norm = use_qk_norm
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.swiglu_limit = swiglu_limit
        self.output_router_logits = output_router_logits

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["MapleConfig"]
