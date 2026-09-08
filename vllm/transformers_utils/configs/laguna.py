# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers.configuration_utils import PretrainedConfig


class LagunaConfig(PretrainedConfig):
    model_type = "laguna"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.g_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int = 100352,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        qkv_bias: bool = False,
        attention_bias: bool = False,
        gating: bool | str = True,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 500000.0,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        partial_rotary_factor: float = 1.0,
        attention_dropout: float = 0.0,
        sliding_window: int | None = None,
        layer_types: list[str] | None = None,
        swa_attention_sink_enabled: bool = False,
        swa_rope_parameters: dict | None = None,
        num_attention_heads_per_layer: list[int] | None = None,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        norm_topk_prob: bool = True,
        decoder_sparse_step: int = 1,
        mlp_only_layers: list[int] | None = None,
        router_aux_loss_coef: float = 0.001,
        output_router_logits: bool = False,
        moe_routed_scaling_factor: float = 1.0,
        moe_apply_router_weight_on_input: bool = False,
        **kwargs,
    ):
        if mlp_only_layers is None:
            mlp_only_layers = [0]

        # Accept either v4-style (rope_theta + rope_scaling) or v5-style
        # (rope_parameters). Translate v5 → v4 so downstream code has one path.
        if rope_parameters is not None:
            rp = dict(rope_parameters)
            rope_theta = float(rp.pop("rope_theta", rope_theta))
            rt = rp.pop("rope_type", None)
            if rt is not None and rt != "default":
                rope_scaling = {"rope_type": rt, **rp}
            elif rp and rope_scaling is None:
                rope_scaling = {"rope_type": "default", **rp}

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qkv_bias = qkv_bias
        self.attention_bias = attention_bias
        self.gating = gating
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.swa_attention_sink_enabled = swa_attention_sink_enabled
        self.swa_rope_parameters = swa_rope_parameters
        self.num_attention_heads_per_layer = num_attention_heads_per_layer
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = mlp_only_layers
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        self.moe_routed_scaling_factor = moe_routed_scaling_factor
        self.moe_apply_router_weight_on_input = moe_apply_router_weight_on_input

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["LagunaConfig"]
