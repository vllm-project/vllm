# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers.configuration_utils import PretrainedConfig


class ZayaConfig(PretrainedConfig):
    model_type = "zaya"
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"hybrid", "hybrid_sliding"}

    def __init__(
        self,
        use_cache=True,
        attention_bias=False,
        lm_head_bias=False,
        vocab_size=262272,
        hidden_size=2048,
        num_hidden_layers=40,
        num_experts=16,
        num_attention_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=106,
        tie_word_embeddings=True,
        attention_dropout=0.0,
        moe_intermediate_size=2048,
        num_experts_per_tok=1,
        output_router_logits=False,
        layer_types=None,
        sliding_window=None,
        rope_parameters=None,
        rope_scaling=None,
        partial_rotary_factor=0.5,
        num_key_value_heads=2,
        cca_time0=2,
        cca_time1=2,
        _attn_implementation="eager",
        **kwargs,
    ):
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.lm_head_bias = lm_head_bias
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_experts = num_experts
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        assert self.head_dim is not None
        self.num_key_value_heads = num_key_value_heads
        self.num_query_groups = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_dropout = attention_dropout
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.output_router_logits = output_router_logits
        self.layer_types = (
            ["hybrid"] * num_hidden_layers if layer_types is None else list(layer_types)
        )
        self.sliding_window = sliding_window
        self.partial_rotary_factor = partial_rotary_factor
        if isinstance(rope_parameters, dict):
            rope_parameters = dict(rope_parameters)
        elif isinstance(rope_scaling, dict):
            rope_parameters = dict(rope_scaling)
        else:
            rope_parameters = {
                "hybrid": {
                    "rope_type": "default",
                    "rope_theta": 5000000,
                    "partial_rotary_factor": partial_rotary_factor,
                },
                "hybrid_sliding": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": partial_rotary_factor,
                },
            }
        if "type" in rope_parameters:
            rope_parameters.setdefault("rope_type", rope_parameters.pop("type"))
        if "hybrid" in rope_parameters or "hybrid_sliding" in rope_parameters:
            rope_parameters.pop("rope_type", None)
        self.rope_parameters = rope_parameters
        self.cca_time0 = cca_time0
        self.cca_time1 = cca_time1
        self._attn_implementation = _attn_implementation
        self.rope_theta = self._rope_theta_for_layer_type("hybrid")

        # Compatibility aliases used by existing vLLM helper code.
        self.cca = True
        self.ffn_hidden_size = 2 * moe_intermediate_size
        self.activation_func = "swiglu"
        self.norm_epsilon = rms_norm_eps
        self.normalization = "RMSNorm"
        self.moe_router_topk = num_experts_per_tok
        self.zaya_mlp_expansion = router_hidden_size = kwargs.pop(
            "router_hidden_size", 256
        )
        self.router_hidden_size = router_hidden_size
        self.zaya_use_mod = True
        self.zaya_high_prec = True
        self.zaya_use_eda = True
        self.add_bias_linear = False
        self.gated_linear_unit = True
        self.scale_residual_merge = True
        self.residual_in_fp32 = True
        self.clamp_temp = False

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            **kwargs,
        )

    def _rope_theta_for_layer_type(self, layer_type: str) -> float:
        layer_rope = self.rope_parameters.get(layer_type, self.rope_parameters)
        if isinstance(layer_rope, dict):
            return layer_rope.get("rope_theta", 5000000)
        return 5000000
