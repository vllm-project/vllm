# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers.configuration_utils import PretrainedConfig

try:
    from transformers.modeling_rope_utils import rope_config_validation
except ImportError:
    rope_config_validation = None


class FlexOlmoConfig(PretrainedConfig):
    model_type = "flex_olmo"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=100352,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=100277,
        bos_token_id=None,
        eos_token_id=100257,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_scaling=None,
        rope_parameters: dict[str, Any] | None = None,
        attention_bias=False,
        attention_dropout=0.0,
        num_experts_per_tok=5,
        num_experts=7,
        output_router_logits=False,
        router_aux_loss_coef=0.01,
        norm_topk_prob=False,
        pre_norm=False,
        head_dim=None,
        use_head_qk_norm=False,
        use_grouped_gemm=True,
        moe_implementation="sonic",
        **kwargs,
    ):
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["FlexOlmoForCausalLM"]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = (
            head_dim
            if head_dim is not None
            else (hidden_size // num_attention_heads)
        )
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # Build rope_parameters for vLLM get_rope() from rope_scaling/rope_theta (HF style) or explicit rope_parameters
        if rope_parameters is not None:
            self.rope_parameters = dict(rope_parameters)
        elif rope_scaling is not None:
            self.rope_parameters = dict(rope_scaling)
        else:
            self.rope_parameters = {"rope_type": "default"}
        if "rope_theta" not in self.rope_parameters:
            self.rope_parameters["rope_theta"] = rope_theta
        # BC: if there is a 'type' field, move it to 'rope_type'
        if "type" in self.rope_parameters:
            self.rope_parameters["rope_type"] = self.rope_parameters["type"]
        if rope_config_validation is not None:
            rope_config_validation(self)
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.norm_topk_prob = norm_topk_prob
        self.pre_norm = pre_norm
        self.use_head_qk_norm = use_head_qk_norm
        self.use_grouped_gemm = use_grouped_gemm
        self.moe_implementation = moe_implementation
