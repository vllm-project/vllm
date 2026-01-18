# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from transformers.configuration_utils import PretrainedConfig, layer_type_validation


class Olmo3_5HybridConfig(PretrainedConfig):
    model_type = "olmo3_5_hybrid"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_rep",
        "layers.*.self_attn.k_proj": "colwise_rep",
        "layers.*.self_attn.v_proj": "colwise_rep",
        "layers.*.self_attn.o_proj": "rowwise_rep",
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
        vocab_size: int | None = 100352,
        hidden_size: int | None = 3840,
        intermediate_size: int | None = 11008,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 30,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 65536,
        initializer_range: float | None = 0.02,
        use_cache: bool | None = True,
        pad_token_id: int | None = 100277,
        bos_token_id: int | None = None,
        eos_token_id: int | None = 100257,
        tie_word_embeddings: bool | None = False,
        rope_parameters=None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        rms_norm_eps: float | None = 1e-06,
        sliding_window: int | None = 4096,
        layer_types: list[str] | None = None,
        fla_hybrid_attention_indices: list[int] | None = None,
        linear_num_key_heads: int | None = None,
        linear_num_value_heads: int | None = None,
        linear_key_head_dim: int | None = None,
        linear_value_head_dim: int | None = None,
        linear_conv_kernel_dim: int = 4,
        linear_use_gate: bool = True,
        linear_allow_neg_eigval: bool = True,
        **kwargs,
    ):
        kwargs["architectures"] = ["Olmo3_5HybridForCausalLM"]

        assert num_hidden_layers is not None, "num_hidden_layers must be provided"

        if layer_types is None:
            if fla_hybrid_attention_indices is None:
                fla_hybrid_attention_indices = [
                    i for i in range(int(num_hidden_layers)) if i % 4 == 3
                ]

            layer_types = ["linear_attention"] * int(num_hidden_layers)
            for idx in fla_hybrid_attention_indices:
                if idx < 0 or idx >= int(num_hidden_layers):
                    raise ValueError(
                        f"`fla_hybrid_attention_indices` contains "
                        f"an out-of-range layer index {idx} "
                        f"for num_hidden_layers={num_hidden_layers}."
                    )
                layer_types[idx] = "full_attention"

        if len(layer_types) != int(num_hidden_layers):
            raise ValueError(
                f"`layer_types` must have length "
                f"num_hidden_layers={num_hidden_layers}, got {len(layer_types)}."
            )

        if "linear_attention" not in layer_types:
            raise ValueError(
                "OLMo3.5 Hybrid expects at least one 'linear_attention' layer."
            )
        if all(t == "linear_attention" for t in layer_types):
            raise ValueError(
                "OLMo3.5 Hybrid expects at least one attention layer (full or sliding)."
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
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window

        self.layer_types = list(layer_types)
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if (i + 1) % 4 != 0 else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.rope_parameters = rope_parameters

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.fla_hybrid_attention_indices = [
            i
            for i, t in enumerate(self.layer_types)
            if t in {"full_attention", "sliding_attention"}
        ]

        assert num_attention_heads is not None
        assert hidden_size is not None

        if linear_num_key_heads is None:
            linear_num_key_heads = int(num_attention_heads)
        if linear_num_value_heads is None:
            linear_num_value_heads = int(num_attention_heads)
        if linear_key_head_dim is None:
            linear_key_head_dim = int(
                0.75 * int(hidden_size) / int(linear_num_key_heads)
            )
        if linear_value_head_dim is None:
            linear_value_head_dim = int(2 * int(linear_key_head_dim))

        self.linear_num_key_heads = int(linear_num_key_heads)
        self.linear_num_value_heads = int(linear_num_value_heads)
        self.linear_key_head_dim = int(linear_key_head_dim)
        self.linear_value_head_dim = int(linear_value_head_dim)
        self.linear_conv_kernel_dim = int(linear_conv_kernel_dim)
        self.linear_use_gate = bool(linear_use_gate)
        self.linear_allow_neg_eigval = bool(linear_allow_neg_eigval)
