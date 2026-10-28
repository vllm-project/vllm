# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hugging Face configuration for Raon-1-mini."""

from transformers.configuration_utils import PreTrainedConfig


class Raon1MiniConfig(PreTrainedConfig):
    model_type = "raon1_mini"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_type="raon1_mini",
        vocab_size=259840,
        hidden_size=2560,
        head_dim=None,
        intermediate_size=12032,
        num_hidden_layers=25,
        num_attention_heads=40,
        num_key_value_heads=None,
        hidden_act="situ",
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_parameters=None,
        tie_word_embeddings=False,
        moe_intermediate_size: int | None = None,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        num_experts: int | None = None,
        num_experts_per_token: int | None = None,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        mla_use_nope: bool | None = False,
        mla_use_output_gate: bool | None = False,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: dict | None = None,
        attn_res_block_size: int | None = None,
        latent_moe_use_norm: bool = False,
        activation_situ_beta: float | None = None,
        activation_situ_linear_beta: float | None = None,
        max_position_embeddings: int = 8192,
        routed_expert_hidden_size: int | None = None,
        topk_method: str = "noaux_tc",
        raon_mhc_streams: int | None = None,
        raon_mhc_post_factor: float | None = None,
        raon_mhc_selector_eps: float | None = None,
        **kwargs,
    ):
        self.model_type = model_type
        for name, value, default in (
            ("mhc_streams", raon_mhc_streams, 4),
            ("mhc_post_factor", raon_mhc_post_factor, 2.0),
            ("mhc_selector_eps", raon_mhc_selector_eps, 1e-5),
        ):
            if name in kwargs:
                legacy = kwargs.pop(name)
                if value is not None and value != legacy:
                    raise ValueError(f"Conflicting raon_{name} and legacy {name}")
                value = legacy
            setattr(self, f"raon_{name}", default if value is None else value)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        # NoPE does not use these values; generic max-length validation does.
        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_parameters = rope_scaling or rope_parameters or {"rope_type": "default"}
        rope_theta = kwargs.pop("rope_theta", 10000.0)
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = rope_theta
        self.rope_parameters = rope_parameters

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.mla_use_output_gate = mla_use_output_gate
        # moe config
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_renormalize = moe_renormalize
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_router_activation_func = moe_router_activation_func
        assert self.moe_router_activation_func in ("softmax", "sigmoid")
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.num_nextn_predict_layers = num_nextn_predict_layers

        self.attn_res_block_size = attn_res_block_size
        self.latent_moe_use_norm = latent_moe_use_norm
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        self.max_position_embeddings = max_position_embeddings
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.topk_method = topk_method

        # Normalize release field names to the shared recurrent layer contract.
        rda_config = kwargs.pop("rda_config", None)
        if rda_config is not None:
            if linear_attn_config is not None:
                raise ValueError(
                    "Pass either rda_config or linear_attn_config, not both"
                )
            names = {
                "conv_kernel_size": "short_conv_kernel_size",
                "full_rank_gate": "use_full_rank_gate",
                "mla_layers": "full_attn_layers",
            }
            linear_attn_config = {
                names.get(key, key): value for key, value in rda_config.items()
            }
        if linear_attn_config is not None:
            linear_attn_config = dict(linear_attn_config)
            if "kda_layers" in linear_attn_config:
                legacy_layers = linear_attn_config.pop("kda_layers")
                if (
                    "rda_layers" in linear_attn_config
                    and linear_attn_config["rda_layers"] != legacy_layers
                ):
                    raise ValueError("Conflicting rda_layers and legacy kda_layers")
                linear_attn_config["rda_layers"] = legacy_layers
        if linear_attn_config is not None:
            assert linear_attn_config["rda_layers"] is not None
            assert linear_attn_config["full_attn_layers"] is not None
        self.linear_attn_config = linear_attn_config

        quant = kwargs.get("quantization_config")
        if quant is not None and "ignored_layers" in quant:
            kwargs["quantization_config"] = {
                **quant,
                "ignored_layers": [
                    name.replace(".block_sparse_moe.", ".moe.")
                    for name in quant["ignored_layers"]
                ],
            }

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self):
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope is True
        )

    @property
    def is_moe(self):
        return self.num_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return not (
            self.linear_attn_config is None
            or (
                isinstance(self.linear_attn_config, dict)
                and self.linear_attn_config["rda_layers"] is not None
                and len(self.linear_attn_config["rda_layers"]) == 0
            )
        )

    def is_rda_layer(self, layer_idx: int):
        # Checkpoint layer masks are 1-based.
        return (
            self.linear_attn_config is not None
            and (layer_idx + 1) in self.linear_attn_config["rda_layers"]
        )
