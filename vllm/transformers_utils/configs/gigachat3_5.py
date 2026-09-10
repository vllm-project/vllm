# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GigaChat 3.5 model configuration."""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class GigaChat35Config(PretrainedConfig):
    model_type = "gigachat3_5"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 7168,
        intermediate_size: int = 18432,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 40,
        nextn_is_sparse: bool | None = None,
        num_nextn_predict_layers: int = 0,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = None,
        n_shared_experts: int = 1,
        n_routed_experts: int = 256,
        routed_scaling_factor: float = 2.5,
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = 1536,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128,
        n_group: int = 1,
        topk_group: int = 1,
        num_experts_per_tok: int = 8,
        first_k_dense_replace: int = 3,
        norm_topk_prob: bool = True,
        scoring_func: str = "sigmoid",
        aux_loss_alpha: float = 0.0001,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.006,
        rms_norm_eps: float = 1e-6,
        optimizer_eps: float = 1e-20,
        use_cache: bool = True,
        pad_token_id: int | None = 2,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = False,
        rope_theta: float = 100000.0,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        rope_interleave: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        attention_type: str = "LlamaLatentAttention",
        num_krope_heads: int = 1,
        norm_type: str = "ZeroCenteredGatedNorm",
        layernorm_type: str | None = "pre_post",
        layernorm_gating_weight: float = 2.0,
        gated_attention: bool = True,
        use_shared_expert_sigmoid: bool = False,
        use_mla_scaling_factor: bool = True,
        attention_distillation: bool = False,
        linear_attention_type: str | None = "Qwen3NextGatedDeltaNet",
        full_attention_layers: list[int] | None = None,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        linear_num_key_heads: int = 32,
        linear_num_value_heads: int = 64,
        linear_use_short_conv: bool = True,
        linear_gating_type: str = "gated_rmsnorm_sigmoid_zero_centered",
        linear_sigmoid_gate_scale: float = 2.0,
        linear_attn_o_norm_eps: float | None = None,
        linear_use_legacy_qkvz_layout: bool = False,
        swiglu_limit: float = 10.0,
        **kwargs,
    ) -> None:
        if layernorm_type is None:
            layernorm_type = "pre"
        if layernorm_type not in {"pre", "post", "pre_post"}:
            raise ValueError(
                "layernorm_type must be one of {'pre', 'post', 'pre_post'}, "
                f"got {layernorm_type!r}"
            )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.nextn_is_sparse = nextn_is_sparse
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.rope_interleave = rope_interleave

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.optimizer_eps = optimizer_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta

        rope_parameters = dict(rope_parameters or rope_scaling or {})
        if "rope_type" not in rope_parameters and "type" in rope_parameters:
            rope_parameters["rope_type"] = rope_parameters["type"]
        rope_parameters.setdefault("rope_type", "default")
        rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_scaling = rope_scaling
        self.rope_parameters = rope_parameters

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_type = attention_type
        self.num_krope_heads = num_krope_heads
        self.norm_type = norm_type
        self.layernorm_type = layernorm_type
        self.layernorm_gating_weight = layernorm_gating_weight
        self.gated_attention = gated_attention
        self.use_shared_expert_sigmoid = use_shared_expert_sigmoid
        self.use_mla_scaling_factor = use_mla_scaling_factor
        self.attention_distillation = attention_distillation

        self.linear_attention_type = linear_attention_type
        layer_types = kwargs.pop("layer_types", None)
        if self.linear_attention_type is not None:
            if full_attention_layers is None and layer_types is not None:
                self.layer_types = list(layer_types)
                self.full_attention_layers = [
                    idx
                    for idx, layer_type in enumerate(self.layer_types)
                    if layer_type == "full_attention"
                ]
            else:
                self.full_attention_layers = full_attention_layers or []
                self.layer_types = [
                    "full_attention"
                    if i in self.full_attention_layers
                    else "linear_attention"
                    for i in range(self.num_hidden_layers)
                ]
        else:
            self.full_attention_layers = list(range(self.num_hidden_layers))
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.linear_attention_type is None:
            kwargs.pop("full_attention_layers", None)

        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_use_short_conv = linear_use_short_conv
        self.linear_gating_type = linear_gating_type
        self.linear_sigmoid_gate_scale = linear_sigmoid_gate_scale
        self.linear_attn_o_norm_eps = linear_attn_o_norm_eps
        self.linear_use_legacy_qkvz_layout = linear_use_legacy_qkvz_layout
        self.swiglu_limit = swiglu_limit

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def _use_pre_layernorm(self) -> bool:
        return self.layernorm_type in {"pre", "pre_post"}

    @property
    def _use_post_layernorm(self) -> bool:
        return self.layernorm_type in {"post", "pre_post"}


__all__ = ["GigaChat35Config"]
