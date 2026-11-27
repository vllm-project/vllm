# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import PreTrainedConfig


class GigaChat4Config(PreTrainedConfig):
    model_type = "gigachat4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 7168,
        intermediate_size: int = 18432,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 64,
        num_attention_heads: int = 64,
        n_shared_experts: int | None = None,
        n_routed_experts: int | None = None,
        num_experts_per_tok: int = 8,
        first_k_dense_replace: int = 3,
        routed_scaling_factor: float = 2.5,
        norm_topk_prob: bool = True,
        scoring_func: str = "sigmoid",
        n_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        max_position_embeddings: int = 1048576,
        rope_theta: float = 100000.0,
        rope_parameters: dict | None = None,
        rope_interleave: bool = True,
        rms_norm_eps: float = 1e-6,
        norm_type: str = "GatedNorm",
        layernorm_type: str = "pre_post",
        layernorm_gating_weight: float = 2.0,
        gated_attention: bool = True,
        attention_bias: bool = False,
        attention_type: str | None = None,
        use_mla_scaling_factor: bool = True,
        hidden_act: str = "silu",
        full_attention_layers: list[int] | None = None,
        layer_types: list[str] | None = None,
        linear_attention_type: str = "GigaChat4GatedDeltaNet",
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        linear_num_key_heads: int = 32,
        linear_num_value_heads: int = 64,
        linear_use_short_conv: bool = True,
        linear_gating_type: str = "gated_rmsnorm_sigmoid",
        linear_sigmoid_gate_scale: float = 2.0,
        swiglu_limit: float | None = 10.0,
        branch_multiplier: float = 1.0,
        linear_attn_output_mult: float = 1.0,
        final_logit_softcapping: float | None = 30.0,
        index_topk: int | None = None,
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        indexer_norm_type: str = "LlamaRMSNorm",
        indexer_rotate_activation_enabled: bool | None = None,
        dense_attention_layers: list[int] | None = None,
        use_shared_expert_sigmoid: bool = False,
        num_nextn_predict_layers: int = 1,
        nextn_is_sparse: bool = False,
        nextn_is_linear: bool = False,
        tie_word_embeddings: bool = False,
        dflash_config: dict | None = None,
        **kwargs,
    ) -> None:
        is_draft = dflash_config is not None
        self.branch_multiplier = float(branch_multiplier)
        self.linear_attn_output_mult = float(linear_attn_output_mult)
        if dflash_config is not None and (
            kwargs.get("is_causal") or dflash_config.get("causal")
        ):
            raise ValueError("GigaChat4 DFlash2 requires non-causal attention")
        if n_shared_experts is None:
            n_shared_experts = 0 if is_draft else 1
        if n_routed_experts is None:
            n_routed_experts = 0 if is_draft else 256
        if is_draft and (n_shared_experts or n_routed_experts):
            raise ValueError("GigaChat4 DFlash2 requires dense draft layers")
        if norm_type != "GatedNorm" or layernorm_type != "pre_post":
            raise ValueError("GigaChat4 requires GatedNorm with pre_post normalization")
        if hidden_act != "silu" or scoring_func != "sigmoid":
            raise ValueError("GigaChat4 requires SiLU activation and sigmoid routing")
        if attention_bias:
            raise ValueError("GigaChat4 attention bias is unsupported")
        if (
            linear_attention_type != "GigaChat4GatedDeltaNet"
            or linear_gating_type != "gated_rmsnorm_sigmoid"
            or kwargs.get("linear_use_legacy_qkvz_layout", False)
            or not linear_use_short_conv
        ):
            raise ValueError("Unsupported GigaChat4 linear attention configuration")
        if dense_attention_layers or use_shared_expert_sigmoid:
            raise ValueError("Dense attention and shared expert gating are unsupported")
        if indexer_rotate_activation_enabled:
            raise ValueError("GigaChat4 indexer activation rotation is unsupported")
        if q_lora_rank is None or q_lora_rank <= 0:
            raise ValueError("GigaChat4 requires a positive q_lora_rank")
        if index_topk is not None and index_topk < 0:
            raise ValueError("index_topk must be nonnegative")
        expected_attention = (
            "LlamaLatentSparseAttention" if index_topk else "LlamaLatentAttention"
        )
        if attention_type is not None and attention_type != expected_attention:
            raise ValueError(
                "attention_type and index_topk must select the same MLA variant"
            )
        if index_topk and (
            index_n_heads is None
            or index_n_heads <= 0
            or index_head_dim != 128
            or qk_rope_head_dim != 64
            or indexer_norm_type != "LlamaRMSNorm"
        ):
            raise ValueError("Unsupported GigaChat4 sparse indexer configuration")

        if layer_types is None:
            if full_attention_layers is None:
                full_attention_layers = (
                    list(range(num_hidden_layers))
                    if is_draft
                    else list(range(3, num_hidden_layers, 4))
                )
            layer_types = [
                "full_attention" if i in full_attention_layers else "linear_attention"
                for i in range(num_hidden_layers)
            ]
        if len(layer_types) != num_hidden_layers or any(
            kind not in ("full_attention", "linear_attention") for kind in layer_types
        ):
            raise ValueError("layer_types must describe every GigaChat4 decoder layer")
        full_layers = [
            i for i, kind in enumerate(layer_types) if kind == "full_attention"
        ]
        if is_draft and len(full_layers) != num_hidden_layers:
            raise ValueError("GigaChat4 DFlash2 requires full attention in every layer")
        if (
            full_attention_layers is not None
            and sorted(full_attention_layers) != full_layers
        ):
            raise ValueError("full_attention_layers and layer_types disagree")

        self.vocab_size = vocab_size
        self.dflash_config = dflash_config
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.n_group = n_group
        self.topk_group = topk_group
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_parameters = dict(rope_parameters or {})
        self.rope_interleave = rope_interleave
        self.rms_norm_eps = rms_norm_eps
        self.norm_type = norm_type
        self.layernorm_type = layernorm_type
        self.layernorm_gating_weight = layernorm_gating_weight
        self.gated_attention = gated_attention
        self.attention_bias = attention_bias
        self.attention_type = expected_attention
        self.use_mla_scaling_factor = use_mla_scaling_factor
        self.hidden_act = hidden_act
        self.full_attention_layers = full_layers
        self.layer_types = list(layer_types)
        self.linear_attention_type = linear_attention_type
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_use_short_conv = linear_use_short_conv
        self.linear_gating_type = linear_gating_type
        self.linear_sigmoid_gate_scale = linear_sigmoid_gate_scale
        self.swiglu_limit = swiglu_limit
        self.final_logit_softcapping = final_logit_softcapping
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.indexer_norm_type = indexer_norm_type
        self.indexer_rotate_activation_enabled = indexer_rotate_activation_enabled
        self.dense_attention_layers = dense_attention_layers or []
        self.use_shared_expert_sigmoid = use_shared_expert_sigmoid
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.nextn_is_sparse = nextn_is_sparse
        self.nextn_is_linear = nextn_is_linear
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def use_dsa(self) -> bool:
        return bool(self.index_topk)
