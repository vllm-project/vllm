# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
# Adapted from
# https://huggingface.co/inclusionAI/Ling-3.0-flash/blob/51fd444268df5267074ef0b289de0e1c4cb9b381/configuration_bailing_moe_v3.py
"""Bailing V3 vision-language model configuration."""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig


def _normalize_mrope_parameters(
    config: PretrainedConfig,
    mrope_section: list[int],
) -> None:
    """Add the top-level Bailing M-RoPE split to a text config."""
    rope_parameters = dict(getattr(config, "rope_parameters", None) or {})
    text_mrope_section = rope_parameters.get("mrope_section")
    if text_mrope_section is not None and list(text_mrope_section) != mrope_section:
        raise ValueError(
            "Bailing V3 VL has conflicting M-RoPE sections: "
            f"text={text_mrope_section}, top-level={mrope_section}"
        )

    if "rope_type" not in rope_parameters and "type" not in rope_parameters:
        rope_parameters["rope_type"] = "default"
    rope_parameters.setdefault("rope_theta", getattr(config, "rope_theta", 10_000))
    rope_parameters["mrope_section"] = list(mrope_section)
    config.rope_parameters = rope_parameters


def _build_layer_types(
    num_hidden_layers: int,
    layer_group_size: int,
) -> list[str]:
    if layer_group_size <= 0:
        raise ValueError(
            f"Bailing V3 VL layer_group_size must be positive, got {layer_group_size}"
        )

    grouped_layers = num_hidden_layers // layer_group_size * layer_group_size
    return [
        "linear_attention"
        if ((layer_idx + 1) % layer_group_size != 0 and layer_idx < grouped_layers)
        else "full_attention"
        for layer_idx in range(num_hidden_layers)
    ]


class BailingMoeV3TextConfig(PretrainedConfig):
    model_type = "bailing_hybrid"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"mrope_section"}

    def __init__(
        self,
        *,
        vocab_size: int = 157184,
        hidden_size: int = 2048,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 20,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        hidden_act: str = "silu",
        use_qkv_bias: bool = False,
        use_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = False,
        embedding_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        output_dropout: float = 0.0,
        initializer_range: float = 0.02,
        max_position_embeddings: int = 32768,
        rope_theta: float = 600000.0,
        rope_parameters: dict[str, Any] | None = None,
        rope_scaling: dict[str, Any] | None = None,
        partial_rotary_factor: float | None = None,
        mrope_section: list[int] | tuple[int, ...] | None = None,
        use_cache: bool = True,
        max_window_layers: int = 20,
        pad_token_id: int = 156892,
        eos_token_id: int = 156892,
        num_experts: int = 256,
        num_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        n_group: int = 8,
        topk_group: int = 4,
        moe_intermediate_size: int = 512,
        moe_shared_expert_intermediate_size: int = 512,
        first_k_dense_replace: int = 1,
        head_dim: int | None = 128,
        output_router_logits: bool = False,
        use_qk_norm: bool = True,
        num_nextn_predict_layers: int = 0,
        mtp_loss_scaling_factor: float = 0,
        moe_router_enable_expert_bias: bool = True,
        routed_scaling_factor: float = 1.0,
        layer_group_size: int = 5,
        layer_types: list[str] | None = None,
        layers_block_type: list[str] | None = None,
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = None,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128,
        rope_interleave: bool = True,
        score_function: str = "sigmoid",
        scoring_func: str = "sigmoid",
        seq_aux: bool = True,
        topk_method: str = "noaux_tc",
        router_dtype: str = "fp32",
        gated_attention_proj_granularity_type: str | None = None,
        no_kda_lora: bool = False,
        kda_safe_gate: bool = False,
        kda_lower_bound: float | None = None,
        short_conv_kernel_size: int = 4,
        architectures: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("model_type", None)
        if architectures is None:
            architectures = ["BailingMoeV3ForCausalLM"]

        if rope_parameters is not None:
            self.rope_parameters = dict(rope_parameters)
        elif rope_scaling is not None:
            self.rope_parameters = dict(rope_scaling)
        else:
            self.rope_parameters = {}
        if (
            "rope_type" not in self.rope_parameters
            and "type" not in self.rope_parameters
        ):
            self.rope_parameters["rope_type"] = "default"
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        if mrope_section is not None:
            _normalize_mrope_parameters(self, list(mrope_section))

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.use_cache = use_cache
        self.max_window_layers = max_window_layers
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.output_router_logits = output_router_logits
        self.use_qk_norm = use_qk_norm
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_loss_scaling_factor = mtp_loss_scaling_factor
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.layer_group_size = layer_group_size
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.rope_interleave = rope_interleave
        self.score_function = score_function
        self.scoring_func = scoring_func
        self.seq_aux = seq_aux
        self.topk_method = topk_method
        self.router_dtype = router_dtype
        self.gated_attention_proj_granularity_type = (
            gated_attention_proj_granularity_type
        )
        self.no_kda_lora = no_kda_lora
        self.kda_safe_gate = kda_safe_gate
        self.kda_lower_bound = kda_lower_bound
        self.short_conv_kernel_size = short_conv_kernel_size

        expected_layer_types = _build_layer_types(
            num_hidden_layers,
            layer_group_size,
        )
        if layers_block_type is not None:
            legacy_type_mapping = {
                "attention": "full_attention",
                "mamba": "linear_attention",
                "conv": "linear_attention",
            }
            normalized_block_types = [
                legacy_type_mapping.get(layer_type, layer_type)
                for layer_type in layers_block_type
            ]
            if layer_types is not None and layer_types != normalized_block_types:
                raise ValueError(
                    "Bailing V3 VL has conflicting layer_types and layers_block_type"
                )
            layer_types = normalized_block_types
        if layer_types is not None and layer_types != expected_layer_types:
            raise ValueError(
                "Bailing V3 VL layer_types must match the layer_group_size schedule"
            )
        self.layer_types = expected_layer_types

        super().__init__(
            architectures=architectures,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.validate_layer_type()


class BailingMoeV3VisionConfig(PretrainedConfig):
    model_type = "qwen3_moe_vit"
    base_config_key = "vision_config"

    def __init__(
        self,
        *,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 3584,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: list[int] | None = None,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.deepstack_visual_indexes = (
            [8, 16, 24]
            if deepstack_visual_indexes is None
            else deepstack_visual_indexes
        )
        self.initializer_range = initializer_range


class BailingMoeV3VLConfig(PretrainedConfig):
    model_type = "bailing_moe_v3_vl"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "text_config": BailingMoeV3TextConfig,
        "vision_config": BailingMoeV3VisionConfig,
    }

    def __init__(
        self,
        *,
        text_config: BailingMoeV3TextConfig | dict[str, Any] | None = None,
        vision_config: BailingMoeV3VisionConfig | dict[str, Any] | None = None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        tie_word_embeddings: bool = False,
        mrope_section: list[int] | tuple[int, ...] | None = None,
        architectures: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("model_type", None)
        if mrope_section is None:
            mrope_section = [12, 10, 10]
        self.mrope_section = list(mrope_section)

        if text_config is None:
            text_config = BailingMoeV3TextConfig()
        elif isinstance(text_config, dict):
            text_config = dict(text_config)
            nested_mrope_section = text_config.pop("mrope_section", None)
            if (
                nested_mrope_section is not None
                and list(nested_mrope_section) != self.mrope_section
            ):
                raise ValueError(
                    "Bailing V3 VL has conflicting M-RoPE sections: "
                    f"text={nested_mrope_section}, "
                    f"top-level={self.mrope_section}"
                )
            text_config = BailingMoeV3TextConfig(**text_config)
        self.text_config = text_config
        _normalize_mrope_parameters(self.text_config, self.mrope_section)

        if vision_config is None:
            vision_config = BailingMoeV3VisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = BailingMoeV3VisionConfig(**vision_config)
        self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        if architectures is None:
            architectures = ["BailingMoeV3VLForConditionalGeneration"]

        super().__init__(
            architectures=architectures,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = [
    "BailingMoeV3TextConfig",
    "BailingMoeV3VisionConfig",
    "BailingMoeV3VLConfig",
]
