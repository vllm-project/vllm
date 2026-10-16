# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers.configuration_utils import PreTrainedConfig


class PerceptionEncoderConfig(PreTrainedConfig):
    """Configuration for the vision tower shared by Step3p7 and Step5."""

    model_type = "perception_encoder"

    def __init__(
        self,
        hidden_act: str = "quick_gelu",
        patch_size: int = 14,
        width: int = 1536,
        layers: int = 47,
        heads: int = 16,
        mlp_ratio: float = 8960 / 1536,
        output_dim: int | None = None,
        ls_init_value: float = 0.1,
        drop_path: float = 0.0,
        image_size: int = 728,
        use_abs_posemb: bool = True,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        pool_type: str = "none",
        attn_pooler_heads: int = 8,
        use_ln_pre: bool = True,
        use_ln_post: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.hidden_act = hidden_act
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.output_dim = output_dim
        self.ls_init_value = ls_init_value
        self.drop_path = drop_path
        self.image_size = image_size
        self.use_abs_posemb = use_abs_posemb
        self.use_cls_token = use_cls_token
        self.use_rope2d = use_rope2d
        self.pool_type = pool_type
        self.attn_pooler_heads = attn_pooler_heads
        self.use_ln_pre = use_ln_pre
        self.use_ln_post = use_ln_post


class Step5Config(PreTrainedConfig):
    model_type = "step5"

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 13312,
        num_attention_heads: int = 40,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 48,
        max_seq_len: int = 4096,
        vocab_size: int = 65536,
        tie_word_embeddings: bool = False,
        rms_norm_eps: float = 1e-5,
        use_moe: bool = False,
        moe_intermediate_size: int = 10240,
        moe_num_experts: int = 16,
        moe_top_k: int = 4,
        rope_theta: float | list[float] | None = 500000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        share_expert_dim: int | None = None,
        norm_dtype: str = "bf16",
        norm_expert_weight: bool = True,
        bos_token_id: list[int] | int | None = None,
        eos_token_id: list[int] | int | None = None,
        moe_router_activation: str = "sigmoid",
        moe_router_scaling_factor: float = 1.0,
        att_impl_type: str = "GQA",
        use_head_wise_attn_gate: bool = False,
        use_moe_router_bias: bool = False,
        need_fp32_gate: bool = False,
        layer_types: list[str] | None = None,
        use_rope_layers: list[bool] | None = None,
        partial_rotary_factors: list[float] | None = None,
        yarn_only_types: list[str] | None = None,
        attention_other_setting: dict[str, Any] | None = None,
        num_nextn_predict_layers: int = 0,
        swa_num_attention_heads: int | None = None,
        swiglu_limits: list[float] | None = None,
        swiglu_limits_shared: list[float] | None = None,
        zero_centered: bool = True,
        fp32_residual_connection: bool = False,
        max_position_embeddings: int | None = None,
        **kwargs,
    ) -> None:
        if "max_position_embedding" in kwargs:
            raise ValueError(
                "Found 'max_position_embedding' in config.json. "
                "Please use 'max_position_embeddings' (with 's') instead."
            )
        if num_nextn_predict_layers < 0:
            raise ValueError(
                "num_nextn_predict_layers must be non-negative, got "
                f"{num_nextn_predict_layers}."
            )
        if max_position_embeddings is not None and max_position_embeddings <= 0:
            raise ValueError(
                "max_position_embeddings must be positive, got "
                f"{max_position_embeddings}."
            )
        if max_position_embeddings is None and max_seq_len <= 0:
            raise ValueError(
                "max_seq_len must be positive when max_position_embeddings is "
                f"omitted, got {max_seq_len}."
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.use_moe = use_moe
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.num_experts_per_tok = moe_top_k
        self.moe_top_k = moe_top_k

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim
        self.share_expert_dim = (
            moe_intermediate_size * moe_top_k
            if share_expert_dim is None
            else share_expert_dim
        )
        self.norm_dtype = norm_dtype
        self.norm_expert_weight = norm_expert_weight

        self.max_position_embeddings = (
            max_seq_len if max_position_embeddings is None else max_position_embeddings
        )
        self.moe_router_activation = moe_router_activation
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.use_moe_router_bias = use_moe_router_bias
        self.need_fp32_gate = need_fp32_gate

        self.att_impl_type = att_impl_type
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        # Step's per-layer arrays cover the dense stack plus the appended MTP
        # layers, which use absolute layer indices.
        # transformers 5.x validates len(layer_types) == num_hidden_layers, so keep
        # the MTP-inclusive copy under a Step-owned name and expose only the dense
        # slice as `layer_types`, which is what transformers and vLLM's
        # hybrid-attention detection both mean by it.
        total_num_layers = num_hidden_layers + num_nextn_predict_layers
        serialized_layer_types = kwargs.pop("layer_types_with_mtp", None)
        if serialized_layer_types is not None:
            if layer_types and list(layer_types) != list(
                serialized_layer_types[:num_hidden_layers]
            ):
                raise ValueError(
                    "layer_types must match the dense prefix of layer_types_with_mtp."
                )
            layer_types = serialized_layer_types
        if layer_types and len(layer_types) < total_num_layers:
            raise ValueError(
                "Step5 layer_types must include the dense stack and every MTP "
                f"layer: expected at least {total_num_layers} entries, got "
                f"{len(layer_types)}."
            )
        self.layer_types_with_mtp = (
            list(layer_types[:total_num_layers]) if layer_types else None
        )
        self.layer_types = (
            self.layer_types_with_mtp[:num_hidden_layers]
            if self.layer_types_with_mtp
            else None
        )
        self.use_rope_layers = use_rope_layers
        self.partial_rotary_factors = partial_rotary_factors
        self.yarn_only_types = yarn_only_types
        self.attention_other_setting = attention_other_setting
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swiglu_limits = swiglu_limits
        self.swiglu_limits_shared = swiglu_limits_shared
        self.zero_centered = zero_centered
        self.fp32_residual_connection = fp32_residual_connection

        resolved_bos_token_id = 1 if bos_token_id is None else bos_token_id
        resolved_eos_token_id = [2, 3] if eos_token_id is None else eos_token_id
        self.bos_token_id = resolved_bos_token_id
        self.eos_token_id = resolved_eos_token_id

        super().__init__(
            bos_token_id=resolved_bos_token_id,
            eos_token_id=resolved_eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Step5VLConfig(PreTrainedConfig):
    """Configuration for the Step5 vision-language model."""

    model_type = "step5_vl"
    architectures = ["Step5ForConditionalGeneration"]
    sub_configs = {
        "vision_config": PerceptionEncoderConfig,
        "text_config": Step5Config,
    }

    def __init__(
        self,
        vision_config: dict[str, Any] | PerceptionEncoderConfig | None = None,
        text_config: dict[str, Any] | Step5Config | None = None,
        understand_projector_stride: int = 2,
        projector_bias: bool = False,
        image_token_id: int = 128001,
        **kwargs: Any,
    ) -> None:
        if vision_config is None:
            vision_config = PerceptionEncoderConfig()
        elif isinstance(vision_config, dict):
            vision_config = PerceptionEncoderConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = Step5Config()
        elif isinstance(text_config, dict):
            text_config = Step5Config(**text_config)
        self.text_config = text_config

        self.understand_projector_stride = understand_projector_stride
        self.projector_bias = projector_bias
        self.hidden_size = text_config.hidden_size
        self.image_token_id = image_token_id

        if (
            "quantization_config" not in kwargs
            and (
                quantization_config := getattr(text_config, "quantization_config", None)
            )
            is not None
        ):
            kwargs["quantization_config"] = quantization_config
        if kwargs.get("architectures") is None:
            kwargs["architectures"] = list(type(self).architectures)
        super().__init__(**kwargs)


class Step5MTPConfig(Step5Config):
    """Configuration for a standalone Step5 MTP checkpoint.

    The usual in-tree MTP path derives its draft config from ``Step5Config``
    and changes the instance model type in memory. A separately published
    draft checkpoint is parsed directly with ``model_type="step5_mtp"`` and
    therefore needs a distinct registry class. Reusing ``Step5Config`` for both
    registry keys would be unsafe because registration mutates its class-level
    ``model_type``.
    """

    model_type = "step5_mtp"
    architectures = ["Step5MTP"]

    def __init__(self, num_nextn_predict_layers: int = 1, **kwargs: Any) -> None:
        if kwargs.get("architectures") is None:
            kwargs["architectures"] = list(type(self).architectures)
        super().__init__(
            num_nextn_predict_layers=num_nextn_predict_layers,
            **kwargs,
        )
