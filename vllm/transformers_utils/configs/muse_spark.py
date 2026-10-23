# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for Muse Spark HF checkpoints and DSpark drafts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from transformers import PreTrainedConfig


class _SubConfigArgs:
    """Base for the MuseSpark1x sub-configs; unknown keys are ignored."""

    def _update(self, data: dict[str, Any] | None, kwargs: dict[str, Any]) -> None:
        values = data if data is not None else kwargs
        for key, value in values.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class MoEArgs(_SubConfigArgs):
    """MoE sub-config of the MuseSpark1x text model."""

    def __init__(self, data: dict[str, Any] | None = None, **kwargs) -> None:
        self.is_enabled: bool = True
        self.topk: int = 1
        self.num_experts: int = 8
        self.router_in_fp32: bool = True
        self.route_scale: float = 1.0
        self.override_routed_hidden_dim: int | None = None
        self.override_shared_hidden_dim: int | None = None
        self.interleave_moe_layer_step: int = 1
        self.n_dense_layers_before_moe: int = 0
        self.use_shared_expert: bool = True
        self.router_score_normalization: str = "sigmoid"
        self.compression_factor: float | None = None
        self.use_pre_expert_norm: bool = False
        self.use_post_expert_norm: bool = False
        self.router_score_division_eps: float | None = None
        self._update(data, kwargs)


class GatedDeltaNetArgs(_SubConfigArgs):
    """GatedDeltaNet (GDN2) sub-config of the MuseSpark1x text model."""

    def __init__(self, data: dict[str, Any] | None = None, **kwargs) -> None:
        self.enable: bool = False
        self.mode: str = "chunk"
        self.use_gate: bool = True
        self.use_short_conv: bool = True
        self.conv_size: int = 4
        self.num_heads: int | None = None
        self.head_dim: int | None = None
        self.gated_delta_net_use_low_rank_proj: bool = False
        self.low_rank_compression_ratio: int = 4
        self._update(data, kwargs)


class MetapArgs(_SubConfigArgs):
    """Metap (muP-style) sub-config of the MuseSpark1x text model."""

    def __init__(self, data: dict[str, Any] | None = None, **kwargs) -> None:
        self.use_metap: bool = False
        self.use_metap_output_multiplier_without_metap: bool = False
        self.metap_mode: str = "mup"
        self.base_width: float = 256
        self.router_has_one_inf_dim: bool = True
        self.m_emb: float | None = None  # defaults to 10.0 with mup
        self._update(data, kwargs)
        if self.use_metap:
            assert self.metap_mode in ["mup", "sp"]


class MuseSpark1xTextConfig(PreTrainedConfig):
    """Configuration for the MuseSpark1x text model."""

    model_type = "musespark1x_text"
    rope_is_neox_style = True
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 202048,
        hidden_size: int = 5120,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 40,
        num_key_value_heads: int | None = 8,
        head_dim: int = 128,
        non_linearity: str = "swiglu",
        max_position_embeddings: int = 4096 * 32,
        rms_norm_eps: float = 1e-5,
        rms_post_norm_eps: float | None = None,
        norm_affine: bool = True,
        post_norm_affine: bool = True,
        post_norm_gain_center_type: str = "one",
        norm_zero_centered_gamma: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int = 200000,
        eos_token_id: int | list[int] | None = None,
        tie_word_embeddings: bool = False,
        moe: dict | MoEArgs | None = None,
        use_qk_norm: bool = True,
        use_residual_gate: bool = False,
        rope_theta: float = 500000,
        use_rope: bool = True,
        rope_use_fp32_in_outer_product: bool = True,
        every_n_layers_nope: int | None = None,
        use_sqrt_attn_scaling: bool = False,
        global_attn_cfg: Any = None,
        ffn_dim_multiplier: float = 1.0,
        metap: dict | MetapArgs | None = None,
        use_fp32_for_output: bool = False,
        numerics_args: dict | None = None,
        output_soft_cap_temp: Any = None,
        normalize_tok_embeddings: bool = False,
        use_attn_o_gate: bool = False,
        use_attn_o_norm: bool = False,
        residual_dtype: str = "bf16",
        output_norm_gain_center_type: str = "zero",
        use_ffn_learnable_scales: bool = False,
        hf_overrides: dict | None = None,
        every_n_layers_global: int | None = None,
        gated_delta_net: dict | GatedDeltaNetArgs | None = None,
        attn_temperature_tuning_floor_scale: float | None = None,
        attn_temperature_tuning_layers: Any = None,
        attn_temperature_tuning_q_scale_constant: Any = None,
        boi_token_index: int = 200080,
        eoi_token_index: int = 200081,
        image_token_index: int = 200092,
        **kwargs,
    ) -> None:
        if eos_token_id is None:
            eos_token_id = [200001, 200008]

        if hf_overrides and "special_stop_tokens" in hf_overrides:
            stop_str = hf_overrides["special_stop_tokens"]
            if stop_str and isinstance(stop_str, str):
                special_stop_tokens = [
                    int(token.strip()) for token in stop_str.split(",") if token.strip()
                ]
                if special_stop_tokens:
                    if isinstance(eos_token_id, int):
                        eos_token_id = [eos_token_id]
                    else:
                        eos_token_id = list(eos_token_id)
                    for token in special_stop_tokens:
                        if token not in eos_token_id:
                            eos_token_id.append(token)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.image_token_index = image_token_index
        self.use_sqrt_attn_scaling = use_sqrt_attn_scaling
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        assert non_linearity == "swiglu", f"{non_linearity=} is not supported"
        self.rms_norm_eps = rms_norm_eps
        self.rms_post_norm_eps = rms_post_norm_eps
        self.norm_affine = norm_affine
        self.post_norm_affine = post_norm_affine
        self.post_norm_gain_center_type = post_norm_gain_center_type
        self.norm_zero_centered_gamma = norm_zero_centered_gamma
        self.output_norm_gain_center_type = output_norm_gain_center_type
        self.head_dim = head_dim
        self.use_qk_norm = use_qk_norm
        self.use_attn_o_gate = use_attn_o_gate
        self.use_attn_o_norm = use_attn_o_norm
        self.use_residual_gate = use_residual_gate
        self.residual_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[
            residual_dtype
        ]
        self.use_ffn_learnable_scales = use_ffn_learnable_scales

        if isinstance(moe, dict):
            self.moe = MoEArgs(data=moe)
        elif isinstance(moe, MoEArgs):
            self.moe = moe
        else:
            self.moe = MoEArgs()

        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.every_n_layers_nope = every_n_layers_nope
        # "all", or one window per layer of the NoPE cycle (0 = global), possibly
        # as a string: "[gdn2,gdn2,gdn2,0]" -> ["gdn2", "gdn2", "gdn2", 0].
        if (
            isinstance(global_attn_cfg, str)
            and global_attn_cfg not in (None, "all")
            and global_attn_cfg.startswith("[")
            and global_attn_cfg.endswith("]")
        ):
            parsed_cfg: list[Any] = []
            for item in global_attn_cfg[1:-1].split(","):
                item = item.strip()
                try:
                    parsed_cfg.append(int(item))
                except ValueError:
                    parsed_cfg.append(item)
            global_attn_cfg = parsed_cfg

        if every_n_layers_global is not None:
            assert global_attn_cfg == "all", (
                "every_n_layers_global and global_attn_cfg cannot be set together"
            )
        self.every_n_layers_global = every_n_layers_global
        self.rope_use_fp32_in_outer_product = rope_use_fp32_in_outer_product
        if every_n_layers_nope is not None and not every_n_layers_global:
            assert len(global_attn_cfg) == every_n_layers_nope, (
                f"global_attn_cfg={global_attn_cfg} has length "
                f"{len(global_attn_cfg)}, but every_n_layers_nope is "
                f"{every_n_layers_nope}"
            )
        self.global_attn_cfg = global_attn_cfg if global_attn_cfg != "all" else None

        if isinstance(gated_delta_net, dict):
            self.gated_delta_net = GatedDeltaNetArgs(data=gated_delta_net)
        elif isinstance(gated_delta_net, GatedDeltaNetArgs):
            self.gated_delta_net = gated_delta_net
        else:
            self.gated_delta_net = GatedDeltaNetArgs()

        self.attn_temperature_tuning_floor_scale = attn_temperature_tuning_floor_scale
        self.attn_temperature_tuning_layers = attn_temperature_tuning_layers
        self.attn_temperature_tuning_q_scale_constant = (
            attn_temperature_tuning_q_scale_constant
        )

        self.intermediate_size_mlp = int(ffn_dim_multiplier * hidden_size)
        self.intermediate_size_shared_experts = int(
            self.moe.override_shared_hidden_dim or ffn_dim_multiplier * hidden_size
        )
        self.intermediate_size_routed_experts = int(
            self.moe.override_routed_hidden_dim or ffn_dim_multiplier * hidden_size
        )

        if isinstance(metap, dict):
            self.metap = MetapArgs(data=metap)
        elif isinstance(metap, MetapArgs):
            self.metap = metap
        else:
            self.metap = MetapArgs()

        self.num_local_experts = self.moe.num_experts

        # Newer checkpoints nest use_fp32_for_output under numerics_args.
        self.numerics_args = numerics_args or {}
        nested_fp32_out = (
            self.numerics_args.get("use_fp32_for_output")
            if isinstance(self.numerics_args, dict)
            else None
        )
        self.use_fp32_for_output = (
            use_fp32_for_output if nested_fp32_out is None else bool(nested_fp32_out)
        )
        self.output_soft_cap_temp = output_soft_cap_temp
        self.normalize_tok_embeddings = normalize_tok_embeddings

        if self.gated_delta_net.enable:
            self.architectures = ["MuseSpark1xHybridForCausalLM"]
        else:
            self.architectures = ["MuseSpark1xForCausalLM"]

        self.hf_overrides = hf_overrides

    def to_dict(self) -> dict[str, Any]:
        """Make the structured sub-configs and torch dtype JSON serializable."""
        output = super().to_dict()
        if self.residual_dtype == torch.bfloat16:
            output["residual_dtype"] = "bf16"
        elif self.residual_dtype == torch.float32:
            output["residual_dtype"] = "fp32"
        for name in ("moe", "gated_delta_net", "metap"):
            value = getattr(self, name, None)
            if value is not None:
                output[name] = value.to_dict()
        return output


class MuseSparkDSparkConfig(MuseSpark1xTextConfig):
    """Standalone configuration for a dense Muse Spark DSpark draft."""

    model_type = "muse_spark_dspark"
    rope_is_neox_style = False
    has_no_defaults_at_init = True

    def __init__(
        self,
        draft_vocab_size: int | None = None,
        draft_position_mode: str | None = None,
        **kwargs,
    ) -> None:
        if draft_position_mode is not None:
            kwargs["use_rope"] = draft_position_mode == "rope"
        super().__init__(**kwargs)
        self.draft_position_mode = "rope" if self.use_rope else "nope"
        self.draft_vocab_size = (
            self.vocab_size if draft_vocab_size is None else draft_vocab_size
        )
        self.architectures = ["MuseSparkDSparkModel"]


class MuseSparkTextConfig(PreTrainedConfig):
    model_type = "muse_spark_text"
    base_config_key = "text_config"
    has_no_defaults_at_init = True

    def __init__(self, use_cache: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.use_cache = use_cache
        if self.hidden_act != "silu" or self.attention_bias:
            raise ValueError("Muse Spark requires SiLU and bias-free attention")
        if self.rope_parameters["rope_type"] != "default":
            raise ValueError("Unsupported Muse Spark text RoPE type")
        if self.num_shared_experts:
            raise ValueError("Muse Spark shared experts are not supported")
        for field, choices in (
            ("layer_types", {"full_attention", "sliding_attention"}),
            ("mlp_layer_types", {"sparse", "dense"}),
        ):
            values = getattr(self, field)
            if len(values) != self.num_hidden_layers or not set(values) <= choices:
                raise ValueError(f"Invalid Muse Spark {field}")
        if len(self.layer_rope_theta) != self.num_hidden_layers:
            raise ValueError("Muse Spark requires one RoPE theta per layer")


class MuseSparkVisionConfig(PreTrainedConfig):
    model_type = "muse_spark_vision"
    base_config_key = "vision_config"
    has_no_defaults_at_init = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.hidden_act != "gelu":
            raise ValueError("Muse Spark requires GELU vision activations")
        if self.rope_parameters["rope_type"] != "axial":
            raise ValueError("Unsupported Muse Spark vision RoPE type")
        if len(self.layer_types) != self.num_hidden_layers or not set(
            self.layer_types
        ) <= {"full_attention", "window_attention"}:
            raise ValueError("Invalid Muse Spark vision attention types")


class MuseSparkAudioConfig(PreTrainedConfig):
    model_type = "muse_spark_audio"
    base_config_key = "audio_config"
    has_no_defaults_at_init = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.hidden_act != "gelu":
            raise ValueError("Muse Spark requires GELU audio activations")
        if not (
            len(self.patch_sizes)
            == len(self.depths)
            == len(self.hidden_sizes)
            == self.num_hidden_layers
        ):
            raise ValueError("Invalid Muse Spark audio stages")


class MuseSparkConfig(PreTrainedConfig):
    model_type = "muse_spark"
    has_no_defaults_at_init = True
    sub_configs = {
        "text_config": MuseSparkTextConfig,
        "vision_config": MuseSparkVisionConfig,
        "audio_config": MuseSparkAudioConfig,
    }

    def __init__(
        self,
        text_config: dict[str, Any] | MuseSparkTextConfig | None = None,
        vision_config: dict[str, Any] | MuseSparkVisionConfig | None = None,
        audio_config: dict[str, Any] | MuseSparkAudioConfig | None = None,
        projector_hidden_size: int | None = None,
        out_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        for name, value in (
            ("text_config", text_config),
            ("vision_config", vision_config),
            ("audio_config", audio_config),
        ):
            config_cls = self.sub_configs[name]
            if isinstance(value, dict):
                setattr(self, name, config_cls(**deepcopy(value)))
            elif isinstance(value, config_cls):
                setattr(self, name, value)
            else:
                raise ValueError(f"Muse Spark requires an HF {name}")
        if projector_hidden_size is None or out_hidden_size is None:
            raise ValueError("Muse Spark requires vision projector dimensions")
        if projector_hidden_act != "gelu":
            raise ValueError("Muse Spark requires a GELU vision projector")
        self.projector_hidden_size = projector_hidden_size
        self.out_hidden_size = out_hidden_size
        self.projector_hidden_act = projector_hidden_act
        if (
            out_hidden_size
            != self.vision_config.hidden_size * self.vision_config.merge_size**2
        ):
            raise ValueError("Muse Spark vision output must match patch merging")
        if self.audio_config.out_hidden_size != self.text_config.hidden_size:
            raise ValueError("Muse Spark audio output must match the text hidden size")
        self.architectures = ["MuseSparkForConditionalGeneration"]


__all__ = [
    "MuseSpark1xTextConfig",
    "MuseSparkConfig",
    "MuseSparkTextConfig",
    "MuseSparkVisionConfig",
    "MuseSparkAudioConfig",
    "MuseSparkDSparkConfig",
]
