# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Nemotron Shortcut model configuration."""

from typing import Literal

from transformers import PreTrainedConfig
from transformers.models.pixtral import PixtralVisionConfig

LAYER_TYPES = (
    "linear_attention",
    "mlp",
    "attn",
    "shortcut_moe_attn",
    "shortcut_moe_gdp",
)


class NemotronShortcutTextConfig(PreTrainedConfig):
    """Config of the Nemotron Shortcut language model.

    Depth comes from `layer_types`, which names the block at each position:
    `linear_attention` (GDP), `mlp`, and the two shortcut MoE blocks whose
    mixer is either attention (`shortcut_moe_attn`) or GDP
    (`shortcut_moe_gdp`).

    Model-variant switches, whose defaults keep the original behavior:
    `moe_router_bias_space` is where the router's correction bias enters
    expert selection: `"score"` picks `topk(sigmoid(logits) + bias)`, `"logit"`
    picks `topk(logits + bias)`; the mixing weights are the unbiased sigmoid
    scores either way. `moe_shortcut_disable_mtp` builds the MTP block as
    attention then MoE instead of a shortcut-MoE block. `gdp_g_clamp_value`
    lower-bounds the GDP log decay `g` at `-gdp_g_clamp_value`.

    `index_topk` makes the attention sparse (SDSA): each query attends to its
    Top-`index_topk` keys, which an indexer of width `index_head_dim` selects.
    Without `index_topk` the attention is full. The `sdsa_*` fields set SDSA's
    other choices, each "auto" by default: the dtype of the k cache and of the
    indexer dot product, and each phase's core and fused-selector rule.
    """

    model_type = "nemotron4_text"

    def __init__(
        self,
        layer_types: list[str] | None = None,
        moe_router_bias_space: str | None = None,
        moe_shortcut_disable_mtp: bool = False,
        gdp_g_clamp_value: float | None = None,
        index_topk: int | None = None,
        index_head_dim: int = 128,
        sdsa_indexer_dtype: Literal["auto", "nvfp4", "bf16"] = "auto",
        sdsa_decode_core: Literal["auto", "token", "fa4"] = "auto",
        sdsa_decode_fused: bool | list[int] | Literal["auto"] = "auto",
        sdsa_prefill_core: Literal["auto", "token", "fa4", "union"] = "auto",
        sdsa_prefill_fused: bool | list[int] | Literal["auto"] = "auto",
        **kwargs,
    ):
        # Older exports name the bias space `moe_router_qb_bias_space`.
        legacy_bias_space = kwargs.pop("moe_router_qb_bias_space", None)
        super().__init__(**kwargs)
        self.layer_types = list(layer_types or [])
        if unknown := set(self.layer_types) - set(LAYER_TYPES):
            raise ValueError(
                f"Unknown Nemotron Shortcut layer types: {sorted(unknown)}"
            )
        self.moe_router_bias_space = (
            moe_router_bias_space or legacy_bias_space or "score"
        )
        if self.moe_router_bias_space not in ("score", "logit"):
            raise ValueError(
                "moe_router_bias_space must be 'score' or 'logit', got "
                f"{self.moe_router_bias_space!r}"
            )
        self.moe_shortcut_disable_mtp = moe_shortcut_disable_mtp
        if gdp_g_clamp_value is not None and not gdp_g_clamp_value > 0:
            raise ValueError("gdp_g_clamp_value must be positive when set")
        self.gdp_g_clamp_value = gdp_g_clamp_value
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.sdsa_indexer_dtype = sdsa_indexer_dtype
        self.sdsa_decode_core = sdsa_decode_core
        self.sdsa_decode_fused = sdsa_decode_fused
        self.sdsa_prefill_core = sdsa_prefill_core
        self.sdsa_prefill_fused = sdsa_prefill_fused

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layer_types)


class NemotronShortcutConfig(PreTrainedConfig):
    model_type = "nemotron4"
    sub_configs = {
        "text_config": NemotronShortcutTextConfig,
        "vision_config": PixtralVisionConfig,
    }

    def __init__(
        self,
        text_config: dict | NemotronShortcutTextConfig | None = None,
        vision_config: dict | PixtralVisionConfig | None = None,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["Nemotron4ForConditionalGeneration"])
        super().__init__(**kwargs)
        if not isinstance(text_config, NemotronShortcutTextConfig):
            text_config = NemotronShortcutTextConfig(**(text_config or {}))
        if not isinstance(vision_config, PixtralVisionConfig):
            vision_config = dict(vision_config or {})
            rope = dict(vision_config.get("rope_parameters") or {})
            if rope.get("rope_type") == "axial":
                # Stock Pixtral's default RoPE implements the same 2D angles.
                rope["rope_type"] = "default"
                vision_config["rope_parameters"] = rope
            vision_config = PixtralVisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
