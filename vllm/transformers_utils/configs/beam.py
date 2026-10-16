# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam model configuration."""

from typing import Any

from transformers.configuration_utils import PreTrainedConfig

# Normalize dtype aliases for ``STR_DTYPE_TO_TORCH_DTYPE``.
_DTYPE_ALIASES = {"fp32": "float32", "bf16": "bfloat16"}


def _canonical_dtype(name: str | None) -> str | None:
    if name is None:
        return None
    canonical = _DTYPE_ALIASES.get(name, name)
    if canonical not in ("float32", "bfloat16"):
        raise ValueError(
            f"Unsupported dtype {name!r}; expected one of "
            f"{sorted(set(_DTYPE_ALIASES) | {'float32', 'bfloat16'})}."
        )
    return canonical


class BeamConfig(PreTrainedConfig):
    """Configuration for the Beam decoder-only MoE causal LM.

    Defaults match the Beam reference, including NoPE layers with
    SSMax query scaling, hybrid sliding-window attention, per-head Q/K RMSNorm,
    sandwich norm, depth-scaled residuals, sigmoid attention output gating, and
    grouped sigmoid ``noaux_tc`` routing.
    """

    model_type = "beam"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        ssmax: bool = True,
        ssmax_offset: int = 1,
        ssmax_divide_by_log_offset: bool = False,
        ssmax_scale_mode: str = "root10",
        use_paramfree_qk_norm: bool = False,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        attention_output_gate: bool = True,
        attention_output_gate_bias: bool = True,
        attention_output_gate_floor: float = 0.0,
        attention_output_gate_temperature: float = 1.0,
        attention_dropout: float = 0.0,
        sliding_window: int = 4096,
        layer_types: list[str] | None = None,
        nope_layer_ids: list[int] | None = None,
        moe_intermediate_size: int = 768,
        num_experts_per_tok: int = 8,
        num_experts: int = 128,
        norm_topk_prob: bool = True,
        scoring_func: str = "sigmoid",
        n_group: int | None = None,
        topk_group: int | None = None,
        routed_scaling_factor: float | None = None,
        topk_method: str | None = "noaux_tc",
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        enable_sandwich_norm: bool = True,
        self_attn_depth_scaled_residual: bool = True,
        mlp_depth_scaled_residual: bool = True,
        mlp_only_layers: list[int] | None = None,
        mlp_layer_types: list[str] | None = None,
        final_logit_softcapping: float | None = None,
        moe_router_dtype: str | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        if "num_local_experts" in kwargs:
            raise ValueError("Beam requires num_experts instead of num_local_experts.")
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["BeamForCausalLM"]

        no_rope_layers = kwargs.get("no_rope_layers")
        if no_rope_layers is not None and (
            len(no_rope_layers) != num_hidden_layers
            or any(flag not in (0, 1) for flag in no_rope_layers)
        ):
            raise ValueError("no_rope_layers must contain one 0/1 flag per layer.")
        if layer_types is not None and (
            len(layer_types) != num_hidden_layers
            or any(
                t not in ("full_attention", "sliding_attention") for t in layer_types
            )
        ):
            raise ValueError(
                "layer_types must contain one valid attention type per layer."
            )
        if nope_layer_ids is None and no_rope_layers is not None:
            # no_rope_layers[i] == 0 marks layer i as NoPE.
            nope_layer_ids = [
                i
                for i, uses_rope in enumerate(kwargs["no_rope_layers"])
                if not uses_rope
            ]
        elif nope_layer_ids is None and layer_types is not None:
            nope_layer_ids = [
                i for i, kind in enumerate(layer_types) if kind == "full_attention"
            ]
        if nope_layer_ids is not None and (
            len(set(nope_layer_ids)) != len(nope_layer_ids)
            or any(i < 0 or i >= num_hidden_layers for i in nope_layer_ids)
        ):
            raise ValueError("nope_layer_ids must contain unique valid layer indices.")
        # These dtype options are not supported by Beam.
        kwargs.pop("residual_in_fp32", None)
        kwargs.pop("lm_head_dtype", None)
        if moe_router_dtype is None:
            moe_router_dtype = kwargs.get("router_dtype")
        moe_router_dtype = _canonical_dtype(moe_router_dtype)
        rope_parameters = kwargs.get("rope_parameters")
        if rope_parameters and set(rope_parameters).issubset(
            {"sliding_attention", "full_attention"}
        ):
            # NoPE layers have no rotary parameters. Canonicalize before HF's
            # generic RoPE validation, which cannot process null entries.
            rotary_parameters = [p for p in rope_parameters.values() if p is not None]
            if rotary_parameters and any(
                p != rotary_parameters[0] for p in rotary_parameters[1:]
            ):
                raise ValueError(
                    "Beam requires identical RoPE parameters across layers."
                )
            rope_parameters = dict(rotary_parameters[0]) if rotary_parameters else {}
            rope_parameters.setdefault("rope_type", "default")
            rope_parameters.setdefault("rope_theta", rope_theta)
            kwargs["rope_parameters"] = rope_parameters
        if rope_parameters:
            rope_theta = rope_parameters.get("rope_theta", rope_theta)
        if layer_types is None:
            # Explicit layer types keep full-attention layers from inheriting
            # the global sliding window.
            nope_ids = set(nope_layer_ids or ())
            layer_types = [
                "full_attention"
                if not sliding_window or layer_id in nope_ids
                else "sliding_attention"
                for layer_id in range(num_hidden_layers)
            ]

        # Newer checkpoints carry per-layer feed-forward types; older ones list
        # the dense layers. Accept either, derive the other, reject disagreement.
        if mlp_layer_types is not None:
            if len(mlp_layer_types) != num_hidden_layers or any(
                t not in ("dense", "sparse") for t in mlp_layer_types
            ):
                raise ValueError(
                    "mlp_layer_types must contain one 'dense' or 'sparse' entry "
                    "per layer."
                )
            dense_layers = [i for i, t in enumerate(mlp_layer_types) if t == "dense"]
            if mlp_only_layers is not None and sorted(mlp_only_layers) != dense_layers:
                raise ValueError(
                    "mlp_only_layers disagrees with mlp_layer_types: "
                    f"{sorted(mlp_only_layers)} vs {dense_layers}."
                )
            mlp_only_layers = dense_layers
        else:
            mlp_only_layers = [] if mlp_only_layers is None else list(mlp_only_layers)
            if len(set(mlp_only_layers)) != len(mlp_only_layers) or any(
                i < 0 or i >= num_hidden_layers for i in mlp_only_layers
            ):
                raise ValueError(
                    "mlp_only_layers must contain unique valid layer indices."
                )
            mlp_layer_types = [
                "dense" if i in set(mlp_only_layers) else "sparse"
                for i in range(num_hidden_layers)
            ]

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.ssmax = ssmax
        self.ssmax_offset = ssmax_offset
        self.ssmax_divide_by_log_offset = ssmax_divide_by_log_offset
        self.ssmax_scale_mode = ssmax_scale_mode
        self.use_paramfree_qk_norm = use_paramfree_qk_norm
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_output_gate = attention_output_gate
        self.attention_output_gate_bias = attention_output_gate_bias
        self.attention_output_gate_floor = attention_output_gate_floor
        self.attention_output_gate_temperature = attention_output_gate_temperature
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.nope_layer_ids = [] if nope_layer_ids is None else nope_layer_ids
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.enable_sandwich_norm = enable_sandwich_norm
        self.self_attn_depth_scaled_residual = self_attn_depth_scaled_residual
        self.mlp_depth_scaled_residual = mlp_depth_scaled_residual
        self.mlp_only_layers = mlp_only_layers
        self.mlp_layer_types = mlp_layer_types
        self.final_logit_softcapping = final_logit_softcapping
        self.moe_router_dtype = moe_router_dtype

        kwargs.update(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
        )
        super().__init__(**kwargs)


__all__ = ["BeamConfig"]
