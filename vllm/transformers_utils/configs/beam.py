# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam model configuration."""

from transformers.configuration_utils import PreTrainedConfig


class BeamConfig(PreTrainedConfig):
    """Configuration for the Beam decoder-only MoE causal LM.

    Full-attention layers are NoPE with SSMax query scaling; sliding-window
    layers use RoPE. ``mlp_layer_types`` marks each layer's feed-forward as
    ``dense`` or ``sparse`` (grouped sigmoid ``noaux_tc`` MoE).
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
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = False,
        sliding_window: int = 4096,
        layer_types: list[str] | None = None,
        mlp_layer_types: list[str] | None = None,
        moe_intermediate_size: int = 768,
        num_experts_per_tok: int = 8,
        num_experts: int = 128,
        norm_topk_prob: bool = True,
        n_group: int | None = None,
        topk_group: int | None = None,
        routed_scaling_factor: float = 1.0,
        final_logit_softcapping: float | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["BeamForCausalLM"]
        if layer_types is None:
            layer_types = ["sliding_attention"] * num_hidden_layers
        if len(layer_types) != num_hidden_layers or any(
            t not in ("full_attention", "sliding_attention") for t in layer_types
        ):
            raise ValueError(
                "layer_types must contain one valid attention type per layer."
            )
        if mlp_layer_types is None:
            mlp_layer_types = ["sparse"] * num_hidden_layers
        if len(mlp_layer_types) != num_hidden_layers or any(
            t not in ("dense", "sparse") for t in mlp_layer_types
        ):
            raise ValueError(
                "mlp_layer_types must contain one 'dense' or 'sparse' entry per layer."
            )
        rope_parameters = kwargs.setdefault(
            "rope_parameters", {"rope_type": "default", "rope_theta": 10000.0}
        )
        if set(rope_parameters) <= {
            "sliding_attention",
            "full_attention",
        }:
            # NoPE layers have no rotary parameters. Canonicalize before HF's
            # generic RoPE validation, which cannot process null entries.
            rotary_parameters = [p for p in rope_parameters.values() if p is not None]
            if len(rotary_parameters) != 1:
                raise ValueError("Beam expects RoPE parameters for exactly one type.")
            kwargs["rope_parameters"] = rotary_parameters[0]

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.mlp_layer_types = mlp_layer_types
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.final_logit_softcapping = final_logit_softcapping

        kwargs.update(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
        )
        super().__init__(**kwargs)


__all__ = ["BeamConfig"]
