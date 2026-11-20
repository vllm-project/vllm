# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ricursive (Ripple) model configuration."""

from transformers.configuration_utils import PreTrainedConfig


class RicursiveConfig(PreTrainedConfig):
    """Configuration for the Ricursive dense causal LM.

    Sliding-window layers use RoPE and full-attention layers are NoPE.
    `embed_scale` multiplies the input embeddings at runtime; checkpoints
    that fold it into `embed_tokens.weight` leave it unset.
    """

    model_type = "ricursive"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 100352,
        hidden_size: int = 5120,
        intermediate_size: int = 40960,
        num_hidden_layers: int = 80,
        num_attention_heads: int = 80,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2097152,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        sliding_window: int = 1024,
        layer_types: list[str] | None = None,
        use_gated_attention: bool = True,
        gate_full_precision: bool = True,
        use_valres: bool = False,
        ffn_clamp_threshold: float | None = None,
        embed_scale: float | None = None,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        **kwargs,
    ):
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["RicursiveForCausalLM"]
        # Derived by the hub config from layer_types; never read here.
        kwargs.pop("is_hybrid_swa", None)
        kwargs.pop("hybrid_layer_pattern", None)
        kwargs.pop("auto_map", None)
        if layer_types is None:
            layer_types = [
                "full_attention" if i % 5 == 4 else "sliding_attention"
                for i in range(num_hidden_layers)
            ]
        if len(layer_types) != num_hidden_layers or any(
            t not in ("full_attention", "sliding_attention") for t in layer_types
        ):
            raise ValueError(
                "layer_types must contain one valid attention type per layer."
            )
        kwargs.setdefault(
            "rope_parameters", {"rope_type": "default", "rope_theta": rope_theta}
        )

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
        self.use_gated_attention = use_gated_attention
        self.gate_full_precision = gate_full_precision
        self.use_valres = use_valres
        self.ffn_clamp_threshold = ffn_clamp_threshold
        self.embed_scale = embed_scale

        kwargs.update(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
        )
        super().__init__(**kwargs)


__all__ = ["RicursiveConfig"]
