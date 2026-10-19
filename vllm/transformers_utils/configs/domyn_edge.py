# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: 2026 Domyn

# Adapted from https://huggingface.co/domyn/Domyn-Edge-Base
"""DomynEdge model configuration."""

from transformers import PreTrainedConfig


class DomynEdgeConfig(PreTrainedConfig):
    """Configuration for the DomynEdge architecture.

    DomynEdge interleaves RoPE and NoPE attention layers, uses per-layer
    head sizes and per-layer sliding windows, and applies Gemma2-style
    sandwiched RMSNorms around each sublayer.
    """

    model_type = "domynedge"

    def __init__(
        self,
        vocab_size: int = 115269,
        hidden_size: int = 2560,
        intermediate_size: int = 9728,
        hidden_act: str = "silu",
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        rope_parameters: dict | None = None,
        rope_theta: float = 10000.0,
        max_window_layers: int | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        position_layers: list[str] | None = None,
        window_sizes: list | None = None,
        head_sizes: list[int] | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        if position_layers is None:
            position_layers = (
                ["nope", "nope"]
                + ["rope", "rope", "rope", "nope"] * 8
                + ["nope", "nope"]
            )
        if window_sizes is None:
            window_sizes = (
                [(-1, 0), (-1, 0)]
                + [(1024, 0), (1024, 0), (1024, 0), (-1, 0)] * 8
                + [(-1, 0), (-1, 0)]
            )
        if head_sizes is None:
            head_sizes = [256, 256] + [64, 64, 64, 256] * 8 + [256, 256]

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.position_layers = position_layers
        self.window_sizes = window_sizes
        self.head_sizes = head_sizes

        # validation (mirrors the reference implementation)
        if len(self.position_layers) != self.num_hidden_layers:
            raise ValueError(
                f"Length of position_layers ({len(self.position_layers)}) must "
                f"equal num_hidden_layers ({self.num_hidden_layers})"
            )
        if any(pl not in ("rope", "nope") for pl in self.position_layers):
            raise ValueError(
                "position_layers must be one of 'rope', 'nope'. "
                f"Got {self.position_layers}"
            )
        if len(self.window_sizes) != self.num_hidden_layers:
            raise ValueError(
                f"Length of window_sizes ({len(self.window_sizes)}) must "
                f"equal num_hidden_layers ({self.num_hidden_layers})"
            )
        if len(self.head_sizes) != self.num_hidden_layers:
            raise ValueError(
                f"Length of head_sizes ({len(self.head_sizes)}) must "
                f"equal num_hidden_layers ({self.num_hidden_layers})"
            )
        if any(not isinstance(hs, int) for hs in self.head_sizes):
            raise ValueError(
                f"head_sizes must be a list of integers. Got {self.head_sizes}"
            )

        # All RoPE layers must share a single head size (the rotary dimension).
        rope_dims = [
            hs for pl, hs in zip(self.position_layers, self.head_sizes) if pl == "rope"
        ]
        if len(set(rope_dims)) > 1:
            raise ValueError(
                f"All rope layers must have the same head size. Got {rope_dims}"
            )
        self.rope_dim = rope_dims[0] if rope_dims else None

        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        elif not isinstance(rope_parameters, dict):
            rope_parameters = {
                "rope_type": getattr(rope_parameters, "rope_type", "default"),
                "rope_theta": getattr(rope_parameters, "rope_theta", rope_theta),
            }
        else:
            rope_parameters = dict(rope_parameters)
        rope_parameters.setdefault("rope_type", "default")
        rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_parameters = rope_parameters

        # Per-layer head sizes; expose the largest for ModelConfig.get_head_size().
        self.head_dim = max(self.head_sizes)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
