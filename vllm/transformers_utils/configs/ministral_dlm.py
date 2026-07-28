# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config for Nemotron Ministral-based masked block-diffusion LMs.

The HF checkpoint ships a trust-remote-code ``MinistralDLMConfig`` that does
not expose ``canvas_length``, which vLLM uses to detect diffusion models
(``ModelConfig.is_diffusion``). This local config mirrors the Ministral
backbone fields and derives ``canvas_length`` from ``block_size``.
"""

from typing import Any

from transformers import PretrainedConfig


class MinistralDLMConfig(PretrainedConfig):
    model_type = "ministral_dlm"

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 3072,
        intermediate_size: int = 9216,
        num_hidden_layers: int = 26,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = False,
        rope_theta: float = 1000000.0,
        rope_parameters: dict[str, Any] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        sliding_window: int | None = None,
        mask_token_id: int = 100,
        block_size: int = 32,
        **kwargs: Any,
    ) -> None:
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
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.sliding_window = sliding_window

        # Masked block-diffusion fields. ``canvas_length`` drives vLLM's
        # diffusion detection and the per-step spec-decode canvas size.
        self.mask_token_id = mask_token_id
        self.block_size = block_size
        self.canvas_length = block_size

        # Normalize RoPE config: the checkpoint carries the same YaRN dict
        # under both ``rope_parameters`` and (legacy) ``rope_scaling``. Pop
        # the legacy key so PretrainedConfig's rope_scaling property setter
        # doesn't overwrite the normalized dict below.
        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_parameters = dict(rope_parameters or rope_scaling or {})
        rope_parameters.setdefault("rope_theta", rope_theta)
        rope_parameters.setdefault("rope_type", rope_parameters.pop("type", "default"))
        self.rope_parameters = rope_parameters
        self.rope_theta = rope_theta

        super().__init__(
            pad_token_id=kwargs.pop("pad_token_id", None),
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),
            **kwargs,
        )
