# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://huggingface.co/apple/OpenELM-270M/blob/main/configuration_openelm.py
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# LICENSE: https://huggingface.co/apple/OpenELM-270M/blob/main/LICENSE
"""OpenELM configuration.

The configuration shipped with the OpenELM checkpoints on the HF Hub is not
compatible with Transformers v5, so vLLM provides its own copy.
"""

from numbers import Number

import numpy as np
from transformers.configuration_utils import PreTrainedConfig


def make_divisible(
    v: float | int,
    divisor: int = 8,
    min_value: float | int | None = None,
) -> int:
    """Round `v` to the nearest multiple of `divisor` (MobileNet rounding)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that rounding down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


def compute_heads(model_dim: int, head_dim: int) -> int:
    if model_dim % head_dim != 0:
        raise ValueError(
            f"Model dimension should be divisible by head dimension. "
            f"Got: {model_dim} and {head_dim}."
        )
    return model_dim // head_dim


def _linspace_multipliers(start: float, end: float, num: int) -> list[float]:
    return [round(float(v), 2) for v in np.linspace(start, end, num=num, dtype=float)]


class OpenELMConfig(PreTrainedConfig):
    model_type = "openelm"
    attribute_map = {
        "hidden_size": "model_dim",
        "num_hidden_layers": "num_transformer_layers",
        "max_position_embeddings": "max_context_length",
    }

    def __init__(
        self,
        vocab_size: int = 32000,
        max_context_length: int = 2048,
        num_transformer_layers: int = 12,
        model_dim: int = 2048,
        head_dim: int = 128,
        qkv_multipliers: float | list[float] = 1.0,
        num_query_heads: int | list[int] | None = None,
        num_gqa_groups: int = 1,
        ffn_multipliers: float | list[float] = 4.0,
        ffn_with_glu: bool = True,
        ffn_dim_divisor: int = 256,
        activation_fn_name: str = "swish",
        normalization_layer_name: str = "rms_norm",
        normalize_qk_projections: bool = False,
        share_input_output_layers: bool = False,
        rope_freq_constant: int = 10000,
        rope_max_length: int = 4096,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ) -> None:
        # Derived per-layer values are always recomputed below.
        kwargs.pop("num_kv_heads", None)
        kwargs.pop("tie_word_embeddings", None)

        self.vocab_size = vocab_size
        self.max_context_length = max_context_length
        self.num_transformer_layers = num_transformer_layers
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.qkv_multipliers = qkv_multipliers
        self.num_gqa_groups = num_gqa_groups
        self.ffn_multipliers: list[float] = self._expand_ffn_multipliers(
            ffn_multipliers, num_transformer_layers
        )
        self.ffn_with_glu = ffn_with_glu
        self.ffn_dim_divisor = ffn_dim_divisor
        self.activation_fn_name = activation_fn_name
        self.normalization_layer_name = normalization_layer_name
        self.normalize_qk_projections = normalize_qk_projections
        self.share_input_output_layers = share_input_output_layers
        self.rope_freq_constant = rope_freq_constant
        self.rope_max_length = rope_max_length
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        self._compute_per_layer_dims()

        kwargs.update(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=share_input_output_layers,
        )
        super().__init__(**kwargs)

    @staticmethod
    def _expand_ffn_multipliers(
        ffn_multipliers: float | list[float], num_layers: int
    ) -> list[float]:
        """Layer-wise scaling of the FFN width."""
        if isinstance(ffn_multipliers, Number):
            return [float(ffn_multipliers)] * num_layers
        if isinstance(ffn_multipliers, (tuple, list)):
            if len(ffn_multipliers) == 2:
                return _linspace_multipliers(
                    ffn_multipliers[0], ffn_multipliers[1], num_layers
                )
            if len(ffn_multipliers) == num_layers:
                return list(ffn_multipliers)
            raise ValueError(
                f"Expected {num_layers} FFN multipliers, got {len(ffn_multipliers)}."
            )
        raise NotImplementedError(
            f"FFN multipliers should be a single number or a list. Got: "
            f"{ffn_multipliers}."
        )

    def _compute_per_layer_dims(self) -> None:
        num_layers = self.num_transformer_layers
        head_multiple_of = self.num_gqa_groups if self.num_gqa_groups is not None else 2
        divisor = self.head_dim * head_multiple_of

        # Layer-wise scaling of the attention width.
        if isinstance(self.qkv_multipliers, Number):
            qkv_dim = make_divisible(self.model_dim * self.qkv_multipliers, divisor)
            query_dims = [qkv_dim] * num_layers
        elif (
            isinstance(self.qkv_multipliers, (tuple, list))
            and len(self.qkv_multipliers) == 2
        ):
            qkv_multipliers = _linspace_multipliers(
                self.qkv_multipliers[0], self.qkv_multipliers[1], num_layers
            )
            query_dims = [
                make_divisible(self.model_dim * m, divisor) for m in qkv_multipliers
            ]
        else:
            raise NotImplementedError(
                "QKV multipliers should be a single number or a list containing "
                f"exactly two numbers. Got: {self.qkv_multipliers}."
            )

        self.num_query_heads = [compute_heads(q, self.head_dim) for q in query_dims]
        self.num_kv_heads = [q // self.num_gqa_groups for q in self.num_query_heads]

        for layer_idx, (q_heads, kv_heads) in enumerate(
            zip(self.num_query_heads, self.num_kv_heads)
        ):
            if kv_heads == 0 or q_heads % kv_heads != 0:
                raise ValueError(
                    f"Layer {layer_idx}: number of query heads ({q_heads}) must be "
                    f"divisible by number of key/value heads ({kv_heads})."
                )

    def get_intermediate_size(self, layer_idx: int) -> int:
        return make_divisible(
            self.ffn_multipliers[layer_idx] * self.model_dim,
            divisor=self.ffn_dim_divisor,
        )
