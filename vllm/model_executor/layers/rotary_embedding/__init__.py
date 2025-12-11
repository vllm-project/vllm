# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rotary Positional Embeddings."""

from typing import Any

import torch

from .base import RotaryEmbedding
from .deepseek_scaling_rope import DeepseekScalingRotaryEmbedding
from .dual_chunk_rope import DualChunkRotaryEmbedding
from .dynamic_ntk_alpha_rope import DynamicNTKAlphaRotaryEmbedding
from .dynamic_ntk_scaling_rope import DynamicNTKScalingRotaryEmbedding
from .linear_scaling_rope import LinearScalingRotaryEmbedding
from .llama3_rope import Llama3RotaryEmbedding
from .llama4_vision_rope import Llama4VisionRotaryEmbedding
from .mrope import MRotaryEmbedding
from .ntk_scaling_rope import NTKScalingRotaryEmbedding
from .phi3_long_rope_scaled_rope import Phi3LongRoPEScaledRotaryEmbedding
from .xdrope import XDRotaryEmbedding
from .yarn_scaling_rope import YaRNScalingRotaryEmbedding

_ROPE_DICT: dict[tuple, RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    max_position: int,
    is_neox_style: bool = True,
    rope_parameters: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    dual_chunk_attention_config: dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_parameters is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_parameters_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in rope_parameters.items()
        }
        rope_parameters_args = tuple(rope_parameters_tuple.items())
    else:
        rope_parameters_args = None

    if dual_chunk_attention_config is not None:
        dual_chunk_attention_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in dual_chunk_attention_config.items()
            if k != "sparse_attention_config"
        }
        dual_chunk_attention_args = tuple(dual_chunk_attention_tuple.items())
    else:
        dual_chunk_attention_args = None

    rope_parameters = rope_parameters or {}
    base = rope_parameters.get("rope_theta", 10000)
    scaling_type = rope_parameters.get("rope_type", "default")
    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)

    if partial_rotary_factor <= 0.0 or partial_rotary_factor > 1.0:
        raise ValueError(f"{partial_rotary_factor=} must be between 0.0 and 1.0")
    rotary_dim = int(head_size * partial_rotary_factor)

    key = (
        head_size,
        rotary_dim,
        max_position,
        is_neox_style,
        rope_parameters_args,
        dual_chunk_attention_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if dual_chunk_attention_config is not None:
        extra_kwargs = {
            k: v
            for k, v in dual_chunk_attention_config.items()
            if k in ("chunk_size", "local_size")
        }
        rotary_emb = DualChunkRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            dtype,
            **extra_kwargs,
        )
    elif scaling_type == "default":
        if "mrope_section" in rope_parameters:
            rotary_emb = MRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                mrope_section=rope_parameters["mrope_section"],
                mrope_interleaved=rope_parameters.get("mrope_interleaved", False),
            )
        else:
            rotary_emb = RotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
            )
    elif scaling_type == "llama3":
        scaling_factor = rope_parameters["factor"]
        low_freq_factor = rope_parameters["low_freq_factor"]
        high_freq_factor = rope_parameters["high_freq_factor"]
        original_max_position = rope_parameters["original_max_position_embeddings"]
        rotary_emb = Llama3RotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            dtype,
            scaling_factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position,
        )
    elif scaling_type == "mllama4":
        rotary_emb = Llama4VisionRotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        )
    elif scaling_type == "linear":
        scaling_factor = rope_parameters["factor"]
        rotary_emb = LinearScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            scaling_factor,
            dtype,
        )
    elif scaling_type == "ntk":
        scaling_factor = rope_parameters["factor"]
        mixed_b = rope_parameters.get("mixed_b")
        rotary_emb = NTKScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            scaling_factor,
            dtype,
            mixed_b,
        )
    elif scaling_type == "dynamic":
        if "alpha" in rope_parameters:
            scaling_alpha = rope_parameters["alpha"]
            rotary_emb = DynamicNTKAlphaRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_alpha,
                dtype,
            )
        elif "factor" in rope_parameters:
            scaling_factor = rope_parameters["factor"]
            rotary_emb = DynamicNTKScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
            )
        else:
            raise ValueError(
                "Dynamic rope scaling must contain either 'alpha' or 'factor' field"
            )
    elif scaling_type == "xdrope":
        scaling_alpha = rope_parameters["alpha"]
        rotary_emb = XDRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            scaling_alpha,
            dtype,
            xdrope_section=rope_parameters["xdrope_section"],
        )
    elif scaling_type == "yarn":
        scaling_factor = rope_parameters["factor"]
        original_max_position = rope_parameters["original_max_position_embeddings"]
        extra_kwargs = {
            k: v
            for k, v in rope_parameters.items()
            if k
            in (
                "extrapolation_factor",
                "attn_factor",
                "beta_fast",
                "beta_slow",
                "apply_yarn_scaling",
                "truncate",
            )
        }
        if "mrope_section" in rope_parameters:
            extra_kwargs.pop("apply_yarn_scaling", None)
            rotary_emb = MRotaryEmbedding(
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                dtype,
                mrope_section=rope_parameters["mrope_section"],
                mrope_interleaved=rope_parameters.get("mrope_interleaved", False),
                scaling_factor=scaling_factor,
                **extra_kwargs,
            )
        else:
            rotary_emb = YaRNScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
                **extra_kwargs,
            )
    elif scaling_type in ["deepseek_yarn", "deepseek_llama_scaling"]:
        scaling_factor = rope_parameters["factor"]
        original_max_position = rope_parameters["original_max_position_embeddings"]
        # assert max_position == original_max_position * scaling_factor
        extra_kwargs = {
            k: v
            for k, v in rope_parameters.items()
            if k
            in (
                "extrapolation_factor",
                "attn_factor",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            )
        }
        rotary_emb = DeepseekScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            original_max_position,
            base,
            is_neox_style,
            scaling_factor,
            dtype,
            **extra_kwargs,
        )
    elif scaling_type == "longrope":
        short_factor = rope_parameters["short_factor"]
        long_factor = rope_parameters["long_factor"]
        original_max_position = rope_parameters["original_max_position_embeddings"]
        extra_kwargs = {
            k: v
            for k, v in rope_parameters.items()
            if k in ("short_mscale", "long_mscale")
        }
        rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            original_max_position,
            base,
            is_neox_style,
            dtype,
            short_factor,
            long_factor,
            **extra_kwargs,
        )
    else:
        raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb
