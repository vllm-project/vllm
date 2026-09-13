# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for dynamic NTK scaling below and above the trained context length."""

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding.dynamic_ntk_scaling_rope import (
    DynamicNTKScalingRotaryEmbedding,
)


def _make_rope(max_position_embeddings: int, max_trained_positions: int = 2048):
    return DynamicNTKScalingRotaryEmbedding(
        head_size=64,
        rotary_dim=64,
        max_position_embeddings=max_position_embeddings,
        max_trained_positions=max_trained_positions,
        base=1000.0,
        is_neox_style=True,
        scaling_factor=2.0,
        dtype=torch.float32,
    )


def _compute_cache(max_position_embeddings: int, inv_freq: torch.Tensor):
    t = torch.arange(max_position_embeddings, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


@pytest.mark.parametrize("max_position_embeddings", [512, 2048])
def test_no_scaling_at_or_below_trained_length(
    default_vllm_config, max_position_embeddings
):
    rope = _make_rope(max_position_embeddings)
    expected = _compute_cache(
        max_position_embeddings, rope._compute_inv_freq(rope.base)
    )
    assert torch.allclose(rope._compute_cos_sin_cache(), expected)


def test_scaling_above_trained_length(default_vllm_config):
    rope = _make_rope(4096)
    scaled_base = rope.base * (
        rope.scaling_factor * rope.max_position_embeddings / rope.max_trained_positions
        - (rope.scaling_factor - 1)
    ) ** (rope.rotary_dim / (rope.rotary_dim - 2))
    expected = _compute_cache(
        rope.max_position_embeddings, rope._compute_inv_freq(scaled_base)
    )
    assert torch.allclose(rope._compute_cos_sin_cache(), expected)
