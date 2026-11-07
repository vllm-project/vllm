# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NTK and Dynamic NTK RoPE cos/sin cache length."""

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding.dynamic_ntk_scaling_rope import (
    DynamicNTKScalingRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.ntk_scaling_rope import (
    NTKScalingRotaryEmbedding,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

HEAD_SIZE = 64
ROTARY_DIM = 64


def _make_ntk(max_position, scaling_factor, mixed_b=None):
    return NTKScalingRotaryEmbedding(
        head_size=HEAD_SIZE,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=max_position,
        base=10000.0,
        is_neox_style=True,
        scaling_factor=scaling_factor,
        dtype=torch.float32,
        mixed_b=mixed_b,
    )


def _make_dynamic_ntk(max_position, max_trained_positions, scaling_factor):
    return DynamicNTKScalingRotaryEmbedding(
        head_size=HEAD_SIZE,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=max_position,
        max_trained_positions=max_trained_positions,
        base=10000.0,
        is_neox_style=True,
        scaling_factor=scaling_factor,
        dtype=torch.float32,
    )


def _rotate(rope, positions):
    query = torch.randn(len(positions), 1, HEAD_SIZE, dtype=torch.float32)
    key = torch.randn(len(positions), 1, HEAD_SIZE, dtype=torch.float32)
    return rope.forward_native(torch.tensor(positions, dtype=torch.long), query, key)


@pytest.mark.parametrize("mixed_b", [None, 0.5])
def test_ntk_cache_covers_scaled_context(default_vllm_config, mixed_b):
    rope = _make_ntk(2048, 2.0, mixed_b)

    assert rope.cos_sin_cache.shape == (4096, ROTARY_DIM)


def test_ntk_rotates_positions_beyond_unscaled_length(default_vllm_config):
    rope = _make_ntk(2048, 2.0)

    out_q, out_k = _rotate(rope, [0, 2047, 2048, 4095])

    assert not torch.isnan(out_q).any()
    assert out_k is not None and not torch.isnan(out_k).any()


def test_ntk_cache_never_shrinks_below_unscaled_length(default_vllm_config):
    rope = _make_ntk(2048, 0.5)

    assert rope.cos_sin_cache.shape == (2048, ROTARY_DIM)


def test_dynamic_ntk_cache_covers_scaled_context(default_vllm_config):
    rope = _make_dynamic_ntk(2048, 2048, 2.0)

    assert rope.cos_sin_cache.shape == (4096, ROTARY_DIM)


def test_dynamic_ntk_rotates_positions_beyond_unscaled_length(default_vllm_config):
    rope = _make_dynamic_ntk(2048, 2048, 2.0)

    out_q, out_k = _rotate(rope, [0, 2047, 2048, 4095])

    assert not torch.isnan(out_q).any()
    assert out_k is not None and not torch.isnan(out_k).any()


def test_dynamic_ntk_cache_is_not_rescaled_when_caller_passes_served_length(
    default_vllm_config,
):
    """Callers such as NomicBertModelConfig pass the already-scaled served
    length as max_position_embeddings, so scaling it again would over-allocate.
    """
    rope = _make_dynamic_ntk(8192, 2048, 4.0)

    assert rope.cos_sin_cache.shape == (8192, ROTARY_DIM)
