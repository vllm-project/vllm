# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.dynamic_ntk_scaling_rope import (
    DynamicNTKScalingRotaryEmbedding,
)

HEAD_SIZE = 64
BASE = 1000.0
MAX_TRAINED_POSITIONS = 2048
SCALING_FACTOR = 2.0


def _dynamic_ntk_cache(max_position_embeddings: int) -> torch.Tensor:
    rope = DynamicNTKScalingRotaryEmbedding(
        HEAD_SIZE,
        HEAD_SIZE,
        max_position_embeddings,
        MAX_TRAINED_POSITIONS,
        BASE,
        is_neox_style=True,
        scaling_factor=SCALING_FACTOR,
        dtype=torch.float32,
    )
    return rope.cos_sin_cache


def _plain_cache(max_position_embeddings: int) -> torch.Tensor:
    rope = RotaryEmbedding(
        HEAD_SIZE,
        HEAD_SIZE,
        max_position_embeddings,
        BASE,
        is_neox_style=True,
        dtype=torch.float32,
    )
    return rope.cos_sin_cache


@pytest.mark.parametrize("max_position_embeddings", [512, MAX_TRAINED_POSITIONS])
def test_no_scaling_up_to_trained_length(default_vllm_config, max_position_embeddings):
    """Serving at or below the trained length must not rescale the base.

    Applying the NTK formula there raises a negative number to a fractional
    power, which produced a complex cache and NaN outputs.
    """
    cache = _dynamic_ntk_cache(max_position_embeddings)

    assert cache.dtype == torch.float32
    assert torch.isfinite(cache).all()
    torch.testing.assert_close(cache, _plain_cache(max_position_embeddings))


def test_scaling_beyond_trained_length(default_vllm_config):
    max_position_embeddings = 2 * MAX_TRAINED_POSITIONS
    cache = _dynamic_ntk_cache(max_position_embeddings)

    assert torch.isfinite(cache).all()
    assert cache.abs().max() <= 1.0
    assert not torch.equal(cache, _plain_cache(max_position_embeddings))
