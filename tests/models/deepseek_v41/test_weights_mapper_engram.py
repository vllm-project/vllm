# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DeepSeek-V4.1 engram scale key regex mapping."""

import pytest

from vllm.models.deepseek_v41.amd.model import (
    _make_deepseek_v4_weights_mapper as _make_amd_mapper,
)
from vllm.models.deepseek_v41.nvidia.model import (
    _make_deepseek_v4_weights_mapper as _make_nvidia_mapper,
)


@pytest.mark.parametrize("expert_dtype", ["fp4", "fp8"])
def test_engram_scale_and_weight_scale_mapping_amd(expert_dtype):
    """Verify AMD mapper translates both .scale and .weight_scale engram keys."""
    mapper = _make_amd_mapper(expert_dtype)

    # .scale key
    mapped_scale = mapper._map_name("layers.1.engram.embed.scale")
    assert mapped_scale == "model.layers.1.engram.embed_tokens.weight_scale_inv"

    # .weight_scale key (Quark export format)
    mapped_weight_scale = mapper._map_name("layers.1.engram.embed.weight_scale")
    assert mapped_weight_scale == "model.layers.1.engram.embed_tokens.weight_scale_inv"

    # Higher layer index
    mapped_layer14 = mapper._map_name("layers.14.engram.embed.weight_scale")
    assert mapped_layer14 == "model.layers.14.engram.embed_tokens.weight_scale_inv"


@pytest.mark.parametrize("expert_dtype", ["fp4", "fp8"])
def test_engram_scale_and_weight_scale_mapping_nvidia(expert_dtype):
    """Verify NVIDIA mapper exhibits parity in translating engram keys."""
    mapper = _make_nvidia_mapper(expert_dtype)

    mapped_scale = mapper._map_name("layers.2.engram.embed.scale")
    assert mapped_scale == "model.layers.2.engram.embed_tokens.weight_scale_inv"

    mapped_weight_scale = mapper._map_name("layers.2.engram.embed.weight_scale")
    assert mapped_weight_scale == "model.layers.2.engram.embed_tokens.weight_scale_inv"


def test_engram_weight_unaffected_by_scale_regex():
    """Verify engram weight mapping is preserved without scale regex interference."""
    mapper = _make_amd_mapper("fp4")
    mapped_weight = mapper._map_name("layers.1.engram.embed.weight")
    assert mapped_weight == "model.layers.1.engram.embed_tokens.weight"


def test_standard_linear_scale_not_corrupted():
    """Verify standard attention scales are mapped correctly."""
    mapper = _make_amd_mapper("fp4", linear_scale_name="weight_scale_inv")
    mapped_attn = mapper._map_name("layers.0.self_attn.q_proj.scale")
    assert mapped_attn == "model.layers.0.self_attn.q_proj.weight_scale_inv"
