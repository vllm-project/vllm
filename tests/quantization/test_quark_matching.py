# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Quark layer pattern matching and prefix stripping."""

import pytest
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.quark.utils import should_ignore_layer


@pytest.fixture
def quark_sample_config():
    """Sample Quark configuration with potential substring collision patterns."""
    config_dict = {
        "quant_method": "quark",
        "layer_quant_config": {
            "layers.1.mlp.gate_proj": {"weight": {"block_size": [128, 128]}},
            "layers.18.mlp.gate_proj": {"weight": {"block_size": [32, 32]}},
            "model.layers.5.self_attn.q_proj": {"weight": {"block_size": [64, 64]}},
            "*layers.2.*": {"weight": {"block_size": [16, 16]}},
        },
    }
    return QuarkConfig(config_dict)


def test_dot_delimited_pattern_matching_prevents_substring_collision(quark_sample_config):
    """Verify layers.1 does not collide with or match layers.18."""
    cfg_1 = quark_sample_config.get_layer_quant_config_from_name("layers.1.mlp.gate_proj")
    cfg_18 = quark_sample_config.get_layer_quant_config_from_name("layers.18.mlp.gate_proj")

    assert cfg_1 is not None
    assert cfg_18 is not None
    assert cfg_1["weight"]["block_size"] == [128, 128]
    assert cfg_18["weight"]["block_size"] == [32, 32]

    # Non-existent layer 19 should not match layer 1 or 18
    cfg_19 = quark_sample_config.get_layer_quant_config_from_name("layers.19.mlp.gate_proj")
    assert cfg_19 is None


def test_prefix_stripping_model_dot_symmetry(quark_sample_config):
    """Verify lookups succeed symmetrically whether model. prefix is present in layer or pattern."""
    # Pattern was defined without model. prefix ("layers.1.mlp.gate_proj")
    cfg_with_model = quark_sample_config.get_layer_quant_config_from_name("model.layers.1.mlp.gate_proj")
    assert cfg_with_model is not None
    assert cfg_with_model["weight"]["block_size"] == [128, 128]

    # Pattern was defined with model. prefix ("model.layers.5.self_attn.q_proj")
    cfg_without_model = quark_sample_config.get_layer_quant_config_from_name("layers.5.self_attn.q_proj")
    assert cfg_without_model is not None
    assert cfg_without_model["weight"]["block_size"] == [64, 64]


def test_wildcard_pattern_with_prefix_stripping(quark_sample_config):
    """Verify fnmatch wildcard patterns match regardless of model. prefix."""
    cfg_wildcard = quark_sample_config.get_layer_quant_config_from_name("model.layers.2.mlp.down_proj")
    assert cfg_wildcard is not None
    assert cfg_wildcard["weight"]["block_size"] == [16, 16]

    # Unrelated layer should not match wildcard
    cfg_wildcard_neg = quark_sample_config.get_layer_quant_config_from_name("model.layers.20.mlp.down_proj")
    assert cfg_wildcard_neg is None


def test_should_ignore_layer_quark_utils():
    """Verify dot-delimited layer ignore checks in quark utils for MoE parent modules."""
    ignored = ["layers.1.mlp.experts.*.down_proj"]

    # Parent layer 1 experts module must be ignored because child is in ignore list
    assert should_ignore_layer("model.layers.1.mlp.experts", ignore=ignored, check_children=True)
    assert should_ignore_layer("layers.1.mlp.experts", ignore=ignored, check_children=True)

    # Parent layer 18 experts module must NOT be ignored
    assert not should_ignore_layer("model.layers.18.mlp.experts", ignore=ignored, check_children=True)
    assert not should_ignore_layer("layers.18.mlp.experts", ignore=ignored, check_children=True)
