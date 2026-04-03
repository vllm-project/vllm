# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test cases for ViT attention score extraction configuration."""

from vllm.config import MultiModalConfig


def test_multimodal_config_has_attention_score_fields():
    """Test that MultiModalConfig has the new attention score fields."""
    config = MultiModalConfig()

    # Test default values
    assert hasattr(config, "extract_vit_attention_score")
    assert not config.extract_vit_attention_score

    assert hasattr(config, "vit_attention_score_layer_index")
    assert config.vit_attention_score_layer_index == -2


def test_multimodal_config_custom_values():
    """Test setting custom values for attention score extraction."""
    config = MultiModalConfig(
        extract_vit_attention_score=True, vit_attention_score_layer_index=-3
    )

    assert config.extract_vit_attention_score
    assert config.vit_attention_score_layer_index == -3


def test_multimodal_config_negative_layer_index():
    """Test that negative layer indices are supported."""
    config = MultiModalConfig(
        extract_vit_attention_score=True,
        vit_attention_score_layer_index=-1,  # Last layer
    )

    assert config.vit_attention_score_layer_index == -1


def test_multimodal_config_positive_layer_index():
    """Test that positive layer indices are supported."""
    config = MultiModalConfig(
        extract_vit_attention_score=True,
        vit_attention_score_layer_index=5,  # Layer 5
    )

    assert config.vit_attention_score_layer_index == 5
