# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for compressed-tensors matching utilities.

Tests verify that the compressed-tensors library's matching functions work
correctly for vLLM use cases, including fused modules, regex patterns, and
class name matching.

Run `pytest tests/quantization/test_compressed_tensors_matching.py`.
"""

import pytest
import torch
from compressed_tensors.utils.match import is_match, match_targets

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)


class MockLinear(torch.nn.Module):
    """Mock linear layer for testing."""

    pass


class MockAttention(torch.nn.Module):
    """Mock attention layer for testing."""

    pass


def test_exact_name_match():
    """Test exact layer name matching."""
    layer = MockLinear()

    assert is_match("layers.0.q_proj", layer, "layers.0.q_proj")
    assert is_match(
        "model.layers.5.mlp.gate_proj", layer, "model.layers.5.mlp.gate_proj"
    )
    assert not is_match("layers.0.q_proj", layer, "layers.1.q_proj")
    assert not is_match("layers.0.q_proj", layer, "layers.0.k_proj")


def test_regex_pattern_match():
    """Test regex pattern matching with 're:' prefix."""
    layer = MockLinear()

    # Test basic regex patterns
    assert is_match("layers.0.q_proj", layer, "re:.*q_proj")
    assert is_match("layers.0.self_attn.q_proj", layer, "re:.*self_attn.*")
    assert is_match("model.layers.5.mlp.gate_proj", layer, "re:.*mlp.*proj")

    # Test layer number patterns
    assert is_match("layers.0.q_proj", layer, "re:layers\\.[0-9]+\\.q_proj")
    assert is_match("layers.15.q_proj", layer, "re:layers\\.[0-9]+\\.q_proj")

    # Test non-matching patterns
    assert not is_match("layers.0.q_proj", layer, "re:.*k_proj")
    assert not is_match("model.mlp.gate", layer, "re:.*attention.*")


def test_multiple_targets():
    """Test matching against multiple targets."""
    layer = MockLinear()

    # Should match if any target matches
    targets = ["layers.0.k_proj", "layers.0.q_proj", "layers.0.v_proj"]
    assert is_match("layers.0.q_proj", layer, targets)
    assert is_match("layers.0.k_proj", layer, targets)
    assert not is_match("layers.0.o_proj", layer, targets)

    # Test with regex patterns
    targets_with_regex = ["re:.*q_proj", "re:.*k_proj"]
    assert is_match("layers.5.q_proj", layer, targets_with_regex)
    assert is_match("model.layers.0.k_proj", layer, targets_with_regex)
    assert not is_match("layers.0.o_proj", layer, targets_with_regex)


def test_fused_module_matching():
    """Test matching for fused modules like qkv_proj."""
    layer = MockLinear()

    # Standard vLLM fused module mappings
    fused_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # Test qkv_proj fused matching
    assert is_match("layers.0.qkv_proj", layer, "layers.0.q_proj", fused=fused_mapping)
    assert is_match("layers.0.qkv_proj", layer, "layers.0.k_proj", fused=fused_mapping)
    assert is_match("layers.0.qkv_proj", layer, "layers.0.v_proj", fused=fused_mapping)
    assert not is_match(
        "layers.0.qkv_proj", layer, "layers.0.o_proj", fused=fused_mapping
    )

    # Test gate_up_proj fused matching
    assert is_match(
        "layers.0.gate_up_proj", layer, "layers.0.gate_proj", fused=fused_mapping
    )
    assert is_match(
        "layers.0.gate_up_proj", layer, "layers.0.up_proj", fused=fused_mapping
    )
    assert not is_match(
        "layers.0.gate_up_proj", layer, "layers.0.down_proj", fused=fused_mapping
    )

    # Test fused with regex patterns
    assert is_match("layers.0.qkv_proj", layer, "re:.*q_proj", fused=fused_mapping)
    assert is_match(
        "model.layers.5.qkv_proj", layer, "re:.*k_proj", fused=fused_mapping
    )


def test_ignore_matching():
    """Test ignore list functionality."""
    layer = MockLinear()

    # Test exact ignore
    ignore = ["layers.0.q_proj", "layers.1.k_proj"]
    assert not is_match("layers.0.q_proj", layer, "layers.0.q_proj", ignore=ignore)
    assert is_match("layers.2.q_proj", layer, "layers.2.q_proj", ignore=ignore)

    # Test regex ignore
    ignore_regex = ["re:.*\\.0\\..*", "re:.*\\.1\\..*"]
    assert not is_match(
        "layers.0.q_proj", layer, "layers.0.q_proj", ignore=ignore_regex
    )
    assert not is_match(
        "layers.1.k_proj", layer, "layers.1.k_proj", ignore=ignore_regex
    )
    assert is_match("layers.2.q_proj", layer, "layers.2.q_proj", ignore=ignore_regex)


def test_class_name_matching():
    """Test matching by module class name."""

    # Test with generic torch.nn modules
    linear = torch.nn.Linear(10, 10)
    assert is_match("any_name", linear, "Linear")

    attention = MockAttention()
    assert is_match("any_name", attention, "MockAttention")
    assert not is_match("any_name", attention, "Linear")


def test_vllm_linear_class_matching():
    """
    Test that vLLM LinearBase classes match 'Linear' target.

    This verifies the compressed-tensors library's built-in support for
    vLLM's LinearBase → Linear mapping.
    """
    # Note: We can't easily instantiate vLLM linear classes without initializing
    # distributed state, so we test the class hierarchy instead

    # Verify all vLLM linear classes inherit from LinearBase
    assert issubclass(ReplicatedLinear, LinearBase)
    assert issubclass(ColumnParallelLinear, LinearBase)
    assert issubclass(RowParallelLinear, LinearBase)

    # Verify LinearBase is in the MRO
    assert LinearBase in ReplicatedLinear.__mro__
    assert LinearBase in ColumnParallelLinear.__mro__
    assert LinearBase in RowParallelLinear.__mro__

    # The compressed-tensors library checks:
    # (cls.__name__ == "LinearBase" and target == "Linear")
    # This means when iterating through MRO, if any class is named "LinearBase"
    # and the target is "Linear", it will match.
    assert LinearBase.__name__ == "LinearBase"


def test_match_targets_ordering():
    """Test that match_targets returns results in priority order."""
    layer = MockLinear()

    targets = [
        "re:.*proj",  # regex
        "MockLinear",  # class name
        "layers.0.q_proj",  # exact name
    ]

    # match_targets returns matches ordered by specificity:
    # 1. exact name matches
    # 2. regex name matches
    # 3. class name matches
    matched = match_targets("layers.0.q_proj", layer, targets)

    # Should match both the exact name and the regex
    assert "layers.0.q_proj" in matched
    assert "re:.*proj" in matched
    assert "MockLinear" in matched

    # Exact match should come first
    assert matched[0] == "layers.0.q_proj"


def test_fused_with_regex_and_ignore():
    """Test complex scenario with fused modules, regex, and ignore."""
    layer = MockLinear()

    fused_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    # Match with regex target
    assert is_match("layers.0.qkv_proj", layer, "re:.*q_proj", fused=fused_mapping)

    # Match with regex target and ignore
    ignore = ["re:.*\\.0\\..*"]
    assert not is_match(
        "layers.0.qkv_proj",
        layer,
        "re:.*q_proj",
        ignore=ignore,
        fused=fused_mapping,
    )

    # Different layer should still match
    assert is_match(
        "layers.5.qkv_proj",
        layer,
        "re:.*q_proj",
        ignore=ignore,
        fused=fused_mapping,
    )


def test_empty_and_none_targets():
    """Test edge cases with empty or None targets."""
    layer = MockLinear()

    # Empty targets should not match
    assert not is_match("layers.0.q_proj", layer, [])
    assert not is_match("layers.0.q_proj", layer, tuple())

    # Empty ignore should allow all matches
    assert is_match("layers.0.q_proj", layer, "layers.0.q_proj", ignore=[])
    assert is_match("layers.0.q_proj", layer, "layers.0.q_proj", ignore=tuple())


def test_case_sensitivity():
    """Test that matching is case-sensitive."""
    layer = MockLinear()

    assert is_match("layers.0.Q_PROJ", layer, "layers.0.Q_PROJ")
    assert not is_match("layers.0.q_proj", layer, "layers.0.Q_PROJ")
    assert not is_match("layers.0.Q_PROJ", layer, "layers.0.q_proj")

    # Regex patterns are also case-sensitive by default
    assert is_match("layers.0.Q_PROJ", layer, "re:.*Q_PROJ")
    assert not is_match("layers.0.q_proj", layer, "re:.*Q_PROJ")


def test_special_characters_in_names():
    """Test matching with special characters in layer names."""
    layer = MockLinear()

    # Test with dots, underscores, numbers
    assert is_match(
        "model.layer_1.self_attn.q_proj", layer, "model.layer_1.self_attn.q_proj"
    )

    # Test regex escaping for dots
    assert is_match("model.layer.proj", layer, "re:model\\.layer\\.proj")
    # Without escaping, dot matches any character
    assert is_match("modelXlayerXproj", layer, "re:model.layer.proj")


def test_moe_expert_matching():
    """Test matching for MoE expert layers."""
    layer = MockLinear()

    # Test matching specific expert
    assert is_match(
        "layers.0.mlp.experts.0.gate_proj", layer, "layers.0.mlp.experts.0.gate_proj"
    )

    # Test regex to match all experts
    assert is_match(
        "layers.0.mlp.experts.0.gate_proj", layer, "re:.*experts\\.[0-9]+\\.gate_proj"
    )
    assert is_match(
        "layers.0.mlp.experts.7.gate_proj", layer, "re:.*experts\\.[0-9]+\\.gate_proj"
    )

    # Test with fused mapping for MoE
    fused_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    assert is_match(
        "layers.0.mlp.experts.0.gate_up_proj",
        layer,
        "re:.*experts\\.[0-9]+\\.gate_proj",
        fused=fused_mapping,
    )


def test_vllm_class_name_mapping_documentation():
    """
    Test that VLLM_CLASS_NAME_MAPPING is properly documented.

    This verifies that the mapping exists and is accessible for future
    extensions.
    """
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        VLLM_CLASS_NAME_MAPPING,
    )

    # Verify the mapping exists
    assert isinstance(VLLM_CLASS_NAME_MAPPING, dict)

    # Verify the Linear → LinearBase mapping is documented
    assert "Linear" in VLLM_CLASS_NAME_MAPPING
    assert VLLM_CLASS_NAME_MAPPING["Linear"] == "LinearBase"


def test_embedding_class_hierarchy():
    """
    Test vLLM embedding class hierarchy.

    This documents the class structure in case we need to add Embedding
    class name matching in the future.
    """
    # Verify embedding class hierarchy
    assert issubclass(ParallelLMHead, VocabParallelEmbedding)

    # Document class names for potential future mapping
    assert VocabParallelEmbedding.__name__ == "VocabParallelEmbedding"
    assert ParallelLMHead.__name__ == "ParallelLMHead"

    # These don't inherit from torch.nn.Embedding, so we would need
    # to add explicit mapping if configs use "Embedding" as a target
    assert not issubclass(VocabParallelEmbedding, torch.nn.Embedding)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
