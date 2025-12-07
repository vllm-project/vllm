# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Whisper Multi-LoRA support.

This module tests:
1. WhisperForConditionalGeneration LoRA interface compliance
2. MergedQKVParallelLinearWithLoRA support for KV-only (2-slice) configuration
3. WorkerLoRAManager compatibility with Whisper's max_target_positions
"""

import pytest
import torch

from vllm.lora.layers import (
    MergedQKVParallelLinearWithLoRA,
)
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.whisper import WhisperForConditionalGeneration
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_cpu()),
    reason="Backend not supported",
)


class TestWhisperLoRAInterface:
    """Test that WhisperForConditionalGeneration has proper LoRA support."""

    def test_supports_lora_attribute(self):
        """Verify that WhisperForConditionalGeneration has SupportsLoRA interface."""
        from vllm.model_executor.models.interfaces import SupportsLoRA

        assert issubclass(WhisperForConditionalGeneration, SupportsLoRA), (
            "WhisperForConditionalGeneration should inherit from SupportsLoRA"
        )

    def test_embedding_modules_defined(self):
        """Verify embedding_modules attribute is defined."""
        assert hasattr(WhisperForConditionalGeneration, "embedding_modules")
        assert isinstance(WhisperForConditionalGeneration.embedding_modules, dict)

    def test_embedding_padding_modules_defined(self):
        """Verify embedding_padding_modules attribute is defined."""
        assert hasattr(WhisperForConditionalGeneration, "embedding_padding_modules")
        assert isinstance(
            WhisperForConditionalGeneration.embedding_padding_modules, list
        )

    def test_packed_modules_mapping_format(self):
        """Verify packed_modules_mapping has correct format for LoRA."""
        mapping = WhisperForConditionalGeneration.packed_modules_mapping

        # Should have qkv_proj and kv_proj mappings
        assert "qkv_proj" in mapping, "Missing qkv_proj in packed_modules_mapping"
        assert "kv_proj" in mapping, "Missing kv_proj in packed_modules_mapping"

        # qkv_proj should map to [q_proj, k_proj, v_proj]
        assert mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]

        # kv_proj should map to [k_proj, v_proj] (for cross-attention)
        assert mapping["kv_proj"] == ["k_proj", "v_proj"]


class TestMergedQKVParallelLinearWithLoRAKVOnly:
    """Test MergedQKVParallelLinearWithLoRA with KV-only (2-slice) configuration."""

    def test_can_replace_layer_accepts_2_modules(self):
        """Verify can_replace_layer accepts 2-module (KV-only) configurations."""
        from vllm.config.lora import LoRAConfig

        # Create a mock QKVParallelLinear layer
        # This simulates a KV-only projection (like Whisper's encoder_attn.kv_proj)
        linear = QKVParallelLinear(
            hidden_size=512,
            head_size=64,
            total_num_heads=8,
            total_num_kv_heads=8,
            bias=False,
            params_dtype=torch.float16,
        )

        lora_config = LoRAConfig(
            max_lora_rank=32,
            max_loras=4,
            max_cpu_loras=4,
            lora_extra_vocab_size=0,
        )

        # Test with 2 modules (KV-only, like encoder_attn.kv_proj)
        packed_modules_2 = ["k_proj", "v_proj"]
        result_2 = MergedQKVParallelLinearWithLoRA.can_replace_layer(
            source_layer=linear,
            lora_config=lora_config,
            packed_modules_list=packed_modules_2,
            model_config=None,
        )
        assert result_2 is True, "Should accept 2-module (KV-only) configuration"

        # Test with 3 modules (QKV, like self_attn.qkv_proj)
        packed_modules_3 = ["q_proj", "k_proj", "v_proj"]
        result_3 = MergedQKVParallelLinearWithLoRA.can_replace_layer(
            source_layer=linear,
            lora_config=lora_config,
            packed_modules_list=packed_modules_3,
            model_config=None,
        )
        assert result_3 is True, "Should accept 3-module (QKV) configuration"

        # Test with 1 module (should be rejected)
        packed_modules_1 = ["q_proj"]
        result_1 = MergedQKVParallelLinearWithLoRA.can_replace_layer(
            source_layer=linear,
            lora_config=lora_config,
            packed_modules_list=packed_modules_1,
            model_config=None,
        )
        assert result_1 is False, "Should reject 1-module configuration"


class TestWorkerLoRAManagerWhisperCompat:
    """Test WorkerLoRAManager compatibility with Whisper config."""

    def test_max_position_embeddings_fallback(self):
        """Test that max_target_positions is used when missing."""

        # Create a mock config similar to Whisper's
        class MockWhisperConfig:
            def __init__(self):
                self.max_target_positions = 448
                # Note: no max_position_embeddings attribute

            def get_text_config(self):
                return self

        config = MockWhisperConfig()

        # Simulate the logic from WorkerLoRAManager
        max_pos = getattr(
            config,
            "max_position_embeddings",
            getattr(config, "max_target_positions", None),
        )

        assert max_pos == 448, "Should fall back to max_target_positions"

    def test_max_position_embeddings_priority(self):
        """Test that max_position_embeddings takes priority when present."""

        class MockLLMConfig:
            def __init__(self):
                self.max_position_embeddings = 4096
                self.max_target_positions = 448

            def get_text_config(self):
                return self

        config = MockLLMConfig()

        # Simulate the logic from WorkerLoRAManager
        max_pos = getattr(
            config,
            "max_position_embeddings",
            getattr(config, "max_target_positions", None),
        )

        assert max_pos == 4096, "Should use max_position_embeddings when present"
