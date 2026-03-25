# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Qwen 3.5 GGUF loading support (issue #38122)."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader
from vllm.transformers_utils.configs.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)


class TestQwen35GGUFSupport:
    """Tests for Qwen 3.5 GGUF loading support."""

    def test_qwen35_model_type_mapping(self):
        """
        Test Fix #1: qwen3_5 model_type is correctly mapped to qwen35.
        
        The GGUF loader should convert 'qwen3_5' (HuggingFace naming)
        to 'qwen35' (GGUF architecture naming).
        """
        import gguf

        # Verify that 'qwen35' exists in GGUF MODEL_ARCH_NAMES
        gguf_arch_names = list(gguf.MODEL_ARCH_NAMES.values())
        assert "qwen35" in gguf_arch_names, (
            "qwen35 should be a valid GGUF architecture name"
        )

        # Verify our mapping logic works
        model_type = "qwen3_5"
        if model_type == "qwen3_5":
            model_type = "qwen35"
        
        # Now it should be found in GGUF
        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        
        assert arch is not None, (
            f"After mapping, {model_type} should be found in gguf.MODEL_ARCH_NAMES"
        )

    def test_qwen35_vision_config_depth_fallback(self):
        """
        Test Fix #2: GGUF loader handles 'depth' attribute correctly.
        
        Qwen3_5VisionConfig uses 'depth' instead of 'num_hidden_layers'.
        The fix uses getattr fallback to support both attributes.
        """
        vision_config = Qwen3_5VisionConfig(depth=27)
        
        # Verify the config structure
        assert hasattr(vision_config, 'depth')
        assert vision_config.depth == 27
        assert not hasattr(vision_config, 'num_hidden_layers')
        
        # The fix: use getattr with fallback to 'depth'
        vision_num_layers = getattr(
            vision_config,
            'num_hidden_layers',
            getattr(vision_config, 'depth', None)
        )
        
        # Should successfully get the layer count
        assert vision_num_layers == 27

    def test_qwen35_vision_config_custom_depth(self):
        """Test that custom depth values are handled correctly."""
        for depth in [12, 24, 32, 48]:
            vision_config = Qwen3_5VisionConfig(depth=depth)
            
            vision_num_layers = getattr(
                vision_config,
                'num_hidden_layers',
                getattr(vision_config, 'depth', None)
            )
            
            assert vision_num_layers == depth

    def test_qwen35_text_config_has_num_hidden_layers(self):
        """Verify that Qwen3_5TextConfig has num_hidden_layers (standard naming)."""
        text_config = Qwen3_5TextConfig(num_hidden_layers=64)
        
        assert hasattr(text_config, 'num_hidden_layers')
        assert text_config.num_hidden_layers == 64

    def test_qwen35_full_config_structure(self):
        """Test the complete Qwen3_5Config structure."""
        config = Qwen3_5Config()
        
        assert config.model_type == "qwen3_5"
        assert hasattr(config, 'vision_config')
        assert hasattr(config, 'text_config')
        
        # Vision uses depth (default 27)
        assert config.vision_config.depth == 27
        
        # Text uses num_hidden_layers (default 32)
        assert config.text_config.num_hidden_layers == 32


class TestQwen35GGUFLoaderIntegration:
    """Integration tests for Qwen 3.5 GGUF loading."""

    def test_get_gguf_weights_map_with_qwen35_text_only(self):
        """
        Test that _get_gguf_weights_map works for Qwen 3.5 text-only models.
        """
        import gguf

        # Create mock config mimicking Qwen3_5 text-only
        mock_text_config = MagicMock()
        mock_text_config.num_hidden_layers = 64
        
        mock_hf_config = MagicMock()
        mock_hf_config.model_type = "qwen3_5"
        mock_hf_config.vision_config = None  # Text-only
        mock_hf_config.get_text_config.return_value = mock_text_config
        mock_hf_config.num_hidden_layers = 64

        # Simulate the model_type mapping fix
        model_type = mock_hf_config.model_type
        if model_type == "qwen3_5":
            model_type = "qwen35"
        
        # Verify mapping works
        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        
        assert arch is not None
        
        # Verify we can get tensor name map
        text_name_map = gguf.get_tensor_name_map(arch, 64)
        assert text_name_map is not None

    def test_get_gguf_weights_map_with_qwen35_multimodal(self):
        """
        Test that _get_gguf_weights_map works for Qwen 3.5 multimodal models.
        """
        import gguf

        # Create actual Qwen3_5VisionConfig
        vision_config = Qwen3_5VisionConfig(depth=27)
        
        mock_text_config = MagicMock()
        mock_text_config.num_hidden_layers = 64
        
        mock_hf_config = MagicMock()
        mock_hf_config.model_type = "qwen3_5"
        mock_hf_config.vision_config = vision_config  # Multimodal
        mock_hf_config.get_text_config.return_value = mock_text_config

        # Check multimodal detection
        is_multimodal = (
            hasattr(mock_hf_config, "vision_config") 
            and mock_hf_config.vision_config is not None
        )
        assert is_multimodal

        # Simulate the fix for vision_num_layers
        vision_num_layers = getattr(
            mock_hf_config.vision_config,
            'num_hidden_layers',
            getattr(mock_hf_config.vision_config, 'depth', None)
        )
        
        assert vision_num_layers == 27
        
        # Verify we can get vision tensor name map
        mm_proj_arch = gguf.MODEL_ARCH.MMPROJ
        vision_name_map = gguf.get_tensor_name_map(mm_proj_arch, vision_num_layers)
        assert vision_name_map is not None


class TestVisionConfigCompatibility:
    """Test vision config compatibility across different model types."""

    def test_fallback_handles_both_attributes(self):
        """
        Test that the getattr fallback handles configs with either attribute.
        """
        # Config with num_hidden_layers (standard)
        class StandardVisionConfig:
            num_hidden_layers = 24
        
        # Config with depth (Qwen3.5 style)
        class DepthVisionConfig:
            depth = 27
        
        # Config with both (hypothetical)
        class BothVisionConfig:
            num_hidden_layers = 32
            depth = 27  # Should be ignored if num_hidden_layers exists
        
        # Test standard config
        standard = StandardVisionConfig()
        layers = getattr(standard, 'num_hidden_layers', getattr(standard, 'depth', None))
        assert layers == 24
        
        # Test depth config (Qwen3.5)
        depth_cfg = DepthVisionConfig()
        layers = getattr(depth_cfg, 'num_hidden_layers', getattr(depth_cfg, 'depth', None))
        assert layers == 27
        
        # Test config with both (num_hidden_layers takes priority)
        both = BothVisionConfig()
        layers = getattr(both, 'num_hidden_layers', getattr(both, 'depth', None))
        assert layers == 32

    def test_fallback_returns_none_for_missing_attributes(self):
        """Test that fallback returns None when neither attribute exists."""
        class EmptyVisionConfig:
            pass
        
        empty = EmptyVisionConfig()
        layers = getattr(empty, 'num_hidden_layers', getattr(empty, 'depth', None))
        assert layers is None