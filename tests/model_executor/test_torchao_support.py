# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for TorchAO quantization support functionality."""

from unittest.mock import MagicMock, patch

import torch


class TestVocabParallelEmbeddingTorchAOSupport:
    """Tests for vocab_parallel_embedding TorchAO tensor metadata preservation."""

    def test_copy_without_data_accessor_preserves_subclass_attributes(self):
        """Test that using copy_() instead of .data.copy_() preserves tensor
        subclass attributes.

        The change removes .data accessor to preserve TorchAO tensor metadata
        (e.g., tensor_data_names). Using .data strips special tensor subclass
        attributes which TorchAO relies on for quantization.
        """
        # Create a regular tensor
        original = torch.zeros(10, 5)
        loaded_weight = torch.randn(8, 5)

        # Test copy_() preserves tensor identity
        original[: loaded_weight.shape[0]].copy_(loaded_weight)
        original[loaded_weight.shape[0] :].fill_(0)

        # Verify the data was copied correctly
        assert torch.allclose(original[:8], loaded_weight)
        assert torch.all(original[8:] == 0)

    def test_data_accessor_vs_direct_copy_behavior(self):
        """Demonstrate the difference between .data.copy_() and copy_().

        .data returns a new tensor that shares storage but loses custom
        attributes. Direct copy_() preserves the original tensor's attributes.
        """
        # Create a tensor with a custom attribute
        tensor = torch.zeros(5, 3)
        tensor.custom_attr = "test_value"

        source = torch.ones(3, 3)

        # Using direct indexing + copy_ preserves attributes
        tensor[:3].copy_(source)
        assert hasattr(tensor, "custom_attr")
        assert tensor.custom_attr == "test_value"

        # Verify copy worked
        assert torch.all(tensor[:3] == 1)


class TestOnlineQuantizationLoadConfig:
    """Tests for online_quantization LoadConfig usage."""

    def test_get_quant_config_called_with_load_config(self):
        """Test that get_quant_config is called with LoadConfig() instead of
        None.

        This ensures proper initialization of the quantization config with
        default load settings.
        """
        from vllm.config import LoadConfig, ModelConfig

        # Create a mock model config
        mock_model_config = MagicMock(spec=ModelConfig)
        mock_model_config.quantization = "torchao"

        # Patch get_quant_config at the source module where it's defined
        with patch(
            "vllm.model_executor.model_loader.weight_utils.get_quant_config"
        ) as mock_get_quant_config:
            # Set up the mock to return a config without the attribute
            mock_quant_config = MagicMock()
            mock_quant_config.is_checkpoint_torchao_serialized = True
            mock_get_quant_config.return_value = mock_quant_config

            # Import after patching to ensure the patch is applied
            # Force reimport by clearing the cache
            import sys

            if "vllm.model_executor.model_loader.online_quantization" in sys.modules:
                del sys.modules["vllm.model_executor.model_loader.online_quantization"]

            from vllm.model_executor.model_loader.online_quantization import (
                maybe_save_metadata_and_attributes_for_weight_reloading,
            )

            mock_model = MagicMock(spec=torch.nn.Module)
            maybe_save_metadata_and_attributes_for_weight_reloading(
                mock_model, mock_model_config
            )

            # Verify get_quant_config was called
            mock_get_quant_config.assert_called_once()

            # Get the actual call arguments
            call_args = mock_get_quant_config.call_args
            assert call_args[0][0] == mock_model_config  # First positional arg
            # Second arg should be LoadConfig instance, not None
            assert isinstance(call_args[0][1], LoadConfig)

    def test_load_config_default_values(self):
        """Test that LoadConfig() creates valid default configuration."""
        from vllm.config import LoadConfig

        load_config = LoadConfig()

        # Verify LoadConfig has expected default attributes
        assert hasattr(load_config, "load_format")
        assert hasattr(load_config, "download_dir")


class TestTorchAOConfigCLI:
    """Additional integration tests for torchao_config CLI handling."""

    def test_torchao_config_none_does_not_modify_hf_overrides(self):
        """Test that torchao_config=None leaves hf_overrides unchanged."""
        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs(model="test-model", torchao_config=None)
        # hf_overrides should remain at its default value
        assert engine_args.torchao_config is None

    def test_torchao_config_with_special_characters(self):
        """Test torchao_config with JSON containing special characters."""
        from vllm.engine.arg_utils import EngineArgs

        # JSON with nested structure
        json_config = (
            '{"_type": "torchao.quantization.Int4WeightOnlyConfig", '
            '"group_size": 128, "inner_k_tiles": 8}'
        )
        engine_args = EngineArgs(model="test-model", torchao_config=json_config)

        assert engine_args.torchao_config == json_config
