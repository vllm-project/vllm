"""Test kv_cache_dtype "auto" behavior with quantization configs."""

import pytest
import torch
from unittest.mock import Mock

from vllm.utils.torch_utils import (
    resolve_kv_cache_dtype_string,
    get_kv_cache_quant_algo_string,
    STR_DTYPE_TO_TORCH_DTYPE,
)


class TestKVCacheDtypeAutoResolve:
    """Test cases for kv_cache_dtype auto resolution with quantization configs."""

    def test_resolve_no_quantization_config(self):
        """When no quantization config, should return 'auto'."""
        mock_model_config = Mock()
        mock_model_config.hf_config = Mock()
        mock_model_config.hf_config.quantization_config = None

        result = resolve_kv_cache_dtype_string("auto", mock_model_config)
        assert result == "auto"

    def test_resolve_with_fp8_quantization(self):
        """When quantization config specifies fp8, should return fp8_e4m3."""
        mock_model_config = Mock()
        mock_model_config.hf_config = Mock()
        quant_config = {
            "quant_method": "modelopt",
            "kv_cache_quant_algo": "fp8"
        }
        mock_model_config.hf_config.quantization_config = quant_config

        result = resolve_kv_cache_dtype_string("auto", mock_model_config)
        assert result == "fp8_e4m3"

    def test_resolve_explicit_dtype_override(self):
        """When explicitly set, should return that dtype regardless of config."""
        mock_model_config = Mock()
        mock_model_config.hf_config = Mock()
        quant_config = {
            "quant_method": "modelopt",
            "kv_cache_quant_algo": "fp8"
        }
        mock_model_config.hf_config.quantization_config = quant_config

        # Explicit override should take precedence
        result = resolve_kv_cache_dtype_string("bfloat16", mock_model_config)
        assert result == "bfloat16"

    def test_kv_cache_quant_algo_string_extraction(self):
        """Test direct extraction of kv_cache_quant_algo string."""
        quant_config = {
            "quant_method": "modelopt",
            "kv_cache_quant_algo": "fp8"
        }

        result = get_kv_cache_quant_algo_string(quant_config)
        assert result == "fp8_e4m3"

    def test_kv_cache_quant_algo_dict_format(self):
        """Test extraction when kv_cache_quant_algo is a dict."""
        quant_config = {
            "quant_method": "modelopt",
            "kv_cache_quant_algo": {
                "dynamic": False,
                "num_bits": 8,
                "type": "float"
            }
        }

        result = get_kv_cache_quant_algo_string(quant_config)
        assert result == "fp8_e4m3"

    def test_unsupported_quant_algo_fallback(self):
        """Test that unsupported algorithms fall back to 'auto'."""
        quant_config = {
            "quant_method": "modelopt",
            "kv_cache_quant_algo": "unsupported_algo"
        }

        result = get_kv_cache_quant_algo_string(quant_config)
        assert result == "auto"

    def test_no_quant_method(self):
        """Test when quantization config exists but no modelopt method."""
        quant_config = {
            "some_other_method": "value",
            "kv_cache_quant_algo": "fp8"
        }

        result = get_kv_cache_quant_algo_string(quant_config)
        assert result is None

    def test_nested_quantization_config(self):
        """Test extraction from nested quantization structure."""
        quant_config = {
            "quant_method": "modelopt",
            "quantization": {
                "kv_cache_quant_algo": "fp8"
            }
        }

        result = get_kv_cache_quant_algo_string(quant_config)
        assert result == "fp8_e4m3"


# Test integration with the expected behavior from issue #34752
class TestKVCacheDtypeIssue34752:
    """Test cases specific to issue #34752 requirements."""

    def test_auto_with_checkpoint_fp8(self):
        """
        Test case from issue: --kv-cache-dtype auto should use checkpoint's FP8.
        """
        mock_model_config = Mock()
        mock_model_config.hf_config = Mock()
        mock_model_config.dtype = torch.bfloat16  # Model dtype

        # Checkpoint specifies FP8
        quant_config = {
            "quant_method": "modelopt",
            "kv_cache_quant_algo": "fp8"
        }
        mock_model_config.hf_config.quantization_config = quant_config

        # With "auto", should use checkpoint's FP8
        result = resolve_kv_cache_dtype_string("auto", mock_model_config)
        assert result == "fp8_e4m3"

    def test_explicit_override_checkpoint(self):
        """
        Test case from issue: --kv-cache-dtype bfloat16 should override checkpoint.
        """
        mock_model_config = Mock()
        mock_model_config.hf_config = Mock()

        # Checkpoint specifies FP8
        quant_config = {
            "quant_method": "modelopt",
            "kv_cache_quant_algo": "fp8"
        }
        mock_model_config.hf_config.quantization_config = quant_config

        # Explicit bfloat16 should override
        result = resolve_kv_cache_dtype_string("bfloat16", mock_model_config)
        assert result == "bfloat16"

    def test_auto_fallback_no_checkpoint_config(self):
        """
        Test case: --kv-cache-dtype auto should fall back to model dtype when
        no checkpoint quantization config is present.
        """
        mock_model_config = Mock()
        mock_model_config.hf_config = Mock()
        mock_model_config.hf_config.quantization_config = None

        # Should return "auto" for downstream handling
        result = resolve_kv_cache_dtype_string("auto", mock_model_config)
        assert result == "auto"