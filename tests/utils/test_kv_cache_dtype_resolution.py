# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for kv_cache_dtype resolution functions in torch_utils.

This tests the fix for issue #34752: Improve `--kv-cache-dtype` behavior when 
checkpoint specifies `kv_cache_quant_algo`.
"""

import pytest
import torch
from unittest.mock import Mock

from vllm.utils.torch_utils import (
    kv_cache_dtype_str_to_dtype,
    resolve_kv_cache_dtype_string,
    STR_DTYPE_TO_TORCH_DTYPE
)


def create_mock_model_config(model_dtype: torch.dtype, hf_quant_config=None):
    """Create a mock ModelConfig with optional quantization config."""
    mock_config = Mock()
    mock_config.dtype = model_dtype
    
    if hf_quant_config:
        mock_hf_config = Mock()
        mock_hf_config.quantization_config = hf_quant_config
        mock_config.hf_config = mock_hf_config
    else:
        mock_config.hf_config = None
    
    return mock_config


class TestKVCacheDtypeResolution:
    """Test kv_cache_dtype resolution behavior with various configurations."""

    def test_auto_with_checkpoint_fp8_quantization(self):
        """Test --kv-cache-dtype auto with checkpoint specifying FP8."""
        # Mock quantization config like nvidia/Qwen3-30B-A3B-NVFP4
        quant_config = {
            "quant_method": "modelopt",
            "quantization": {
                "kv_cache_quant_algo": "fp8"
            }
        }
        
        model_config = create_mock_model_config(torch.bfloat16, quant_config)
        
        # Should resolve to fp8_e4m3 (mapped from "fp8")
        resolved = resolve_kv_cache_dtype_string("auto", model_config)
        assert resolved == "fp8_e4m3"
        
        # kv_cache_dtype_str_to_dtype should use the resolved fp8_e4m3
        result_dtype = kv_cache_dtype_str_to_dtype("auto", model_config)
        expected_dtype = STR_DTYPE_TO_TORCH_DTYPE["fp8_e4m3"]
        assert result_dtype == expected_dtype

    def test_auto_without_checkpoint_quantization(self):
        """Test --kv-cache-dtype auto without checkpoint quantization."""
        # No quantization config
        model_config = create_mock_model_config(torch.bfloat16)
        
        # Should remain "auto" (no checkpoint quantization found)
        resolved = resolve_kv_cache_dtype_string("auto", model_config)
        assert resolved == "auto"
        
        # kv_cache_dtype_str_to_dtype should fall back to model_config.dtype
        result_dtype = kv_cache_dtype_str_to_dtype("auto", model_config)
        assert result_dtype == torch.bfloat16

    def test_explicit_override_with_checkpoint_quantization(self):
        """Test explicit dtype override even when checkpoint has quantization."""
        # Model with FP8 quantization config
        quant_config = {
            "quant_method": "modelopt",
            "quantization": {
                "kv_cache_quant_algo": "fp8"
            }
        }
        
        model_config = create_mock_model_config(torch.float16, quant_config)
        
        # Explicit bfloat16 should override checkpoint's fp8
        result_dtype = kv_cache_dtype_str_to_dtype("bfloat16", model_config)
        assert result_dtype == torch.bfloat16
        
        # Explicit fp8 should work
        result_dtype = kv_cache_dtype_str_to_dtype("fp8", model_config)
        assert result_dtype == STR_DTYPE_TO_TORCH_DTYPE["fp8"]

    def test_unit_test_compatibility_none_model_config(self):
        """Test that unit tests still work with None model_config."""
        # Should default to torch.half for unit tests
        result_dtype = kv_cache_dtype_str_to_dtype("auto", None)
        assert result_dtype == torch.half

    def test_various_quantization_formats(self):
        """Test different quantization config formats."""
        # Test different kv_cache_quant_algo locations
        test_configs = [
            # Direct in quantization dict
            {
                "quant_method": "modelopt",
                "quantization": {"kv_cache_quant_algo": "fp8"}
            },
            # Direct in root
            {
                "quant_method": "modelopt", 
                "kv_cache_quant_algo": "fp8"
            },
            # Using kv_cache_scheme (alternative name)
            {
                "quant_method": "modelopt",
                "quantization": {"kv_cache_scheme": "fp8"}
            }
        ]
        
        for quant_config in test_configs:
            model_config = create_mock_model_config(torch.bfloat16, quant_config)
            resolved = resolve_kv_cache_dtype_string("auto", model_config)
            assert resolved == "fp8_e4m3", f"Failed for config: {quant_config}"

    def test_unsupported_quantization_fallback(self):
        """Test fallback behavior for unsupported quantization algorithms."""
        # Unsupported quantization algorithm
        quant_config = {
            "quant_method": "modelopt",
            "quantization": {"kv_cache_quant_algo": "unsupported_algo"}
        }
        
        model_config = create_mock_model_config(torch.bfloat16, quant_config)
        
        # Should fall back to "auto" and then use model dtype
        resolved = resolve_kv_cache_dtype_string("auto", model_config)
        assert resolved == "auto"
        
        result_dtype = kv_cache_dtype_str_to_dtype("auto", model_config)
        assert result_dtype == torch.bfloat16

    def test_non_modelopt_quantization_methods(self):
        """Test that non-modelopt quantization methods are ignored."""
        quant_config = {
            "quant_method": "other_method",
            "kv_cache_quant_algo": "fp8"
        }
        
        model_config = create_mock_model_config(torch.bfloat16, quant_config)
        
        # Should not find any kv_cache_quant_algo for non-modelopt methods
        resolved = resolve_kv_cache_dtype_string("auto", model_config)
        assert resolved == "auto"
        
        result_dtype = kv_cache_dtype_str_to_dtype("auto", model_config)
        assert result_dtype == torch.bfloat16