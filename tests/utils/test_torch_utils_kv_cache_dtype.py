"""Tests for KV cache dtype resolution functions in torch_utils.py"""

import pytest

from vllm.utils.torch_utils import (
    get_kv_cache_quant_algo_string,
    resolve_kv_cache_dtype_string,
)


class MockHFConfig:
    """Mock HuggingFace config for testing"""
    
    def __init__(self, quantization_config=None):
        self.quantization_config = quantization_config


class MockModelConfig:
    """Mock model config for testing"""
    
    def __init__(self, dtype="bfloat16", hf_config=None):
        self.dtype = dtype
        self.hf_config = hf_config


class TestGetKVCacheQuantAlgoString:
    """Tests for get_kv_cache_quant_algo_string function"""
    
    def test_modelopt_fp8_quantization(self):
        """Test FP8 quantization detection in ModelOpt configs"""
        quant_cfg = {
            "quant_method": "modelopt",
            "kv_cache_quant_algo": "fp8"
        }
        result = get_kv_cache_quant_algo_string(quant_cfg)
        assert result == "fp8_e4m3"
    
    def test_no_quantization_method(self):
        """Test config without quantization method"""
        quant_cfg = {"some_other_key": "value"}
        result = get_kv_cache_quant_algo_string(quant_cfg)
        assert result is None
    
    def test_non_modelopt_quantization(self):
        """Test non-ModelOpt quantization methods"""
        quant_cfg = {
            "quant_method": "other_method",
            "kv_cache_quant_algo": "fp8"
        }
        result = get_kv_cache_quant_algo_string(quant_cfg)
        assert result is None
    
    def test_unknown_kv_cache_algo(self):
        """Test unknown KV cache algorithm (should return None)"""
        quant_cfg = {
            "quant_method": "modelopt", 
            "kv_cache_quant_algo": "unknown_algo"
        }
        result = get_kv_cache_quant_algo_string(quant_cfg)
        assert result == "auto"  # Falls back to "auto" for unknown algorithms


class TestResolveKVCacheDtypeString:
    """Tests for resolve_kv_cache_dtype_string function - addresses issue #34752"""
    
    def test_explicit_dtype_passthrough(self):
        """Test that explicit dtype values are passed through unchanged"""
        model_config = MockModelConfig(dtype="bfloat16")
        
        # Test various explicit dtypes
        assert resolve_kv_cache_dtype_string("bfloat16", model_config) == "bfloat16"
        assert resolve_kv_cache_dtype_string("fp8", model_config) == "fp8"
        assert resolve_kv_cache_dtype_string("float16", model_config) == "float16"
    
    def test_auto_with_fp8_quantization(self):
        """Test auto resolution with FP8 quantization config"""
        hf_config = MockHFConfig({
            "quant_method": "modelopt",
            "kv_cache_quant_algo": "fp8"
        })
        model_config = MockModelConfig(dtype="bfloat16", hf_config=hf_config)
        
        result = resolve_kv_cache_dtype_string("auto", model_config)
        assert result == "fp8_e4m3"
    
    def test_auto_without_quantization_fallback_to_model_dtype(self):
        """Test auto resolution fallback to model dtype (fixes issue #34752)"""
        # Test case 1: Model with no hf_config
        model_config = MockModelConfig(dtype="bfloat16", hf_config=None)
        result = resolve_kv_cache_dtype_string("auto", model_config)
        assert result == "bfloat16"
        
        # Test case 2: Model with hf_config but no quantization_config
        hf_config = MockHFConfig(quantization_config=None)
        model_config = MockModelConfig(dtype="float16", hf_config=hf_config)
        result = resolve_kv_cache_dtype_string("auto", model_config)
        assert result == "float16"
        
        # Test case 3: Model with quantization_config but no supported algorithm
        hf_config = MockHFConfig({"quant_method": "other_method"})
        model_config = MockModelConfig(dtype="float32", hf_config=hf_config)
        result = resolve_kv_cache_dtype_string("auto", model_config)
        assert result == "float32"
    
    def test_auto_with_unknown_quant_algo_fallback(self):
        """Test auto resolution with unknown quantization algorithm"""
        hf_config = MockHFConfig({
            "quant_method": "modelopt",
            "kv_cache_quant_algo": "unknown_algo"
        })
        model_config = MockModelConfig(dtype="bfloat16", hf_config=hf_config)
        
        # Should fall back to model dtype since unknown algo returns "auto"
        result = resolve_kv_cache_dtype_string("auto", model_config)
        assert result == "bfloat16"
    
    def test_explicit_dtype_override_with_quantization(self):
        """Test explicit dtype override when model has quantization (fixes issue #34752)"""
        hf_config = MockHFConfig({
            "quant_method": "modelopt",
            "kv_cache_quant_algo": "fp8"
        })
        model_config = MockModelConfig(dtype="bfloat16", hf_config=hf_config)
        
        # Should allow explicit override of quantized model's KV cache dtype
        result = resolve_kv_cache_dtype_string("bfloat16", model_config)
        assert result == "bfloat16"
        
        result = resolve_kv_cache_dtype_string("float16", model_config)
        assert result == "float16"
    
    @pytest.mark.parametrize("model_dtype", ["bfloat16", "float16", "float32"])
    def test_different_model_dtypes(self, model_dtype):
        """Test auto resolution with different model dtypes"""
        model_config = MockModelConfig(dtype=model_dtype, hf_config=None)
        result = resolve_kv_cache_dtype_string("auto", model_config)
        assert result == model_dtype