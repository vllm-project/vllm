#!/usr/bin/env python3
"""
Reproduction script for vllm issue #34752:
Improve `--kv-cache-dtype` behavior when checkpoint specifies `kv_cache_quant_algo`
"""

from vllm.config import ModelConfig
from vllm.utils.torch_utils import resolve_kv_cache_dtype_string


def create_mock_model_config_with_quant(kv_cache_quant_algo="fp8"):
    """Create a mock ModelConfig with quantization config like nvidia/Qwen3-30B-A3B-NVFP4"""
    
    class MockHFConfig:
        def __init__(self):
            self.quantization_config = {
                "quant_method": "modelopt",
                "kv_cache_quant_algo": kv_cache_quant_algo
            }
    
    class MockModelConfig:
        def __init__(self):
            self.hf_config = MockHFConfig()
            self.dtype = "bfloat16"  # Default model dtype
    
    return MockModelConfig()


def create_mock_model_config_without_quant():
    """Create a mock ModelConfig without quantization config"""
    
    class MockModelConfig:
        def __init__(self):
            self.hf_config = None
            self.dtype = "bfloat16"  # Default model dtype
    
    return MockModelConfig()


def test_current_behavior():
    """Test the current behavior to understand the bug"""
    print("=== Testing Current resolve_kv_cache_dtype_string Behavior ===\n")
    
    # Test case 1: Model with kv_cache_quant_algo="fp8"
    print("1. Model WITH kv_cache_quant_algo='fp8':")
    model_config_with_quant = create_mock_model_config_with_quant("fp8")
    
    print(f"  kv_cache_dtype='auto' -> {resolve_kv_cache_dtype_string('auto', model_config_with_quant)}")
    print(f"  kv_cache_dtype='fp8' -> {resolve_kv_cache_dtype_string('fp8', model_config_with_quant)}")
    print(f"  kv_cache_dtype='bfloat16' -> {resolve_kv_cache_dtype_string('bfloat16', model_config_with_quant)}")
    print()
    
    # Test case 2: Model without kv_cache_quant_algo
    print("2. Model WITHOUT kv_cache_quant_algo:")
    model_config_without_quant = create_mock_model_config_without_quant()
    
    print(f"  kv_cache_dtype='auto' -> {resolve_kv_cache_dtype_string('auto', model_config_without_quant)}")
    print(f"  kv_cache_dtype='fp8' -> {resolve_kv_cache_dtype_string('fp8', model_config_without_quant)}")
    print(f"  kv_cache_dtype='bfloat16' -> {resolve_kv_cache_dtype_string('bfloat16', model_config_without_quant)}")
    print()


if __name__ == "__main__":
    test_current_behavior()
    
    print("=== Expected vs Current Behavior Analysis ===\n")
    print("For models WITH kv_cache_quant_algo='fp8':")
    print("  kv_cache_dtype='auto' should return 'fp8_e4m3' (current: returns it)")
    print("  kv_cache_dtype='bfloat16' should return 'bfloat16' (current: returns 'bfloat16' but downstream fails)")
    print()
    print("For models WITHOUT kv_cache_quant_algo:")
    print("  kv_cache_dtype='auto' should return model_config.dtype (current: returns 'auto')")
    print("  This is the main bug - 'auto' should resolve to actual dtype, not stay as 'auto'")