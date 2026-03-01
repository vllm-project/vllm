#!/usr/bin/env python3
"""
Test script to verify the kv_cache_dtype fix for issue #34752.

This script tests all scenarios described in the GitHub issue:
1. --kv-cache-dtype auto with checkpoint specifying kv_cache_quant_algo
2. --kv-cache-dtype auto without checkpoint specifying kv_cache_quant_algo  
3. Explicit overrides (e.g., --kv-cache-dtype bfloat16)
"""

import sys
import torch
from typing import Dict, Any
from unittest.mock import Mock

# Add the current directory to Python path to import vllm modules
sys.path.insert(0, '/tmp/oss-vllm')

from vllm.utils.torch_utils import (
    kv_cache_dtype_str_to_dtype,
    resolve_kv_cache_dtype_string,
    get_kv_cache_quant_algo_string,
    STR_DTYPE_TO_TORCH_DTYPE
)


def create_mock_model_config(model_dtype: torch.dtype, hf_quant_config: Dict[str, Any] = None):
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


def test_scenario_1_auto_with_checkpoint_fp8():
    """Test: --kv-cache-dtype auto with checkpoint specifying FP8."""
    print("Testing Scenario 1: auto with checkpoint specifying FP8...")
    
    # Mock quantization config that specifies FP8 (like nvidia/Qwen3-30B-A3B-NVFP4)
    quant_config = {
        "quant_method": "modelopt",
        "quantization": {
            "kv_cache_quant_algo": "fp8"
        }
    }
    
    model_config = create_mock_model_config(torch.bfloat16, quant_config)
    
    # Test resolve_kv_cache_dtype_string
    resolved = resolve_kv_cache_dtype_string("auto", model_config)
    print(f"  resolve_kv_cache_dtype_string('auto') -> '{resolved}'")
    assert resolved == "fp8_e4m3", f"Expected 'fp8_e4m3', got '{resolved}'"
    
    # Test kv_cache_dtype_str_to_dtype
    result_dtype = kv_cache_dtype_str_to_dtype("auto", model_config)
    expected_dtype = STR_DTYPE_TO_TORCH_DTYPE["fp8_e4m3"]
    print(f"  kv_cache_dtype_str_to_dtype('auto') -> {result_dtype}")
    assert result_dtype == expected_dtype, f"Expected {expected_dtype}, got {result_dtype}"
    
    print("  ✅ PASSED: auto correctly uses checkpoint's FP8 quantization")


def test_scenario_2_auto_without_checkpoint():
    """Test: --kv-cache-dtype auto without checkpoint specifying kv_cache_quant_algo."""
    print("Testing Scenario 2: auto without checkpoint quantization...")
    
    model_config = create_mock_model_config(torch.bfloat16)  # No quantization config
    
    # Test resolve_kv_cache_dtype_string
    resolved = resolve_kv_cache_dtype_string("auto", model_config)
    print(f"  resolve_kv_cache_dtype_string('auto') -> '{resolved}'")
    assert resolved == "auto", f"Expected 'auto', got '{resolved}'"
    
    # Test kv_cache_dtype_str_to_dtype
    result_dtype = kv_cache_dtype_str_to_dtype("auto", model_config)
    expected_dtype = torch.bfloat16  # Should fall back to model_config.dtype
    print(f"  kv_cache_dtype_str_to_dtype('auto') -> {result_dtype}")
    assert result_dtype == expected_dtype, f"Expected {expected_dtype}, got {result_dtype}"
    
    print("  ✅ PASSED: auto correctly falls back to model dtype")


def test_scenario_3_explicit_override():
    """Test: explicit dtype override (e.g., --kv-cache-dtype bfloat16)."""
    print("Testing Scenario 3: explicit dtype override...")
    
    # Model with FP8 quantization config
    quant_config = {
        "quant_method": "modelopt", 
        "quantization": {
            "kv_cache_quant_algo": "fp8"
        }
    }
    
    model_config = create_mock_model_config(torch.float16, quant_config)
    
    # Test explicit override with bfloat16
    result_dtype = kv_cache_dtype_str_to_dtype("bfloat16", model_config)
    expected_dtype = torch.bfloat16
    print(f"  kv_cache_dtype_str_to_dtype('bfloat16') -> {result_dtype}")
    assert result_dtype == expected_dtype, f"Expected {expected_dtype}, got {result_dtype}"
    
    # Test explicit override with fp8
    result_dtype = kv_cache_dtype_str_to_dtype("fp8", model_config)
    expected_dtype = STR_DTYPE_TO_TORCH_DTYPE["fp8"]
    print(f"  kv_cache_dtype_str_to_dtype('fp8') -> {result_dtype}")
    assert result_dtype == expected_dtype, f"Expected {expected_dtype}, got {result_dtype}"
    
    print("  ✅ PASSED: explicit overrides work correctly")


def test_scenario_4_unit_test_compatibility():
    """Test: ensure unit tests still work (None model_config case)."""
    print("Testing Scenario 4: unit test compatibility...")
    
    # Test with None model_config (unit test scenario)
    result_dtype = kv_cache_dtype_str_to_dtype("auto", None)
    expected_dtype = torch.half
    print(f"  kv_cache_dtype_str_to_dtype('auto', None) -> {result_dtype}")
    assert result_dtype == expected_dtype, f"Expected {expected_dtype}, got {result_dtype}"
    
    print("  ✅ PASSED: unit test compatibility maintained")


def run_all_tests():
    """Run all test scenarios."""
    print("=" * 60)
    print("Testing kv_cache_dtype fix for issue #34752")
    print("=" * 60)
    
    try:
        test_scenario_1_auto_with_checkpoint_fp8()
        print()
        test_scenario_2_auto_without_checkpoint()
        print()
        test_scenario_3_explicit_override()
        print()
        test_scenario_4_unit_test_compatibility()
        print()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("The fix correctly handles all scenarios from issue #34752:")
        print("  ✅ auto + checkpoint FP8 quantization -> uses FP8")
        print("  ✅ auto + no checkpoint quantization -> uses model dtype")
        print("  ✅ explicit overrides -> uses specified dtype")
        print("  ✅ unit test compatibility -> maintained")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)