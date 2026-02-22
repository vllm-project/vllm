#!/usr/bin/env python3
"""
Test script to verify the kv_cache_dtype "auto" behavior improvement.

This tests the fix for issue #34752 where --kv-cache-dtype auto should
use the checkpoint's specified kv_cache_quant_algo instead of falling 
back to model_config.dtype.
"""

import torch
from unittest.mock import Mock
from vllm.utils.torch_utils import resolve_kv_cache_dtype_string, get_kv_cache_quant_algo_string


def test_resolve_kv_cache_dtype_string():
    """Test the resolve_kv_cache_dtype_string function with different configs."""
    
    # Test case 1: No quantization config
    print("Test 1: No quantization config")
    mock_model_config = Mock()
    mock_model_config.hf_config = Mock()
    mock_model_config.hf_config.quantization_config = None
    
    result = resolve_kv_cache_dtype_string("auto", mock_model_config)
    assert result == "auto", f"Expected 'auto', got '{result}'"
    print("✓ No quantization config: returns 'auto'")
    
    # Test case 2: Quantization config with fp8
    print("\nTest 2: Quantization config with fp8")
    mock_model_config = Mock()
    mock_model_config.hf_config = Mock()
    quant_config = {
        "quant_method": "modelopt",
        "kv_cache_quant_algo": "fp8"
    }
    mock_model_config.hf_config.quantization_config = quant_config
    
    result = resolve_kv_cache_dtype_string("auto", mock_model_config)
    assert result == "fp8_e4m3", f"Expected 'fp8_e4m3', got '{result}'"
    print("✓ FP8 quantization config: returns 'fp8_e4m3'")
    
    # Test case 3: Explicit dtype override
    print("\nTest 3: Explicit dtype override") 
    result = resolve_kv_cache_dtype_string("bfloat16", mock_model_config)
    assert result == "bfloat16", f"Expected 'bfloat16', got '{result}'"
    print("✓ Explicit override: returns the specified dtype")
    
    # Test case 4: Get kv_cache_quant_algo_string directly
    print("\nTest 4: Direct kv_cache_quant_algo_string extraction")
    algo = get_kv_cache_quant_algo_string(quant_config)
    assert algo == "fp8_e4m3", f"Expected 'fp8_e4m3', got '{algo}'"
    print("✓ Direct extraction works")
    
    print("\n✅ All tests passed!")


def test_model_runner_integration():
    """Test that the model runner uses the fix correctly."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.config import ModelConfig, CacheConfig, VllmConfig
    from unittest.mock import patch
    
    print("\nIntegration Test: GPUModelRunner")
    
    # Create a mock VllmConfig with quantization
    mock_vllm_config = Mock(spec=VllmConfig)
    mock_model_config = Mock(spec=ModelConfig)
    mock_model_config.dtype = torch.bfloat16
    mock_model_config.hf_config = Mock()
    
    # Simulate a model with FP8 quantization config
    quant_config = {
        "quant_method": "modelopt",
        "kv_cache_quant_algo": "fp8"
    }
    mock_model_config.hf_config.quantization_config = quant_config
    
    mock_cache_config = Mock(spec=CacheConfig)
    mock_cache_config.cache_dtype = "auto"
    
    # Set up the mock config
    mock_vllm_config.model_config = mock_model_config
    mock_vllm_config.cache_config = mock_cache_config
    mock_vllm_config.compilation_config = Mock()
    mock_vllm_config.lora_config = None
    mock_vllm_config.load_config = Mock()
    mock_vllm_config.parallel_config = Mock()
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.scheduler_config.max_num_batched_tokens = 1024
    mock_vllm_config.scheduler_config.max_num_seqs = 256
    mock_vllm_config.scheduler_config.async_scheduling = False
    mock_vllm_config.speculative_config = None
    mock_vllm_config.observability_config = Mock()
    
    # Add other required attributes
    mock_model_config.get_vocab_size = Mock(return_value=50000)
    mock_model_config.max_model_len = 2048
    mock_model_config.get_inputs_embeds_size = Mock(return_value=768)
    mock_model_config.uses_mrope = False
    mock_model_config.logprobs_mode = False
    
    mock_vllm_config.parallel_config.pipeline_parallel_size = 1
    mock_vllm_config.parallel_config.decode_context_parallel_size = 1
    mock_vllm_config.parallel_config.data_parallel_size = 1
    mock_vllm_config.parallel_config.data_parallel_rank = 0
    mock_vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
    
    device = torch.device("cpu")  # Use CPU for testing
    
    try:
        runner = GPUModelRunner(mock_vllm_config, device)
        
        # Check that kv_cache_dtype was set correctly
        # With the fix, it should use fp8_e4m3 instead of the model dtype (bfloat16)
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        expected_dtype = STR_DTYPE_TO_TORCH_DTYPE["fp8_e4m3"]  # uint8
        actual_dtype = runner.kv_cache_dtype
        
        print(f"Model dtype: {mock_model_config.dtype}")
        print(f"Expected KV cache dtype: {expected_dtype}")
        print(f"Actual KV cache dtype: {actual_dtype}")
        
        if actual_dtype == expected_dtype:
            print("✅ Integration test passed: KV cache dtype uses quantization config")
            return True
        else:
            print(f"❌ Integration test failed: Expected {expected_dtype}, got {actual_dtype}")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_resolve_kv_cache_dtype_string()
    success = test_model_runner_integration()
    if not success:
        exit(1)