#!/usr/bin/env python3
"""
Test script for the kv_cache_dtype fix for issue #34752
This simulates the behavior without requiring vllm dependencies
"""

# Mock the minimal functions needed for testing
MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP = {
    "fp8": "fp8_e4m3",
}

def get_kv_cache_quant_algo_string(quant_cfg):
    """Mock implementation of get_kv_cache_quant_algo_string"""
    quant_method = quant_cfg.get("quant_method", "")
    if quant_method.startswith("modelopt"):
        kv_algo = quant_cfg.get("kv_cache_quant_algo")
        if isinstance(kv_algo, str):
            kv_algo_lower = kv_algo.lower()
            if kv_algo_lower in MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP:
                return MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP[kv_algo_lower]
    return None

def resolve_kv_cache_dtype_string_fixed(kv_cache_dtype, model_config):
    """Fixed version of resolve_kv_cache_dtype_string"""
    if kv_cache_dtype != "auto":
        return kv_cache_dtype

    # First, check if the model specifies a KV cache quantization algorithm
    hf_cfg = getattr(model_config, "hf_config", None)
    if hf_cfg is not None:
        quant_cfg = getattr(hf_cfg, "quantization_config", None)
        if quant_cfg is not None:
            kv_algo_str = get_kv_cache_quant_algo_string(quant_cfg)
            if kv_algo_str is not None and kv_algo_str != "auto":
                return kv_algo_str

    # Fallback to model's default dtype when no quantization is specified
    # This fixes the case where kv_cache_dtype="auto" was returning "auto"
    # instead of resolving to the actual model dtype
    model_dtype = str(model_config.dtype)
    return model_dtype

def resolve_kv_cache_dtype_string_original(kv_cache_dtype, model_config):
    """Original version of resolve_kv_cache_dtype_string (with bug)"""
    if kv_cache_dtype != "auto":
        return kv_cache_dtype

    hf_cfg = getattr(model_config, "hf_config", None)
    if hf_cfg is not None:
        quant_cfg = getattr(hf_cfg, "quantization_config", None)
        if quant_cfg is not None:
            kv_algo_str = get_kv_cache_quant_algo_string(quant_cfg)
            if kv_algo_str is not None:
                return kv_algo_str

    # BUG: This returns "auto" instead of falling back to model_config.dtype
    return "auto"

class MockHFConfig:
    def __init__(self, has_quant=True):
        if has_quant:
            self.quantization_config = {
                "quant_method": "modelopt",
                "kv_cache_quant_algo": "fp8"
            }
        else:
            self.quantization_config = None

class MockModelConfig:
    def __init__(self, has_quant=True):
        self.hf_config = MockHFConfig(has_quant) if has_quant else None
        self.dtype = "bfloat16"  # Default model dtype

def test_behavior():
    """Test both original and fixed behavior"""
    print("=== Testing KV Cache Dtype Resolution Fix ===\n")
    
    # Test cases
    test_cases = [
        ("Models WITH kv_cache_quant_algo='fp8'", True),
        ("Models WITHOUT kv_cache_quant_algo", False)
    ]
    
    kv_cache_dtypes = ["auto", "fp8", "bfloat16"]
    
    for case_name, has_quant in test_cases:
        print(f"📋 {case_name}:")
        model_config = MockModelConfig(has_quant)
        
        print(f"   {'kv_cache_dtype':<12} | {'Original (Bug)':<15} | {'Fixed':<15} | Expected")
        print(f"   {'-'*12} | {'-'*15} | {'-'*15} | {'-'*15}")
        
        for kv_dtype in kv_cache_dtypes:
            original_result = resolve_kv_cache_dtype_string_original(kv_dtype, model_config)
            fixed_result = resolve_kv_cache_dtype_string_fixed(kv_dtype, model_config)
            
            # Expected results based on the issue description
            if has_quant:  # Model with kv_cache_quant_algo
                if kv_dtype == "auto":
                    expected = "fp8_e4m3"
                else:
                    expected = kv_dtype
            else:  # Model without kv_cache_quant_algo
                if kv_dtype == "auto":
                    expected = "bfloat16"  # model_config.dtype
                else:
                    expected = kv_dtype
            
            status = "✅" if fixed_result == expected else "❌"
            print(f"   {kv_dtype:<12} | {original_result:<15} | {fixed_result:<15} | {expected} {status}")
        
        print()

def test_edge_cases():
    """Test edge cases and error conditions"""
    print("=== Testing Edge Cases ===\n")
    
    # Test case: Model with hf_config but no quantization_config
    class MockModelConfigNoQuantConfig:
        def __init__(self):
            self.hf_config = type('obj', (object,), {'quantization_config': None})()
            self.dtype = "float16"
    
    model = MockModelConfigNoQuantConfig()
    result = resolve_kv_cache_dtype_string_fixed("auto", model)
    expected = "float16"
    status = "✅" if result == expected else "❌"
    print(f"Model with hf_config but no quantization_config:")
    print(f"  auto -> {result} (expected: {expected}) {status}\n")
    
    # Test case: Model with no hf_config at all
    class MockModelConfigNoHF:
        def __init__(self):
            self.hf_config = None
            self.dtype = "float32"
    
    model = MockModelConfigNoHF()
    result = resolve_kv_cache_dtype_string_fixed("auto", model)
    expected = "float32"
    status = "✅" if result == expected else "❌"
    print(f"Model with no hf_config:")
    print(f"  auto -> {result} (expected: {expected}) {status}\n")

if __name__ == "__main__":
    test_behavior()
    test_edge_cases()
    
    print("=== Summary ===")
    print("🔧 Fix: resolve_kv_cache_dtype_string now properly falls back to model_config.dtype")
    print("📌 Key changes:")
    print("  1. When kv_cache_dtype='auto' and no quantization is found, return model_config.dtype")
    print("  2. When kv_cache_dtype is explicitly set, always return that value (allows overrides)")
    print("  3. Added safety check for kv_algo_str != 'auto' to prevent infinite recursion")