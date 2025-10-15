#!/usr/bin/env python3
"""
Simple test script for CheXagent model implementation
"""

def test_import():
    """Test that we can import the CheXagent model"""
    try:
        from vllm.model_executor.models.chexagent import CheXagentForConditionalGeneration
        print("✓ CheXagent model imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import CheXagent model: {e}")
        return False

def test_registry():
    """Test that CheXagent is registered in the model registry"""
    try:
        from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
        if "CheXagentForConditionalGeneration" in _MULTIMODAL_MODELS:
            print("✓ CheXagent is registered in the model registry")
            return True
        else:
            print("✗ CheXagent is not registered in the model registry")
            return False
    except Exception as e:
        print(f"✗ Failed to check registry: {e}")
        return False

def test_multimodal_registry():
    """Test that CheXagent is registered in the multimodal registry"""
    try:
        from vllm.multimodal import MULTIMODAL_REGISTRY
        from vllm.model_executor.models.chexagent import CheXagentForConditionalGeneration
        
        if MULTIMODAL_REGISTRY._processor_factories.contains(CheXagentForConditionalGeneration, strict=True):
            print("✓ CheXagent is registered in the multimodal registry")
            return True
        else:
            print("✗ CheXagent is not registered in the multimodal registry")
            return False
    except Exception as e:
        print(f"✗ Failed to check multimodal registry: {e}")
        return False

def test_model_architecture():
    """Test that we can resolve the model architecture"""
    try:
        from vllm.config import ModelConfig
        from vllm.model_executor.model_loader import get_model_architecture
        
        model_config = ModelConfig(
            "StanfordAIMI/CheXagent-8b",
            task="auto",
            trust_remote_code=True,
            seed=0,
            dtype="auto",
        )
        
        model_cls, arch = get_model_architecture(model_config)
        if arch == "CheXagentForConditionalGeneration":
            print("✓ Model architecture resolved correctly")
            return True
        else:
            print(f"✗ Unexpected architecture: {arch}")
            return False
    except Exception as e:
        print(f"✗ Failed to resolve model architecture: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing CheXagent model implementation...")
    print("=" * 50)
    
    tests = [
        test_import,
        test_registry,
        test_multimodal_registry,
        test_model_architecture,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! CheXagent implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main() 