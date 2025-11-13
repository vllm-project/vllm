#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test actual vLLM integration with model registration.

This test requires proper vLLM setup with LD_PRELOAD.
"""

# Import the models from our test file
import sys

sys.path.insert(0, "tests")
from test_generic_model_support import (
    GenericTransformerForCausalLMVLLM,
)


def test_model_registration():
    """Test actual model registration with vLLM."""
    print("=" * 70)
    print("Testing Model Registration with vLLM")
    print("=" * 70)

    # Import vLLM registry
    from vllm.model_executor.models.registry import ModelRegistry

    print("\n1. Registering GenericTransformerForCausalLM...")

    # Register the model
    ModelRegistry.register_model(
        "GenericTransformerForCausalLM",
        GenericTransformerForCausalLMVLLM,
    )

    print("   ✓ Model registered successfully")

    # Verify it's in the registry
    supported_archs = ModelRegistry.get_supported_archs()

    if "GenericTransformerForCausalLM" in supported_archs:
        print("   ✓ Model found in registry")
    else:
        print("   ✗ Model NOT found in registry")
        return False

    print(f"\n2. Total supported architectures: {len(supported_archs)}")
    print("   (including our custom model)")

    return True


def test_lazy_registration():
    """Test lazy registration with string format."""
    print("\n" + "=" * 70)
    print("Testing Lazy Registration (module:class format)")
    print("=" * 70)

    from vllm.model_executor.models.registry import ModelRegistry

    print("\n1. Registering with lazy import string...")

    # Register using the lazy string format
    ModelRegistry.register_model(
        "GenericTransformerLazy",
        "test_generic_model_support:GenericTransformerForCausalLMVLLM",
    )

    print("   ✓ Lazy registration successful")

    # Verify it's registered
    if "GenericTransformerLazy" in ModelRegistry.get_supported_archs():
        print("   ✓ Lazy model found in registry")
        return True
    else:
        print("   ✗ Lazy model NOT found in registry")
        return False


def test_model_inspection():
    """Test that vLLM can inspect our model."""
    print("\n" + "=" * 70)
    print("Testing Model Inspection")
    print("=" * 70)

    from vllm.model_executor.models.registry import ModelRegistry

    # Register if not already
    if "GenericTransformerForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "GenericTransformerForCausalLM",
            GenericTransformerForCausalLMVLLM,
        )

    print("\n1. Attempting to load model class...")

    try:
        # Try to load the model class
        model_cls = ModelRegistry._try_load_model_cls("GenericTransformerForCausalLM")

        if model_cls is not None:
            print(f"   ✓ Model class loaded: {model_cls.__name__}")
            print(f"   ✓ Module: {model_cls.__module__}")

            # Check attributes
            print("\n2. Checking model attributes:")
            print(f"   - supports_pp: {getattr(model_cls, 'supports_pp', 'not set')}")
            print(
                f"   - supports_multimodal: "
                f"{getattr(model_cls, 'supports_multimodal', 'not set')}"
            )

            return True
        else:
            print("   ✗ Model class failed to load")
            return False

    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("vLLM Integration Tests")
    print("Requires: LD_PRELOAD with cublas libraries")
    print("=" * 70)

    results = []

    # Test 1: Direct registration
    try:
        results.append(("Model Registration", test_model_registration()))
    except Exception as e:
        print(f"\n✗ Model Registration failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Model Registration", False))

    # Test 2: Lazy registration
    try:
        results.append(("Lazy Registration", test_lazy_registration()))
    except Exception as e:
        print(f"\n✗ Lazy Registration failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Lazy Registration", False))

    # Test 3: Model inspection
    try:
        results.append(("Model Inspection", test_model_inspection()))
    except Exception as e:
        print(f"\n✗ Model Inspection failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Model Inspection", False))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")

    passed = sum(1 for _, p in results if p)
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 70)

    return all(p for _, p in results)


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
