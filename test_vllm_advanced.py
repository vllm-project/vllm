#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Advanced vLLM integration test - actually instantiate the model.

This creates a mock HuggingFace config and instantiates our model
through vLLM's system.
"""

import json
import os

# Import our model
import sys
import tempfile

import torch

sys.path.insert(0, "tests")
from test_generic_model_support import (
    GenericTransformerForCausalLMVLLM,
)


def create_mock_model_directory():
    """Create a temporary directory with model config files."""
    tmpdir = tempfile.mkdtemp(prefix="vllm_test_model_")

    # Create a minimal HuggingFace config
    config = {
        "architectures": ["GenericTransformerForCausalLM"],
        "model_type": "generic_transformer",
        "vocab_size": 1024,
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "max_position_embeddings": 512,
        "torch_dtype": "float16",
    }

    config_path = os.path.join(tmpdir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created mock model directory: {tmpdir}")
    print(f"Config: {config_path}")

    return tmpdir


def test_model_instantiation():
    """Test that we can instantiate our model through vLLM's config system."""
    print("=" * 70)
    print("Testing Model Instantiation with vLLM Config")
    print("=" * 70)

    from vllm.model_executor.models.registry import ModelRegistry

    # Register our model
    print("\n1. Registering model...")
    ModelRegistry.register_model(
        "GenericTransformerForCausalLM",
        GenericTransformerForCausalLMVLLM,
    )
    print("   ✓ Model registered")

    # Create mock config
    print("\n2. Creating mock HuggingFace config...")
    model_dir = create_mock_model_directory()

    try:
        # Create vLLM ModelConfig
        print("\n3. Creating vLLM ModelConfig...")

        # Create a minimal config object that our model wrapper expects
        class MockHFConfig:
            def __init__(self):
                self.vocab_size = 1024
                self.hidden_size = 256
                self.num_hidden_layers = 4
                self.num_attention_heads = 4
                self.max_position_embeddings = 512

        class MockVLLMConfig:
            def __init__(self):
                self.hf_config = MockHFConfig()

        vllm_config = MockVLLMConfig()
        print("   ✓ Config created")

        # Instantiate the model
        print("\n4. Instantiating model...")
        model = GenericTransformerForCausalLMVLLM(vllm_config=vllm_config)
        print(f"   ✓ Model instantiated: {type(model).__name__}")

        # Test forward pass
        print("\n5. Testing forward pass...")
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, 1024, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        with torch.no_grad():
            hidden_states = model(input_ids, positions)
            logits = model.compute_logits(hidden_states)

        print("   ✓ Forward pass successful")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Hidden states shape: {hidden_states.shape}")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Expected logits: ({batch_size}, {seq_len}, 1024)")

        assert logits.shape == (batch_size, seq_len, 1024), (
            f"Unexpected output shape: {logits.shape}"
        )

        print("   ✓ Output shape correct")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        import shutil

        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            print("\n6. Cleaned up temporary directory")


def test_model_resolve():
    """Test that vLLM can resolve our model architecture."""
    print("\n" + "=" * 70)
    print("Testing Model Architecture Resolution")
    print("=" * 70)

    from vllm.model_executor.models.registry import ModelRegistry

    # Register
    print("\n1. Registering model...")
    ModelRegistry.register_model(
        "GenericTransformerForCausalLM",
        GenericTransformerForCausalLMVLLM,
    )

    print("\n2. Creating mock config for resolution...")

    class MockHFConfig:
        architectures = ["GenericTransformerForCausalLM"]
        vocab_size = 1024
        hidden_size = 256
        num_hidden_layers = 4
        num_attention_heads = 4

    class MockModelConfig:
        hf_config = MockHFConfig()
        model_impl = "auto"
        runner_type = None
        convert_type = None

        def _get_transformers_backend_cls(self):
            return None

    config = MockModelConfig()

    try:
        print("\n3. Resolving model class...")
        model_cls, resolved_arch = ModelRegistry.resolve_model_cls(
            ["GenericTransformerForCausalLM"],
            config,
        )

        print(f"   ✓ Model resolved: {model_cls.__name__}")
        print(f"   ✓ Architecture: {resolved_arch}")

        return True

    except Exception as e:
        print(f"\n✗ Resolution failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_interfaces():
    """Test that vLLM correctly identifies our model's capabilities."""
    print("\n" + "=" * 70)
    print("Testing Model Interface Detection")
    print("=" * 70)

    from vllm.model_executor.models.registry import ModelRegistry

    # Register
    print("\n1. Registering model...")
    ModelRegistry.register_model(
        "GenericTransformerForCausalLM",
        GenericTransformerForCausalLMVLLM,
    )

    print("\n2. Creating config...")

    class MockHFConfig:
        architectures = ["GenericTransformerForCausalLM"]

    class MockModelConfig:
        hf_config = MockHFConfig()
        model_impl = "auto"
        runner_type = None
        convert_type = None

    config = MockModelConfig()

    try:
        print("\n3. Inspecting model capabilities...")

        # Inspect the model
        model_info, arch = ModelRegistry.inspect_model_cls(
            ["GenericTransformerForCausalLM"],
            config,
        )

        print(f"\n   Architecture: {model_info.architecture}")
        print(f"   Is text generation model: {model_info.is_text_generation_model}")
        print(f"   Is pooling model: {model_info.is_pooling_model}")
        print(f"   Supports multimodal: {model_info.supports_multimodal}")
        print(f"   Supports PP: {model_info.supports_pp}")
        print(f"   Has inner state: {model_info.has_inner_state}")
        print(f"   Is attention free: {model_info.is_attention_free}")

        print("\n   ✓ Model inspection successful")

        return True

    except Exception as e:
        print(f"\n✗ Inspection failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all advanced integration tests."""
    print("\n" + "=" * 70)
    print("Advanced vLLM Integration Tests")
    print("=" * 70)

    results = []

    # Test 1: Model instantiation
    try:
        results.append(("Model Instantiation", test_model_instantiation()))
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Model Instantiation", False))

    # Test 2: Model resolution
    try:
        results.append(("Model Resolution", test_model_resolve()))
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Model Resolution", False))

    # Test 3: Interface detection
    try:
        results.append(("Interface Detection", test_model_interfaces()))
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Interface Detection", False))

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
    success = main()
    sys.exit(0 if success else 1)
