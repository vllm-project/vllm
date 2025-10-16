# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch
from torch.torch_version import TorchVersion

from vllm import LLM, SamplingParams
from vllm.config.compilation import DynamicShapesType


def cleanup_gpu_memory():
    """Clean up GPU memory after each test"""
    gc.collect()  # Clear Python objects
    torch.cuda.empty_cache()  # Clear PyTorch GPU memory cache
    torch.cuda.synchronize()  # Wait for all GPU operations to complete


def get_test_models():
    """Get list of models to test based on PyTorch version"""
    # Parse PyTorch version
    result = ["microsoft/DialoGPT-small", "gpt2", "facebook/opt-125m"]
    # Handle alpha versions by removing pre-release suffixes
    version_parts = torch.__version__.split("+")[0].split("a")[0]
    clean_version = version_parts.split("b")[0].split("rc")[0]
    if TorchVersion(clean_version) >= TorchVersion("2.10"):
        # Requires some fixes only available in PyTorch 2.10+
        result.append("Qwen/Qwen2-1.5B-Instruct")
        result.append("Qwen/Qwen2-7B-Instruct")
        result.append("openlm-research/open_llama_13b")

    return result


@pytest.mark.parametrize("model_name", get_test_models())
def test_dynamic_shapes_compilation(monkeypatch, model_name):
    """Test that all dynamic shapes types produce compiles"""
    print(f"\nTesting model: {model_name}")

    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    # Note USE_AOT_COMPILE fails https://github.com/vllm-project/vllm/issues/27040.
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "0")

    prompt = "Hello, my name is"
    results = {}

    print("Testing EAGER (no compilation) baseline...")
    cleanup_gpu_memory()

    eager_model = LLM(
        model=model_name,
        compilation_config={
            "level": 0,  # NO_COMPILATION - eager mode
        },
        # gpu_memory_utilization=0.2,
    )

    # Generate text with deterministic sampling parameters
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.0,  # Deterministic generation
        seed=42,  # Fixed seed for consistency
    )
    eager_output = eager_model.generate(prompt, sampling_params=sampling_params)
    results["EAGER"] = eager_output[0].outputs[0].text

    # Cleanup model
    del eager_model
    cleanup_gpu_memory()

    # Test all dynamic shapes types with compilation
    for shapes_type in [
        DynamicShapesType.BACKED,
        DynamicShapesType.UNBACKED,
        DynamicShapesType.BACKED_SIZE_OBLIVIOUS,
    ]:
        print(f"Testing {shapes_type.name} dynamic shapes...")

        # Initialize the model with specific dynamic shapes configuration
        model = LLM(
            model=model_name,
            compilation_config={
                "level": 3,  # PIECEWISE compilation
                "dynamic_shapes_config": {
                    "dynamic_shapes_type": shapes_type.value,
                    "eval_dynamo_ds_guards": False,
                },
            },
            # gpu_memory_utilization=0.2,
        )

        output = model.generate(prompt, sampling_params=sampling_params)

        # Store results for comparison
        results[shapes_type.name] = output[0].outputs[0].text

        # Cleanup model
        del model
        cleanup_gpu_memory()

    # Verify all results are non-empty strings
    for shape_type, result in results.items():
        assert isinstance(result, str), f"{shape_type} should return a string"
        assert len(result.strip()) > 0, f"{shape_type} should generate non-empty text"

    # Print results
    for shape_type, result in results.items():
        print(f"{shape_type}: '{result}'")


if __name__ == "__main__":
    """Run the test directly as a Python script"""
    import os

    print("Running dynamic shapes compilation test...")

    # Get test models based on PyTorch version
    test_models = get_test_models()
    print(f"Testing {len(test_models)} models: {test_models}")

    # Create a mock monkeypatch object for environment variables
    class MockMonkeypatch:
        def setenv(self, key, value):
            os.environ[key] = value

    monkeypatch = MockMonkeypatch()

    # Run test for each model
    for model_name in test_models:
        try:
            print(f"\n{'=' * 60}")
            print(f"Testing model: {model_name}")
            print(f"{'=' * 60}")

            test_dynamic_shapes_compilation(monkeypatch, model_name)

            print(f"✅ Test passed for {model_name}")

        except Exception as e:
            print(f"❌ Test failed for {model_name}: {e}")
            raise

    print("\n🎉 All tests completed successfully!")
