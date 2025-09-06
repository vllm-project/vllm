#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example script demonstrating the enhanced error handling in vLLM V1.

This script intentionally triggers common initialization errors to show
the improved error messages and suggestions.
"""

import os
import sys

# Set V1 mode
os.environ["VLLM_USE_V1"] = "1"

from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.engine.initialization_errors import V1InitializationError


def test_memory_error():
    """Test insufficient memory error with helpful suggestions."""
    print("\n=== Testing Insufficient Memory Error ===")

    try:
        # Try to load a large model with very low memory utilization
        engine_args = EngineArgs(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",  # Large model
            gpu_memory_utilization=0.1,  # Very low memory
            max_model_len=4096,
        )
        _ = LLM.from_engine_args(engine_args)
    except V1InitializationError as e:
        print(f"Caught enhanced error: {type(e).__name__}")
        print(f"Error message:\n{str(e)}")
        return True
    except Exception as e:
        print(f"Caught unexpected error: {type(e).__name__}: {str(e)}")
        return False

    print("No error occurred (unexpected)")
    return False


def test_kv_cache_error():
    """Test insufficient KV cache memory error."""
    print("\n=== Testing Insufficient KV Cache Memory Error ===")

    try:
        # Try to use a very large max_model_len that won't fit in memory
        engine_args = EngineArgs(
            model="microsoft/DialoGPT-small",  # Small model
            gpu_memory_utilization=0.95,
            max_model_len=50000,  # Very large context length
        )
        _ = LLM.from_engine_args(engine_args)
    except V1InitializationError as e:
        print(f"Caught enhanced error: {type(e).__name__}")
        print(f"Error message:\n{str(e)}")
        return True
    except Exception as e:
        print(f"Caught unexpected error: {type(e).__name__}: {str(e)}")
        return False

    print("No error occurred (unexpected)")
    return False


def test_model_loading_error():
    """Test model loading error with helpful suggestions."""
    print("\n=== Testing Model Loading Error ===")

    try:
        # Try to load a non-existent model
        engine_args = EngineArgs(
            model="non-existent-model/does-not-exist",
            gpu_memory_utilization=0.8,
        )
        _ = LLM.from_engine_args(engine_args)
    except V1InitializationError as e:
        print(f"Caught enhanced error: {type(e).__name__}")
        print(f"Error message:\n{str(e)}")
        return True
    except Exception as e:
        print(f"Caught unexpected error: {type(e).__name__}: {str(e)}")
        return False

    print("No error occurred (unexpected)")
    return False


def test_successful_initialization():
    """Test successful initialization with a small model."""
    print("\n=== Testing Successful Initialization ===")

    try:
        # Use a small model that should work
        engine_args = EngineArgs(
            model="microsoft/DialoGPT-small",
            gpu_memory_utilization=0.8,
            max_model_len=512,
        )
        llm = LLM.from_engine_args(engine_args)
        print("✅ Successfully initialized LLM!")

        # Test a simple generation
        outputs = llm.generate(["Hello, how are you?"], max_tokens=10)
        print(f"✅ Successfully generated: {outputs[0].outputs[0].text}")
        return True

    except Exception as e:
        print(f"❌ Unexpected error during successful test: "
              f"{type(e).__name__}: {str(e)}")
        return False


def main():
    """Run all test cases."""
    print("Enhanced Error Handling Demo for vLLM V1")
    print("=" * 50)

    # Check if V1 is enabled
    if os.environ.get("VLLM_USE_V1") != "1":
        print(
            "❌ VLLM_USE_V1 is not set to 1. Please set it to enable V1 mode.")
        return 1

    tests = [
        ("Memory Error", test_memory_error),
        ("KV Cache Error", test_kv_cache_error),
        ("Model Loading Error", test_model_loading_error),
        ("Successful Initialization", test_successful_initialization),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print(f"\n❌ Test '{test_name}' interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with unexpected error: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("Test Results Summary:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
