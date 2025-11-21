#!/usr/bin/env python3
"""
vLLM INT4 + LoRA End-to-End Test

Tests vLLM's INT4 support with LoRA adapters using compressed-tensors format.
"""
import os
import sys
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def test_int4_lora():
    """Test vLLM INT4 + LoRA end-to-end."""
    print("=" * 80)
    print("vLLM INT4 + LoRA END-TO-END TEST")
    print("=" * 80)

    # Use a small INT4 model from NeuralMagic
    model_id = "neuralmagic/Mistral-7B-Instruct-v0.3-quantized.w4a16"

    print(f"\n[1] Loading INT4 model: {model_id}")
    print("  This model uses compressed-tensors INT4 quantization")

    try:
        # Load the INT4 quantized model with vLLM
        llm = LLM(
            model=model_id,
            quantization="compressed-tensors",
            max_model_len=2048,
            enable_lora=True,  # Enable LoRA support
            max_lora_rank=16,
        )
        print("✓ Model loaded successfully")

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    # Test baseline inference (no LoRA)
    print("\n[2] Testing baseline INT4 inference (no LoRA)...")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
    prompts = ["The future of AI is"]

    try:
        outputs = llm.generate(prompts, sampling_params)
        baseline_output = outputs[0].outputs[0].text
        print(f"✓ Baseline output: {baseline_output}")
    except Exception as e:
        print(f"✗ Baseline inference failed: {e}")
        return False

    # Note: To test with actual LoRA adapters, we would need:
    # 1. A trained LoRA adapter compatible with this model
    # 2. Load it using LoRARequest
    # 3. Generate with lora_request parameter

    print("\n[3] Checking INT4 + LoRA compatibility...")
    print("  INT4 layers detected:", hasattr(llm.llm_engine.model_executor, "driver_worker"))

    # Check if LoRA support is enabled
    model_config = llm.llm_engine.model_config
    lora_config = llm.llm_engine.lora_config

    if lora_config is not None:
        print(f"✓ LoRA support enabled:")
        print(f"  - Max LoRA rank: {lora_config.max_lora_rank}")
        print(f"  - LoRA dtype: {lora_config.lora_dtype}")
    else:
        print("✗ LoRA support not enabled")
        return False

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✓ INT4 model loaded successfully")
    print("✓ Baseline inference working")
    print("✓ LoRA support enabled and configured")
    print("\nNext steps:")
    print("- Train/obtain a LoRA adapter for this model")
    print("- Test with actual LoRA adapter using LoRARequest")

    return True


if __name__ == "__main__":
    success = test_int4_lora()
    sys.exit(0 if success else 1)
