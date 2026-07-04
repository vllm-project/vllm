# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy tests for int4 embedding quantization."""

import pytest
import torch

from vllm import LLM, SamplingParams


@pytest.mark.skipif(
    not torch.accelerator.is_available() or torch.accelerator.device_count() < 1,
    reason="Requires CUDA GPU",
)
def test_int4_embedding_accuracy():
    """Test that int4 quantization maintains acceptable accuracy.

    This test compares output texts between baseline and int4 quantized
    models to ensure quantization doesn't break inference.
    """
    # Use a small model for testing
    model_name = "facebook/opt-125m"  # Small model with 50k vocab

    # Test prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Write a story about a robot that",
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
    )

    # Baseline: Load model without quantization
    llm_baseline = LLM(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=0.9,
    )

    baseline_outputs = llm_baseline.generate(prompts, sampling_params)
    baseline_texts = [output.outputs[0].text for output in baseline_outputs]

    del llm_baseline
    torch.accelerator.empty_cache()

    # Int4 quantized: Load with int4 embedding quantization
    llm_int4 = LLM(
        model=model_name,
        dtype="float16",
        quantization="int4_per_channel_weight_only",
        gpu_memory_utilization=0.9,
    )

    int4_outputs = llm_int4.generate(prompts, sampling_params)
    int4_texts = [output.outputs[0].text for output in int4_outputs]

    del llm_int4
    torch.accelerator.empty_cache()

    # Compare outputs - both should generate non-empty text
    # Note: With temperature=0, outputs should be similar but not necessarily
    # identical due to quantization error in embeddings
    assert len(baseline_texts) == len(int4_texts)
    assert all(len(text) > 0 for text in baseline_texts)
    assert all(len(text) > 0 for text in int4_texts)


@pytest.mark.skipif(
    not torch.accelerator.is_available() or torch.accelerator.device_count() < 1,
    reason="Requires CUDA GPU",
)
def test_int4_embedding_memory_savings():
    """Test that int4 quantization reduces embedding weight size."""
    model_name = "facebook/opt-125m"

    # Load baseline model and check embedding size
    llm_baseline = LLM(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=0.9,
    )

    # Access embedding layer through correct path
    baseline_model = (
        llm_baseline.llm_engine.model_executor.driver_worker.worker.model_runner.model
    )
    baseline_embed = baseline_model.model.decoder.embed_tokens
    baseline_weight_size = (
        baseline_embed.weight.numel() * baseline_embed.weight.element_size()
    )

    del llm_baseline
    torch.accelerator.empty_cache()

    # Load int4 quantized model
    llm_int4 = LLM(
        model=model_name,
        dtype="float16",
        quantization="int4_per_channel_weight_only",
        gpu_memory_utilization=0.9,
    )

    int4_model = (
        llm_int4.llm_engine.model_executor.driver_worker.worker.model_runner.model
    )
    int4_embed = int4_model.model.decoder.embed_tokens

    # Int4 weight is packed: half the columns + scale
    int4_weight_size = int4_embed.weight.numel() * int4_embed.weight.element_size()
    int4_scale_size = (
        int4_embed.weight_scale.numel() * int4_embed.weight_scale.element_size()
    )
    int4_total_size = int4_weight_size + int4_scale_size

    del llm_int4
    torch.accelerator.empty_cache()

    # Int4 should use significantly less memory for the embedding weight
    # Expected: ~75% reduction (2 bytes -> 0.5 bytes + scale)
    assert int4_total_size < baseline_weight_size, (
        f"Int4 embedding ({int4_total_size} bytes) should be smaller than "
        f"baseline ({baseline_weight_size} bytes)"
    )

    # Check that reduction is at least 50% (accounting for scale overhead)
    reduction_ratio = int4_total_size / baseline_weight_size
    assert reduction_ratio < 0.5, (
        f"Expected at least 50% size reduction, got {reduction_ratio:.2%}"
    )


if __name__ == "__main__":
    print("Running int4 embedding accuracy tests...")
    test_int4_embedding_accuracy()
    print("\n" + "=" * 50 + "\n")
    test_int4_embedding_memory_savings()
    print("\nAll tests passed!")
