# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA exposed prefill/decode split with torch.compile.

This tests the VLLM_MLA_EXPOSED_SPLIT feature which exposes the batch
splitting and GEMMs to torch.compile for fusion opportunities.
"""

import gc

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationMode
from vllm.platforms import current_platform


def is_mla_model_supported() -> bool:
    """Check if MLA models are supported on this platform."""
    if not torch.cuda.is_available():
        return False
    # MLA requires sufficient GPU capability
    capability = torch.cuda.get_device_capability()
    return capability[0] >= 8  # Ampere or newer


MLA_TEST_MODEL = "deepseek-ai/DeepSeek-V2-Lite"


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up GPU memory after each test."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.mark.skipif(
    not is_mla_model_supported(),
    reason="MLA requires GPU with compute capability >= 8.0",
)
@pytest.mark.parametrize("exposed_split", [False, True])
def test_mla_exposed_split_correctness(monkeypatch, exposed_split):
    """Test that exposed split produces correct results."""
    if exposed_split:
        monkeypatch.setenv("VLLM_MLA_EXPOSED_SPLIT", "1")
    else:
        monkeypatch.setenv("VLLM_MLA_EXPOSED_SPLIT", "0")

    prompt = "The capital of France is"
    sampling_params = SamplingParams(max_tokens=10, temperature=0)

    model = LLM(
        model=MLA_TEST_MODEL,
        max_model_len=512,
        trust_remote_code=True,
        compilation_config={
            "mode": CompilationMode.VLLM_COMPILE,
        },
    )

    output = model.generate(prompt, sampling_params)
    result = output[0].outputs[0].text

    # Basic sanity check - should produce non-empty output
    assert len(result) > 0, f"Expected non-empty output, got: {result}"

    del model


@pytest.mark.skipif(
    not is_mla_model_supported(),
    reason="MLA requires GPU with compute capability >= 8.0",
)
def test_mla_exposed_split_piecewise_cudagraph(monkeypatch):
    """Test that exposed split works with piecewise cudagraph."""
    monkeypatch.setenv("VLLM_MLA_EXPOSED_SPLIT", "1")

    prompt = "Hello, world!"
    sampling_params = SamplingParams(max_tokens=5, temperature=0)

    model = LLM(
        model=MLA_TEST_MODEL,
        max_model_len=256,
        trust_remote_code=True,
        compilation_config={
            "mode": CompilationMode.VLLM_COMPILE,
            "cudagraph_mode": "PIECEWISE",
        },
    )

    # Run multiple times to exercise cudagraph capture and replay
    for _ in range(3):
        output = model.generate(prompt, sampling_params)
        assert len(output[0].outputs[0].text) > 0

    del model


@pytest.mark.skipif(
    not is_mla_model_supported(),
    reason="MLA requires GPU with compute capability >= 8.0",
)
def test_mla_exposed_split_mixed_batch(monkeypatch):
    """Test exposed split with mixed prefill/decode batch."""
    monkeypatch.setenv("VLLM_MLA_EXPOSED_SPLIT", "1")

    prompts = [
        "Short prompt",
        "This is a longer prompt that should have more context",
        "Another prompt for testing mixed batch scenarios",
    ]
    sampling_params = SamplingParams(max_tokens=10, temperature=0)

    model = LLM(
        model=MLA_TEST_MODEL,
        max_model_len=512,
        trust_remote_code=True,
        compilation_config={
            "mode": CompilationMode.VLLM_COMPILE,
        },
    )

    outputs = model.generate(prompts, sampling_params)

    # All prompts should produce output
    for i, output in enumerate(outputs):
        assert len(output.outputs[0].text) > 0, f"Prompt {i} produced empty output"

    del model


@pytest.mark.skipif(
    not is_mla_model_supported(),
    reason="MLA requires GPU with compute capability >= 8.0",
)
def test_mla_custom_ops_registered():
    """Test that the new MLA custom ops are properly registered."""
    # Check that the ops exist in the vllm namespace
    assert hasattr(torch.ops.vllm, "mla_split_batch"), (
        "mla_split_batch op not registered"
    )
    assert hasattr(torch.ops.vllm, "mla_attention_decode"), (
        "mla_attention_decode op not registered"
    )
    assert hasattr(torch.ops.vllm, "mla_attention_prefill_with_output"), (
        "mla_attention_prefill_with_output op not registered"
    )


@pytest.mark.skipif(
    not is_mla_model_supported(),
    reason="MLA requires GPU with compute capability >= 8.0",
)
@pytest.mark.parametrize("use_inductor_partition", [False, True])
def test_mla_exposed_split_inductor_partition(monkeypatch, use_inductor_partition):
    """Test exposed split with inductor graph partitioning."""
    monkeypatch.setenv("VLLM_MLA_EXPOSED_SPLIT", "1")

    prompt = "Testing inductor partitioning with"
    sampling_params = SamplingParams(max_tokens=5, temperature=0)

    model = LLM(
        model=MLA_TEST_MODEL,
        max_model_len=256,
        trust_remote_code=True,
        compilation_config={
            "mode": CompilationMode.VLLM_COMPILE,
            "use_inductor_graph_partition": use_inductor_partition,
            "cudagraph_mode": "PIECEWISE" if not use_inductor_partition else "FULL",
        },
    )

    output = model.generate(prompt, sampling_params)
    assert len(output[0].outputs[0].text) > 0

    del model

