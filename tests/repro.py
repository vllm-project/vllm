# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA exposed split implementation."""

import os

from vllm import LLM, SamplingParams
from vllm.config.compilation import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
)
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = "deepseek-ai/DeepSeek-V2-Lite"
PROMPTS = ["The capital of France is"]


def test_mla_exposed_eager():
    """Test that MLA exposed split path works end-to-end."""
    os.environ["VLLM_MLA_EXPOSED_SPLIT"] = "1"
    llm = LLM(
        model=MODEL,
        max_model_len=256,
        trust_remote_code=True,
        disable_log_stats=True,
        enforce_eager=True,
        gpu_memory_utilization=0.75,
    )

    # Verify that generation works with exposed_split=True
    outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=5, temperature=0))
    assert len(outputs) > 0
    assert len(outputs[0].outputs[0].text) > 0


def test_mla_exposed_compiled():
    """Test that MLA exposed split path works end-to-end."""
    os.environ["VLLM_MLA_EXPOSED_SPLIT"] = "1"
    llm = LLM(
        model=MODEL,
        max_model_len=256,
        trust_remote_code=True,
        disable_log_stats=True,
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            use_inductor_graph_partition=True,
        ),
        gpu_memory_utilization=0.75,
    )

    # Verify that generation works with exposed_split=True
    outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=5, temperature=0))
    assert len(outputs) > 0
    assert len(outputs[0].outputs[0].text) > 0


def test_mla_numerical_accuracy_eager():
    """Test numerical accuracy when env var is 0 vs 1."""
    import gc
    import time

    import torch

    os.environ["VLLM_MLA_EXPOSED_SPLIT"] = "0"
    llm_custom = LLM(
        model=MODEL,
        max_model_len=256,
        trust_remote_code=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.5,  # Reduced to allow sequential runs
        enforce_eager=True,
    )

    sampling_params = SamplingParams(max_tokens=20, temperature=0, seed=42)
    expected_result = llm_custom.generate(PROMPTS, sampling_params)

    # Clean up first instance to free GPU memory
    del llm_custom
    cleanup_dist_env_and_memory()
    gc.collect()
    torch.accelerator.empty_cache()
    time.sleep(2)  # Give time for cleanup

    os.environ["VLLM_MLA_EXPOSED_SPLIT"] = "1"
    llm_exposed = LLM(
        model=MODEL,
        max_model_len=256,
        trust_remote_code=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.5,  # Reduced to allow sequential runs
        enforce_eager=True,
    )

    actual_result = llm_exposed.generate(PROMPTS, sampling_params)

    assert len(expected_result) == len(actual_result)
    for expected, actual in zip(expected_result, actual_result):
        # breakpoint()
        assert expected.outputs[0].text == actual.outputs[0].text

    # Clean up second instance
    del llm_exposed
    cleanup_dist_env_and_memory()


def test_mla_numerical_accuracy_compile():
    """Test numerical accuracy when env var is 0 vs 1."""
    import gc
    import time

    import torch

    os.environ["VLLM_MLA_EXPOSED_SPLIT"] = "0"
    llm_custom = LLM(
        model=MODEL,
        max_model_len=256,
        trust_remote_code=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.5,  # Reduced to allow sequential runs
        compilation_config=CompilationConfig(
            mode=CompilationMode.STOCK_TORCH_COMPILE,
            cudagraph_mode=CUDAGraphMode.NONE,
            use_inductor_graph_partition=True,
        ),
    )

    sampling_params = SamplingParams(max_tokens=20, temperature=0, seed=42)
    expected_result = llm_custom.generate(PROMPTS, sampling_params)

    # Clean up first instance to free GPU memory
    del llm_custom
    cleanup_dist_env_and_memory()
    gc.collect()
    torch.accelerator.empty_cache()
    time.sleep(2)  # Give time for cleanup

    os.environ["VLLM_MLA_EXPOSED_SPLIT"] = "1"
    llm_exposed = LLM(
        model=MODEL,
        max_model_len=256,
        trust_remote_code=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.5,  # Reduced to allow sequential runs
        compilation_config=CompilationConfig(
            mode=CompilationMode.STOCK_TORCH_COMPILE,
            cudagraph_mode=CUDAGraphMode.NONE,
            use_inductor_graph_partition=True,
        ),
    )

    actual_result = llm_exposed.generate(PROMPTS, sampling_params)

    assert len(expected_result) == len(actual_result)
    for expected, actual in zip(expected_result, actual_result):
        assert expected.outputs[0].text == actual.outputs[0].text

    # Clean up second instance
    del llm_exposed
    cleanup_dist_env_and_memory()


def test_mla_numerical_accuracy_compile_cuda_graphs():
    """Test numerical accuracy when env var is 0 vs 1."""
    import gc
    import time

    import torch

    os.environ["VLLM_MLA_EXPOSED_SPLIT"] = "0"
    llm_custom = LLM(
        model=MODEL,
        max_model_len=256,
        trust_remote_code=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.5,  # Reduced to allow sequential runs
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            use_inductor_graph_partition=True,
        ),
    )

    sampling_params = SamplingParams(max_tokens=20, temperature=0, seed=42)
    expected_result = llm_custom.generate(PROMPTS, sampling_params)

    # Clean up first instance to free GPU memory
    del llm_custom
    cleanup_dist_env_and_memory()
    gc.collect()
    torch.accelerator.empty_cache()
    time.sleep(2)  # Give time for cleanup

    os.environ["VLLM_MLA_EXPOSED_SPLIT"] = "1"
    llm_exposed = LLM(
        model=MODEL,
        max_model_len=256,
        trust_remote_code=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.5,  # Reduced to allow sequential runs
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            use_inductor_graph_partition=True,
        ),
    )

    actual_result = llm_exposed.generate(PROMPTS, sampling_params)

    assert len(expected_result) == len(actual_result)
    for expected, actual in zip(expected_result, actual_result):
        assert expected.outputs[0].text == actual.outputs[0].text

    # Clean up second instance
    del llm_exposed
    cleanup_dist_env_and_memory()
