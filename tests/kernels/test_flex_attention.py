# SPDX-License-Identifier: Apache-2.0
"""Integration tests for FlexAttention backend vs default backend"""

import random

import numpy as np
import pytest
import torch
from packaging import version

TORCH_VERSION = version.parse(torch.__version__)
MINIMUM_TORCH_VERSION = version.parse("2.7.0")


def set_seed(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flash_attention_env(monkeypatch):
    """Setup environment for default backend"""
    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASH_ATTN_VLLM_V1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def flex_attention_env(monkeypatch):
    """Setup environment for flex attention backend"""
    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLEX_ATTENTION_VLLM_V1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


@pytest.mark.skipif(not torch.cuda.is_available()
                    or TORCH_VERSION < MINIMUM_TORCH_VERSION,
                    reason="CUDA not available or PyTorch version < 2.7")
def test_flex_attention_vs_default(monkeypatch, vllm_runner):
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    seed = 42
    max_tokens = 16
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    set_seed(seed)

    # First run with default attention
    flash_attention_env(monkeypatch)
    with vllm_runner(
            model_name,
            tensor_parallel_size=1,
            num_gpu_blocks_override=128,
    ) as llm1:
        results1 = llm1.generate_greedy(prompts, max_tokens=max_tokens)

    set_seed(seed)

    # Second run with flex attention
    flex_attention_env(monkeypatch)
    with vllm_runner(
            model_name,
            tensor_parallel_size=1,
            num_gpu_blocks_override=128,
    ) as llm2:
        results2 = llm2.generate_greedy(prompts, max_tokens=max_tokens)

    # Potentially flaky but for these small sizes appear to always match
    for i, (r1, r2) in enumerate(zip(results1, results2)):
        assert r1 == r2, (
            f"Non-deterministic results for prompt {i}: {prompts[i]!r}")


if __name__ == "__main__":
    pytest.main([__file__])
