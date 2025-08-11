# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for FlexAttention backend vs default backend"""

import random

import numpy as np
import pytest
import torch
from packaging import version

from vllm import SamplingParams

from ..models.utils import check_embeddings_close

TORCH_VERSION = version.parse(torch.__version__)
MINIMUM_TORCH_VERSION = version.parse("2.7.0")


def set_seed(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.mark.skipif(
    not torch.cuda.is_available() or TORCH_VERSION < MINIMUM_TORCH_VERSION,
    reason="CUDA not available or PyTorch version < 2.7",
)
def test_flex_attention_vs_default_backend(vllm_runner, monkeypatch):
    """Test that FlexAttention produces the same outputs as the default backend.

    This test compares the outputs from the FlexAttention backend with
    the default backend, ensuring they are identical when using the same seed.
    """
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    seed = 42
    max_tokens = 24
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
    ]

    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=1.0,
                                     seed=seed,
                                     max_tokens=max_tokens)

    # Run with flex attention
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_ATTENTION_BACKEND", "FLEX_ATTENTION")

        set_seed(seed)
        with vllm_runner(model_name,
                         runner="generate",
                         tensor_parallel_size=1,
                         num_gpu_blocks_override=128,
                         enforce_eager=True) as llm_flex:
            output_flex = llm_flex.generate(prompts, sampling_params)

    # Run with default backend
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        set_seed(seed)
        with vllm_runner(model_name,
                         runner="generate",
                         tensor_parallel_size=1,
                         num_gpu_blocks_override=128,
                         enforce_eager=True) as llm_default:
            output_default = llm_default.generate(prompts, sampling_params)

    # Compare outputs from both backends
    for i, (flex_result,
            default_result) in enumerate(zip(output_flex, output_default)):
        prompt = prompts[i]
        flex_text = flex_result[1][0]
        default_text = default_result[1][0]

        assert flex_text == default_text, (
            f"FlexAttention output doesn't match default for: {prompt!r}\n"
            f"FlexAttention: {flex_text!r}\n"
            f"Default: {default_text!r}")


@pytest.mark.skipif(
    not torch.cuda.is_available() or TORCH_VERSION < MINIMUM_TORCH_VERSION,
    reason="CUDA not available or PyTorch version < 2.7",
)
def test_encoder_flex_attention_vs_default_backend(vllm_runner, monkeypatch):
    """Test that FlexAttention produces the same outputs as the default backend.

    This test compares the outputs from the FlexAttention backend with
    the default backend for encoder models.
    """
    model_name = "BAAI/bge-base-en-v1.5"
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
    ]

    # Run with flex attention
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_ATTENTION_BACKEND", "FLEX_ATTENTION")
        with vllm_runner(model_name,
                         runner="pooling",
                         dtype=torch.bfloat16,
                         tensor_parallel_size=1,
                         max_model_len=100,
                         enforce_eager=True) as llm_flex:
            flex_outputs = llm_flex.embed(prompts)

    # Run with default backend
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        with vllm_runner(model_name,
                         runner="pooling",
                         dtype=torch.bfloat16,
                         tensor_parallel_size=1,
                         max_model_len=100,
                         enforce_eager=True) as llm_default:
            default_outputs = llm_default.embed(prompts)

    check_embeddings_close(
        embeddings_0_lst=flex_outputs,
        embeddings_1_lst=default_outputs,
        name_0="flex",
        name_1="default",
        tol=1e-2,
    )


if __name__ == "__main__":
    pytest.main([__file__])
