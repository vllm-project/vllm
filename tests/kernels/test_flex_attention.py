# SPDX-License-Identifier: Apache-2.0
"""Integration tests for FlexAttention backend vs default backend"""

import random

import numpy as np
import pytest
import torch
from packaging import version

from vllm import LLM, SamplingParams

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
def test_flex_attention_vs_default_backend(monkeypatch):
    """Test that FlexAttention produces the same outputs as the default backend.

    This test compares the outputs from the FlexAttention backend with
    the default backend, ensuring they are identical when using the same seed.
    """
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    seed = 42
    max_tokens = 32
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
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        set_seed(seed)

        llm_flex = LLM(
            model_name,
            tensor_parallel_size=1,
            num_gpu_blocks_override=128,
            enforce_eager=True,
        )
        output_flex = llm_flex.generate(prompts, sampling_params)

    # Run with default backend
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        set_seed(seed)
        llm_default = LLM(
            model_name,
            tensor_parallel_size=1,
            num_gpu_blocks_override=128,
            enforce_eager=True,
        )
        output_default = llm_default.generate(prompts, sampling_params)

    # Compare outputs from both backends
    for i, (flex_result,
            default_result) in enumerate(zip(output_flex, output_default)):
        prompt = prompts[i]
        flex_text = flex_result.outputs[0].text
        default_text = default_result.outputs[0].text

        assert flex_text == default_text, (
            f"FlexAttention output doesn't match default for: {prompt!r}\n"
            f"FlexAttention: {flex_text!r}\n"
            f"Default: {default_text!r}")


if __name__ == "__main__":
    pytest.main([__file__])
