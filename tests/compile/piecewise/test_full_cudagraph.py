# SPDX-License-Identifier: Apache-2.0
import contextlib
import os

import pytest

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig

MODEL = "Qwen/Qwen2-1.5B-Instruct"


@contextlib.contextmanager
def temporary_environ(env_vars):
    """
    Temporarily set environment variables and restore them afterward.
    We have to do this vs monkeypatch because monkeypatch doesn't work
    with "module" scoped fixtures.
    """
    original_env = {k: os.environ.get(k) for k in env_vars}
    try:
        os.environ.update(env_vars)
        yield
    finally:
        for k, v in original_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def full_cudagraph_llm():
    with temporary_environ({
            "VLLM_USE_V1": "1",
            "VLLM_FLASH_ATTN_VERSION": "3"
    }):
        return LLM(model=MODEL,
                   gpu_memory_utilization=0.2,
                   compilation_config=CompilationConfig(full_cuda_graph=True))


@pytest.fixture(scope="module")
def piecewise_llm():
    with temporary_environ({
            "VLLM_USE_V1": "1",
            "VLLM_FLASH_ATTN_VERSION": "3"
    }):
        return LLM(model=MODEL,
                   gpu_memory_utilization=0.5,
                   compilation_config=CompilationConfig())


def generate_text(llm: LLM, batch_size: int, max_tokens: int):
    prompts = ["Hi my name is"] * batch_size
    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=max_tokens,
                                     top_p=0.95)

    return llm.generate(prompts, sampling_params)


@pytest.mark.parametrize(("batch_size", "max_tokens"), [(1, 10), (7, 10),
                                                        (16, 10), (25, 10),
                                                        (32, 10), (45, 10),
                                                        (64, 10), (8, 5),
                                                        (8, 20), (8, 200)])
def test_full_cudagraph(batch_size, max_tokens, full_cudagraph_llm,
                        piecewise_llm):
    """
    Load full cudagraph model and piecewise model once, and at the same time to
    reuse them across various test cases.

    Test various batch sizes and max_tokens to ensure that the full cudagraph
    compilation works for padded cases too.
    """
    piecewise_responses = generate_text(piecewise_llm,
                                        batch_size=batch_size,
                                        max_tokens=max_tokens)
    full_cudagraph_responses = generate_text(full_cudagraph_llm,
                                             batch_size=batch_size,
                                             max_tokens=max_tokens)

    # Check that all responses are the same
    for i in range(len(piecewise_responses)):
        assert piecewise_responses[i].outputs[
            0].text == full_cudagraph_responses[i].outputs[0].text


def test_full_cudagraph_with_invalid_backend():
    with temporary_environ({
            "VLLM_USE_V1": "1",
            "VLLM_FLASH_ATTN_VERSION":
            "2"  #FA2 not supported with full_cuda_graph
    }), pytest.raises(RuntimeError):
        LLM(model=MODEL,
            compilation_config=CompilationConfig(full_cuda_graph=True))
