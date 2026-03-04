# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for MPS (Apple Metal) platform: load a model and generate text."""

import weakref

import pytest

from vllm.platforms import current_platform

if not current_platform.is_mps():
    pytest.skip("MPS-only tests", allow_module_level=True)

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_NAME = "distilbert/distilgpt2"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

TOKEN_IDS = [
    [0],
    [0, 1],
    [0, 2, 1],
    [0, 3, 1, 2],
]


@pytest.fixture(scope="module")
def llm():
    llm = LLM(
        model=MODEL_NAME,
        max_num_batched_tokens=4096,
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="float32",
        load_format="dummy",
        hf_overrides={"num_hidden_layers": 2},
    )

    yield weakref.proxy(llm)

    del llm
    cleanup_dist_env_and_memory()


@pytest.mark.skip_global_cleanup
def test_generate_basic(llm: LLM):
    """Generate with simple prompts, verify outputs are non-empty."""
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10)
    outputs = llm.generate(PROMPTS, sampling_params=sampling_params)
    assert len(outputs) == len(PROMPTS)
    for output in outputs:
        assert len(output.outputs) > 0
        assert len(output.outputs[0].text) > 0
        assert len(output.outputs[0].token_ids) > 0


@pytest.mark.skip_global_cleanup
def test_generate_multiple_sampling_params(llm: LLM):
    """Different sampling params per prompt."""
    sampling_params = [
        SamplingParams(temperature=0.01, top_p=0.95, max_tokens=10),
        SamplingParams(temperature=0.3, top_p=0.95, max_tokens=10),
        SamplingParams(temperature=0.7, top_p=0.95, max_tokens=10),
        SamplingParams(temperature=0.99, top_p=0.95, max_tokens=10),
    ]
    outputs = llm.generate(PROMPTS, sampling_params=sampling_params)
    assert len(outputs) == len(PROMPTS)


@pytest.mark.skip_global_cleanup
def test_generate_token_ids(llm: LLM):
    """Generate from token ID inputs."""
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10)
    prompts = [{"prompt_token_ids": ids} for ids in TOKEN_IDS]
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    assert len(outputs) == len(TOKEN_IDS)
    for output in outputs:
        assert len(output.outputs) > 0
        assert len(output.outputs[0].token_ids) > 0


@pytest.mark.skip_global_cleanup
def test_generate_max_tokens(llm: LLM):
    """Verify max_tokens is respected."""
    max_tokens = 5
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.generate(PROMPTS, sampling_params=sampling_params)
    for output in outputs:
        assert len(output.outputs[0].token_ids) <= max_tokens
