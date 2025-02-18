# SPDX-License-Identifier: Apache-2.0

import random
from typing import List, Optional, Tuple

import pytest

from tests.v1.engine.utils import PLP_APC_UNSUPPORTED_MSG
from vllm import LLM, SamplingParams

MODEL = "facebook/opt-125m"
DTYPE = "half"


@pytest.fixture(
    scope="module",
    # Prefix caching
    params=[False, True])
def vllm_model(vllm_runner, request):
    """VllmRunner test fixture parameterized by APC."""
    enable_prefix_caching = request.param
    with vllm_runner(
            MODEL,
            dtype=DTYPE,
            max_model_len=128,
            enforce_eager=True,
            enable_prefix_caching=enable_prefix_caching,
            gpu_memory_utilization=0.5,
    ) as vllm_model:
        # VllmRunner instance is cleaned up after test.
        yield vllm_model


def _get_test_sampling_params(
    prompt_list: List[str],
    seed: Optional[int] = 42,
) -> Tuple[List[SamplingParams], List[int]]:
    """Generate random sampling params for a batch."""

    def get_mostly_n_gt1() -> int:
        """Mostly n \in [2,20], ~1/3 n=1"""
        x = random.randint(0, 28)
        if x < 10:
            return 1
        else:
            return x - 8

    n_list = [get_mostly_n_gt1() for _ in range(len(prompt_list))]
    # High temperature to maximize the chance of unique completions
    return [
        SamplingParams(temperature=0.95, top_p=0.95, n=n, seed=seed)
        for n in n_list
    ], n_list


def test_parallel_sampling(monkeypatch, vllm_model, example_prompts) -> None:
    """Test passes if parallel sampling `n>1` yields `n` uniques completions.
    
    Args:
      monkeypatch: test fixture for modifying text env, scoped to the test.
      vllm_model: VllmRunner instance under test.
      example_prompt: test fixture providing prompts for testing.
    """
    monkeypatch.setenv("VLLM_USE_V1", "1")
    # Generate batch sampling params
    sampling_params_list, n_list = _get_test_sampling_params(example_prompts)
    # Process requests
    model: LLM = vllm_model.model
    outputs = model.generate(example_prompts, sampling_params_list)

    # Validate each request response
    for out, n in zip(outputs, n_list):
        completion_counts = {}
        # Assert correct number of completions
        assert len(out.outputs) == n, (
            f"{len(out.outputs)} completions; {n} expected.")
        for idx in range(n):
            comp = out.outputs[idx]
            # Assert correct completion indices
            assert comp.index == idx, (f"Index {comp.index}; expected {idx}.")
            text = comp.text
            completion_counts[text] = completion_counts.get(text, 0) + 1
        # Assert unique completions
        if len(completion_counts) != n:
            repeats = {
                txt: num
                for (txt, num) in completion_counts.items() if num > 1
            }
            raise AssertionError(
                f"{len(completion_counts)} unique completions; expected"
                f" {n}. Repeats: {repeats}")


def test_llm_engine_refuses_prompt_logprobs_with_apc(monkeypatch):
    """Test passes if LLMEngine raises an exception when it is configured
    for automatic prefix caching and it receives a request with
    prompt_logprobs enabled, which is incompatible."""

    monkeypatch.setenv("VLLM_USE_V1", "1")
    # TODO(nick): Single-proc to work around a ZMQ shutdown hang for now.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with pytest.raises(ValueError) as excinfo:
        LLM(model=MODEL, enable_prefix_caching=True).generate(
            "Hello, my name is",
            SamplingParams(temperature=0.8, top_p=0.95, prompt_logprobs=5))

    # Validate exception string is correct
    assert str(excinfo.value) == PLP_APC_UNSUPPORTED_MSG
