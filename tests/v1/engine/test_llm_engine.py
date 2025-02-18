# SPDX-License-Identifier: Apache-2.0

import random

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
    with vllm_runner(
            MODEL,
            dtype=DTYPE,
            max_logprobs=7,
            # Very small number of batched tokens to ensure
            # that we test chunking.
            max_num_batched_tokens=16,
            max_num_seqs=16,
            max_model_len=128,
            enforce_eager=True,
            enable_prefix_caching=request.param,
            gpu_memory_utilization=0.5,
    ) as vllm_model:
        yield vllm_model


def _get_test_sampling_params(prompt_lst):

    def get_mostly_n_gt1() -> int:
        """Mostly n>1, sometimes n=1"""
        x = random.randint(0, 28)
        if x < 10:
            return 1
        else:
            return x - 8

    n_list = [get_mostly_n_gt1() for _ in range(len(prompt_lst))]
    return [SamplingParams(temperature=0.8, top_p=0.95, n=n)
            for n in n_list], n_list


def test_parallel_sampling(monkeypatch, vllm_model, example_prompts):
    monkeypatch.setenv("VLLM_USE_V1", "1")
    sampling_params_list, n_list = _get_test_sampling_params(example_prompts)
    model: LLM = vllm_model.model
    outputs = model.generate(example_prompts, sampling_params_list)
    for out, n in zip(outputs, n_list):
        unique_texts = set()
        # Correct number of completions
        assert len(out.outputs) == n, (
            f"{len(out.outputs)} completions; {n} expected.")
        for idx in range(n):
            comp = out.outputs[idx]
            # Correct completion indices
            assert comp.index == idx, (f"Index {comp.index}; expected {idx}.")
            unique_texts.add(comp.text)
        # Unique completions
        assert len(unique_texts) == n, (
            f"{len(unique_texts)} unique completions; expected"
            f" {n}.")


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
