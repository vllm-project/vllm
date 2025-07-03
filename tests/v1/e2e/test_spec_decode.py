# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import Any

import pytest

from vllm import LLM, SamplingParams


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0, max_tokens=10, ignore_eos=False)


@pytest.fixture
def model_name():
    return "meta-llama/Llama-3.1-8B-Instruct"


def eagle_model_name():
    return "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"


def eagle3_model_name():
    return "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


def test_ngram_correctness(
    monkeypatch: pytest.MonkeyPatch,
    test_prompts: list[list[dict[str, Any]]],
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using ngram speculative decoding.
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        ref_llm = LLM(model=model_name, max_model_len=1024)
        ref_outputs = ref_llm.chat(test_prompts, sampling_config)
        del ref_llm

        spec_llm = LLM(
            model=model_name,
            speculative_config={
                "method": "ngram",
                "prompt_lookup_max": 5,
                "prompt_lookup_min": 3,
                "num_speculative_tokens": 3,
            },
            max_model_len=1024,
        )
        spec_outputs = spec_llm.chat(test_prompts, sampling_config)
        matches = 0
        misses = 0
        for ref_output, spec_output in zip(ref_outputs, spec_outputs):
            if ref_output.outputs[0].text == spec_output.outputs[0].text:
                matches += 1
            else:
                misses += 1
                print(f"ref_output: {ref_output.outputs[0].text}")
                print(f"spec_output: {spec_output.outputs[0].text}")

        # Heuristic: expect at least 70% of the prompts to match exactly
        # Upon failure, inspect the outputs to check for inaccuracy.
        assert matches > int(0.7 * len(ref_outputs))
        del spec_llm


@pytest.mark.parametrize("use_eagle3", [False, True], ids=["eagle", "eagle3"])
def test_eagle_correctness(
    monkeypatch: pytest.MonkeyPatch,
    test_prompts: list[list[dict[str, Any]]],
    sampling_config: SamplingParams,
    model_name: str,
    use_eagle3: bool,
):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using eagle speculative decoding.
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        ref_llm = LLM(model=model_name, max_model_len=2048)
        ref_outputs = ref_llm.chat(test_prompts, sampling_config)
        del ref_llm

        spec_model_name = eagle3_model_name(
        ) if use_eagle3 else eagle_model_name()
        spec_llm = LLM(
            model=model_name,
            trust_remote_code=True,
            speculative_config={
                "method": "eagle3" if use_eagle3 else "eagle",
                "model": spec_model_name,
                "num_speculative_tokens": 3,
                "max_model_len": 2048,
            },
            max_model_len=2048,
        )
        spec_outputs = spec_llm.chat(test_prompts, sampling_config)
        matches = 0
        misses = 0
        for ref_output, spec_output in zip(ref_outputs, spec_outputs):
            if ref_output.outputs[0].text == spec_output.outputs[0].text:
                matches += 1
            else:
                misses += 1
                print(f"ref_output: {ref_output.outputs[0].text}")
                print(f"spec_output: {spec_output.outputs[0].text}")

        # Heuristic: expect at least 66% of the prompts to match exactly
        # Upon failure, inspect the outputs to check for inaccuracy.
        assert matches > int(0.66 * len(ref_outputs))
        del spec_llm
