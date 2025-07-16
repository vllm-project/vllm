# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import random
from typing import Any

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory


@pytest.fixture
def test_prompts():
    prompt_types = ["repeat", "sentence"]
    num_prompts = 100
    prompts = []

    random.seed(0)
    random_prompt_type_choices = random.choices(prompt_types, k=num_prompts)

    # Generate a mixed batch of prompts, some of which can be easily
    # predicted by n-gram matching and some which likely cannot.
    for kind in random_prompt_type_choices:
        word_choices = ["test", "temp", "hello", "where"]
        word = random.choice(word_choices)
        if kind == "repeat":
            prompt = f"""
            please repeat the word '{word}' 10 times.
            give no other output than the word at least ten times in a row,
            in lowercase with spaces between each word and without quotes.
            """
        elif kind == "sentence":
            prompt = f"""
            please give a ten-word sentence that
            uses the word {word} at least once.
            give no other output than that simple sentence without quotes.
            """
        else:
            raise ValueError(f"Unknown prompt type: {kind}")
        prompts.append([{"role": "user", "content": prompt}])

    return prompts


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0, max_tokens=10, ignore_eos=False)


@pytest.fixture
def model_name():
    return "meta-llama/Llama-3.1-8B-Instruct"


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
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()

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
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize("model_setup", [
    ("eagle", "meta-llama/Llama-3.1-8B-Instruct",
     "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", 1),
    ("eagle3", "meta-llama/Llama-3.1-8B-Instruct",
     "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", 1),
    pytest.param(
        ("eagle", "meta-llama/Llama-4-Scout-17B-16E-Instruct",
         "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct", 4),
        marks=pytest.mark.skip(reason="Skipping due to CI OOM issues")),
],
                         ids=["llama3_eagle", "llama3_eagle3", "llama4_eagle"])
def test_eagle_correctness(
    monkeypatch: pytest.MonkeyPatch,
    test_prompts: list[list[dict[str, Any]]],
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, str, int],
):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using eagle speculative decoding.
    model_setup: (method, model_name, eagle_model_name, tp_size)
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        method, model_name, spec_model_name, tp_size = model_setup

        ref_llm = LLM(model=model_name,
                      max_model_len=2048,
                      tensor_parallel_size=tp_size)
        ref_outputs = ref_llm.chat(test_prompts, sampling_config)
        del ref_llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()

        spec_llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            speculative_config={
                "method": method,
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
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()
