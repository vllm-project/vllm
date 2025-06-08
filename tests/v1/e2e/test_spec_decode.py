# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Metric


def get_spec_acceptance_metrics(metrics: list[Metric], k: int):
    num_drafts = 0
    num_accepted = 0
    acceptance_counts = [0] * k
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            num_accepted += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]
    acceptance_rate_per_pos = [
        count / num_drafts for count in acceptance_counts
    ]
    mean_acceptance_length = 1 + (num_accepted / num_drafts)
    return {
        "num_drafts": num_drafts,
        "num_accepted": num_accepted,
        "acceptance_rate_per_pos": acceptance_rate_per_pos,
        "mean_acceptance_length": mean_acceptance_length,
    }


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
def test_ngram_acceptance_rate_prompts():
    prompts = []
    words = ["test", "temp", "hello", "where"]
    for i in range(len(words)):
        word = words[i]
        prompt = f"Please repeat the word '{word}' 50 times.\n"
        prompt += "Here is an example of how it should look like: " + " ".join(
            [word] * 10) + "...\n"
        prompt += "Give no other output than the word at least "
        prompt += "fifty times in a row in lowercase "
        prompt += "with spaces between each word and without quotes."
        prompts.append([{"role": "user", "content": prompt}])
    return prompts


@pytest.fixture
def test_draft_acceptance_rate_prompts():
    prompts = [
        "Please write a short story about a cat that loves to chase mice.",
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
        "Describe the process of photosynthesis in plants.",
        "What are the main ingredients in a traditional pizza?",
    ]
    return [[{"role": "user", "content": prompt}] for prompt in prompts]


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

        # Heuristic: expect at least 65% of the prompts to match exactly
        # Upon failure, inspect the outputs to check for inaccuracy.
        assert matches > int(0.65 * (matches + misses))
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

        # Heuristic: expect at least 65% of the prompts to match exactly
        # Upon failure, inspect the outputs to check for inaccuracy.
        assert matches > int(0.65 * len(ref_outputs))
        del spec_llm


def test_ngram_acceptance_rate(
    monkeypatch: pytest.MonkeyPatch,
    test_ngram_acceptance_rate_prompts: list[list[dict[str, Any]]],
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Test the acceptance rate of speculative decoding using ngram method.
    The acceptance rate should be very high on the sample prompts,
    as they are designed for 100% matches with the ngram method.
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        spec_llm = LLM(
            model=model_name,
            speculative_config={
                "method": "ngram",
                "prompt_lookup_max": 5,
                "prompt_lookup_min": 3,
                "num_speculative_tokens": 3,
            },
            max_model_len=1024,
            disable_log_stats=False,
        )
        sampling_config.max_tokens = 50
        spec_llm.chat(test_ngram_acceptance_rate_prompts, sampling_config)

        metrics = get_spec_acceptance_metrics(spec_llm.get_metrics(), k=3)

        # Expect nearly all (90%) of drafted tokens to be accepted
        mean_acceptance_rate = np.mean(metrics["acceptance_rate_per_pos"])
        assert mean_acceptance_rate > 0.90

        # Expect the average acceptance length to be greater than 3
        assert metrics["mean_acceptance_length"] > 3

        del spec_llm


@pytest.mark.parametrize("use_eagle3", [False, True], ids=["eagle", "eagle3"])
def test_eagle_acceptance_rate(
    monkeypatch: pytest.MonkeyPatch,
    test_draft_acceptance_rate_prompts: list[dict[str, Any]],
    sampling_config: SamplingParams,
    model_name: str,
    use_eagle3: bool,
):
    '''
    Test the acceptance rate of speculative decoding using EAGLE methods.
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

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
            disable_log_stats=False,
        )
        sampling_config.max_tokens = 50
        spec_llm.chat(test_draft_acceptance_rate_prompts, sampling_config)

        metrics = get_spec_acceptance_metrics(spec_llm.get_metrics(), k=3)

        # Expect many of drafted tokens to be accepted
        if use_eagle3:
            # EAGLE3 is more accurate, so we expect a higher acceptance rate
            assert metrics["acceptance_rate_per_pos"][0] > 0.75
            assert metrics["acceptance_rate_per_pos"][2] > 0.4
            assert metrics["mean_acceptance_length"] > 2.75
        else:
            assert metrics["acceptance_rate_per_pos"][0] > 0.6
            assert metrics["acceptance_rate_per_pos"][2] > 0.2
            assert metrics["mean_acceptance_length"] > 2

        del spec_llm
