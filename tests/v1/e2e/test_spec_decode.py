# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import random
from typing import Any, Union

import pytest
import torch

from tests.utils import get_attn_backend_list_based_on_platform
from vllm import LLM, SamplingParams
from vllm.assets.base import VLLM_S3_BUCKET_URL
from vllm.assets.image import VLM_IMAGES_DIR
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform


def get_test_prompts(mm_enabled: bool):
    prompt_types = ["repeat", "sentence"]
    if mm_enabled:
        prompt_types.append("mm")
    num_prompts = 100
    prompts = []

    random.seed(0)
    random_prompt_type_choices = random.choices(prompt_types, k=num_prompts)
    print(f"Prompt types: {random_prompt_type_choices}")

    # Generate a mixed batch of prompts, some of which can be easily
    # predicted by n-gram matching and some which likely cannot.
    for kind in random_prompt_type_choices:
        word_choices = ["test", "temp", "hello", "where"]
        word = random.choice(word_choices)
        prompt: Union[str, list[dict[str, Any]]] = ""
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
        elif kind == "mm":
            placeholders = [{
                "type": "image_url",
                "image_url": {
                    "url":
                    f"{VLLM_S3_BUCKET_URL}/{VLM_IMAGES_DIR}/stop_sign.jpg"
                },
            }]
            prompt = [
                *placeholders,
                {
                    "type": "text",
                    "text": "The meaning of the image is"
                },
            ]
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
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using ngram speculative decoding.
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        test_prompts = get_test_prompts(mm_enabled=False)

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


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled"],
    [
        # TODO: Re-enable this once tests/models/test_initialization.py is fixed, see PR #22333 #22611  # noqa: E501
        # (("eagle3", "Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3", 1), False),
        (("eagle", "meta-llama/Llama-3.1-8B-Instruct",
          "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", 1), False),
        (("eagle3", "meta-llama/Llama-3.1-8B-Instruct",
          "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", 1), False),
        pytest.param(
            ("eagle", "meta-llama/Llama-4-Scout-17B-16E-Instruct",
             "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct", 4),
            False,
            marks=pytest.mark.skip(reason="Skipping due to CI OOM issues")),
        pytest.param(
            ("eagle", "meta-llama/Llama-4-Scout-17B-16E-Instruct",
             "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct", 4),
            True,
            marks=pytest.mark.skip(reason="Skipping due to CI OOM issues")),
    ],
    ids=[
        "qwen3_eagle3", "llama3_eagle", "llama3_eagle3", "llama4_eagle",
        "llama4_eagle_mm"
    ])
@pytest.mark.parametrize("attn_backend",
                         get_attn_backend_list_based_on_platform())
def test_eagle_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, str, int],
    mm_enabled: bool,
    attn_backend: str,
):
    # Generate test prompts inside the function instead of using fixture
    test_prompts = get_test_prompts(mm_enabled)
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using eagle speculative decoding.
    model_setup: (method, model_name, eagle_model_name, tp_size)
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_ATTENTION_BACKEND", attn_backend)

        if (attn_backend == "TRITON_ATTN_VLLM_V1"
                and not current_platform.is_rocm()):
            pytest.skip("TRITON_ATTN_VLLM_V1 does not support "
                        "multi-token eagle spec decode on current platform")

        if attn_backend == "FLASH_ATTN_VLLM_V1" and current_platform.is_rocm():
            m.setenv("VLLM_ROCM_USE_AITER", "1")

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
