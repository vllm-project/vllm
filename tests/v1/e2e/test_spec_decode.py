# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import random
from typing import Any, Union

import pytest
import torch

from tests.utils import get_attn_backend_list_based_on_platform, large_gpu_mark
from vllm import LLM, SamplingParams
from vllm.assets.base import VLLM_S3_BUCKET_URL
from vllm.assets.image import VLM_IMAGES_DIR
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

MTP_SIMILARITY_RATE = 0.8


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
    Compare the outputs of an original LLM and a speculative LLM
    should be the same when using ngram speculative decoding.
    '''
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

    # Heuristic: expect at least 66% of the prompts to match exactly
    # Upon failure, inspect the outputs to check for inaccuracy.
    assert matches >= int(0.66 * len(ref_outputs))
    del spec_llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()


def test_suffix_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Compare the outputs of an original LLM and a speculative LLM
    should be the same when using suffix speculative decoding.
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        # Create test prompts with repetitive patterns that suffix decode
        # can leverage
        test_prompts = []

        # Add prompts with repetitive patterns
        repetitive_prompts = [
            # Code-like patterns
            [{
                "role":
                "user",
                "content":
                "Write a Python function that prints numbers 1 to 10, "
                "each on a new line using a for loop."
            }],
            [{
                "role":
                "user",
                "content":
                "Create a list of dictionaries where each dictionary "
                "has 'id' and 'name' keys for 5 users."
            }],
            # Repetitive text patterns
            [{
                "role":
                "user",
                "content":
                "List the days of the week, each followed by a colon "
                "and the word 'workday' or 'weekend'."
            }],
            [{
                "role":
                "user",
                "content":
                "Generate a multiplication table for 5 "
                "(5x1=5, 5x2=10, etc.) up to 5x5."
            }],
            # Template-like patterns
            [{
                "role":
                "user",
                "content":
                "Create 3 email signatures, each with Name, Title, "
                "Company, and Email format."
            }],
        ]

        # Add some of the original test prompts for variety
        original_prompts = get_test_prompts(mm_enabled=False)[:20]

        test_prompts = repetitive_prompts + original_prompts

        ref_llm = LLM(model=model_name, max_model_len=1024)
        ref_outputs = ref_llm.chat(test_prompts, sampling_config)
        del ref_llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()

        spec_llm = LLM(
            model=model_name,
            speculative_config={
                "method": "suffix",
                "num_speculative_tokens": 8,
                "suffix_cache_max_depth": 64,
                "suffix_cache_max_requests": 1000,
                "suffix_min_token_prob": 0.1,
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

        # Suffix decode should maintain correctness
        # We expect at least 75% match rate - suffix decode may have
        # slightly different token boundaries but should produce
        # semantically similar outputs
        assert matches >= int(0.75 * len(ref_outputs)), \
            f"Suffix decode correctness too low: " \
            f"{matches}/{len(ref_outputs)} matches"

        # Also ensure we have a reasonable number of matches
        assert matches >= 15, f"Too few matches: {matches}"
        del spec_llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "suffix_config",
    [
        # (num_speculative_tokens, cache_max_depth, min_token_prob)
        (4, 32, 0.1),  # Conservative configuration
        (8, 64, 0.1),  # Default configuration
        (16, 128, 0.05),  # Aggressive configuration
    ])
def test_suffix_with_configs(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
    suffix_config: tuple[int, int, float],
):
    '''
    Test suffix decode with different configurations to ensure
    correctness is maintained across various parameter settings.
    '''
    num_spec_tokens, max_depth, min_prob = suffix_config

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        # Use a smaller set of prompts for parametrized tests
        test_prompts = [
            # Highly repetitive pattern
            [{
                "role":
                "user",
                "content":
                "Count from 1 to 20, writing each number on a new line."
            }],
            # Code pattern
            [{
                "role":
                "user",
                "content":
                "Write a for loop that prints 'Hello World' "
                "5 times."
            }],
            # Mixed pattern
            [{
                "role":
                "user",
                "content":
                "List three colors and their RGB values in format: "
                "Color: R=X, G=Y, B=Z"
            }],
        ]

        ref_llm = LLM(model=model_name, max_model_len=512)
        ref_outputs = ref_llm.chat(test_prompts, sampling_config)
        del ref_llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()

        spec_llm = LLM(
            model=model_name,
            speculative_config={
                "method": "suffix",
                "num_speculative_tokens": num_spec_tokens,
                "suffix_cache_max_depth": max_depth,
                "suffix_cache_max_requests": 100,
                "suffix_min_token_prob": min_prob,
            },
            max_model_len=512,
        )
        spec_outputs = spec_llm.chat(test_prompts, sampling_config)

        # Verify all outputs match exactly
        for i, (ref_output,
                spec_output) in enumerate(zip(ref_outputs, spec_outputs)):
            assert ref_output.outputs[0].text == spec_output.outputs[0].text, \
                f"Mismatch with config {suffix_config} on prompt {i}: " \
                f"ref='{ref_output.outputs[0].text}' vs " \
                f"spec='{spec_output.outputs[0].text}'"

        del spec_llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled"],
    [
        (("eagle3", "Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3", 1), False),
        pytest.param(("eagle3", "Qwen/Qwen2.5-VL-7B-Instruct",
                      "Rayzl/qwen2.5-vl-7b-eagle3-sgl", 1),
                     False,
                     marks=pytest.mark.skip(reason="Skipping due to its " \
                               "head_dim not being a a multiple of 32")),
        (("eagle", "meta-llama/Llama-3.1-8B-Instruct",
          "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", 1), False),
        (("eagle3", "meta-llama/Llama-3.1-8B-Instruct",
          "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", 1), False),
        pytest.param(("eagle", "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                      "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct", 4),
                     False,
                     marks=large_gpu_mark(min_gb=80)),  # works on 4x H100
        pytest.param(("eagle", "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                      "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct", 4),
                     True,
                     marks=large_gpu_mark(min_gb=80)),  # works on 4x H100
        (("eagle", "eagle618/deepseek-v3-random",
          "eagle618/eagle-deepseek-v3-random", 1), False),
    ],
    ids=[
        "qwen3_eagle3", "qwen2_5_vl_eagle3", "llama3_eagle", "llama3_eagle3",
        "llama4_eagle", "llama4_eagle_mm", "deepseek_eagle"
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
    if attn_backend == "TREE_ATTN":
        # TODO: Fix this flaky test
        pytest.skip(
            "TREE_ATTN is flaky in the test disable for now until it can be "
            "resolved (see https://github.com/vllm-project/vllm/issues/22922)")

    # Generate test prompts inside the function instead of using fixture
    test_prompts = get_test_prompts(mm_enabled)
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using eagle speculative decoding.
    model_setup: (method, model_name, eagle_model_name, tp_size)
    '''
    with monkeypatch.context() as m:
        if "Llama-4-Scout" in model_setup[1] and attn_backend == "FLASH_ATTN":
            # Scout requires default backend selection
            # because vision encoder has head_dim 88 being incompatible
            #  with FLASH_ATTN and needs to fall back to Flex Attn
            pass
        else:
            m.setenv("VLLM_MLA_DISABLE", "1")
            m.setenv("VLLM_ATTENTION_BACKEND", attn_backend)

        if (attn_backend == "TRITON_ATTN" and not current_platform.is_rocm()):
            pytest.skip("TRITON_ATTN does not support "
                        "multi-token eagle spec decode on current platform")

        if attn_backend == "FLASH_ATTN" and current_platform.is_rocm():
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


@pytest.mark.parametrize(["model_setup", "mm_enabled"], [
    (("mtp", "XiaomiMiMo/MiMo-7B-Base", 1), False),
    (("mtp", "ZixiQi/DeepSeek-V3-4layers-MTP-FP8", 1), False),
],
                         ids=["mimo", "deepseek"])
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int],
    mm_enabled: bool,
):
    # Generate test prompts inside the function instead of using fixture
    test_prompts = get_test_prompts(mm_enabled)
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using MTP speculative decoding.
    model_setup: (method, model_name, tp_size)
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_MLA_DISABLE", "1")

        method, model_name, tp_size = model_setup

        ref_llm = LLM(model=model_name,
                      max_model_len=2048,
                      tensor_parallel_size=tp_size,
                      trust_remote_code=True)
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
                "num_speculative_tokens": 1,
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

        # Heuristic: expect at least 80% of the prompts to match exactly
        # Upon failure, inspect the outputs to check for inaccuracy.
        assert matches > int(MTP_SIMILARITY_RATE * len(ref_outputs))
        del spec_llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()
