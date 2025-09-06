# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Union

import pytest
import torch

from tests.utils import get_attn_backend_list_based_on_platform
from vllm import LLM, SamplingParams
from vllm.assets.base import VLLM_S3_BUCKET_URL
from vllm.assets.image import VLM_IMAGES_DIR
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.v1.spec_decode.metrics import compute_acceptance_rate


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
    return greedy_sampling()


def greedy_sampling() -> SamplingParams:
    return SamplingParams(temperature=0, max_tokens=10, ignore_eos=False)


def stochastic_sampling() -> SamplingParams:
    return SamplingParams(temperature=1.0, max_tokens=10, ignore_eos=False)


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
        (("eagle", "eagle618/deepseek-v3-random",
          "eagle618/eagle-deepseek-v3-random", 1), False),
    ],
    ids=[
        # TODO: Re-enable this once tests/models/test_initialization.py is fixed, see PR #22333 #22611  # noqa: E501
        # "qwen3_eagle3",
        "llama3_eagle",
        "llama3_eagle3",
        "llama4_eagle",
        "llama4_eagle_mm",
        "deepseek_eagle"
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
            "reolved (see https://github.com/vllm-project/vllm/issues/22922)")

    # Generate test prompts inside the function instead of using fixture
    test_prompts = get_test_prompts(mm_enabled)
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using eagle speculative decoding.
    model_setup: (method, model_name, eagle_model_name, tp_size)
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_MLA_DISABLE", "1")
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


@dataclass
class ArgsTest:
    model: str
    draft_model: str
    sampling_config: SamplingParams
    expected_acceptance_rate: float
    expected_same_output_fraction: float
    # Defaults
    enforce_eager: bool = True
    max_model_len: int = 1024
    gpu_memory_utilization: float = 0.5


cases = [
    ArgsTest(
        model="baidu/ERNIE-4.5-0.3B-PT",
        draft_model="baidu/ERNIE-4.5-0.3B-PT",
        sampling_config=greedy_sampling(),
        expected_acceptance_rate=1.0,
        expected_same_output_fraction=1.0,
    ),
    ArgsTest(
        model="baidu/ERNIE-4.5-0.3B-PT",
        draft_model="baidu/ERNIE-4.5-0.3B-PT",
        sampling_config=stochastic_sampling(),
        expected_acceptance_rate=0.2,
        expected_same_output_fraction=0.0,
    ),
    ArgsTest(
        model="meta-llama/Llama-3.2-1B-Instruct",
        draft_model="meta-llama/Llama-3.2-1B-Instruct",
        sampling_config=greedy_sampling(),
        expected_acceptance_rate=0.8,
        expected_same_output_fraction=0.5,
    ),
    ArgsTest(
        model="meta-llama/Llama-3.2-1B-Instruct",
        draft_model="meta-llama/Llama-3.2-1B-Instruct",
        sampling_config=stochastic_sampling(),
        expected_acceptance_rate=0.4,
        expected_same_output_fraction=0.15,
    ),
    ArgsTest(
        model="Qwen/Qwen3-1.7B",
        draft_model="Qwen/Qwen3-0.6B",
        sampling_config=greedy_sampling(),
        expected_acceptance_rate=1.0,
        expected_same_output_fraction=1.0,
    ),
    ArgsTest(
        model="Qwen/Qwen3-1.7B",
        draft_model="Qwen/Qwen3-0.6B",
        sampling_config=stochastic_sampling(),
        expected_acceptance_rate=0.9,
        expected_same_output_fraction=0.9,
    ),
]


@pytest.mark.parametrize("args", cases)
def test_draft_model_correctness(args: ArgsTest,
                                 monkeypatch: pytest.MonkeyPatch):
    """Compare the outputs using and not using speculative decoding.
    In the greedy decoding case, the outputs must match EXACTLY."""
    monkeypatch.setenv("VLLM_USE_V1", "1")
    test_prompts = get_test_prompts(mm_enabled=False)

    spec_llm = LLM(
        model=args.model,
        speculative_config={
            "model": args.draft_model,
            "method": "draft_model",
            "num_speculative_tokens": 3,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
        },
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,  # enables get_metrics()
    )
    spec_outputs = spec_llm.chat(test_prompts, args.sampling_config)
    acceptance_rate = compute_acceptance_rate(spec_llm.get_metrics())
    del spec_llm  # CLEANUP
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    assert acceptance_rate >= args.expected_acceptance_rate

    ref_llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    ref_outputs = ref_llm.chat(test_prompts, args.sampling_config)
    del ref_llm  # CLEANUP
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    assert len(ref_outputs) > 0
    assert len(ref_outputs) == len(spec_outputs)

    match_fraction = compute_exact_matches(ref_outputs, spec_outputs)
    assert match_fraction >= args.expected_same_output_fraction

    print(f"spec-decode: target={args.model}, draft={args.draft_model}, "
          f"temperature={args.sampling_config.temperature:.2f}, "
          f"acceptance_rate={acceptance_rate:.2f}, "
          f"match_fraction={match_fraction:.2f}")


def compute_exact_matches(ref_outputs: list[RequestOutput],
                          spec_outputs: list[RequestOutput]) -> float:
    """Compute the fraction of the prompts that match exactly"""
    assert len(ref_outputs) == len(spec_outputs)
    matches = 0
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        if ref_output.outputs[0].text == spec_output.outputs[0].text:
            matches += 1
        else:
            print(f"ref_output: {ref_output.outputs[0].text}")
            print(f"spec_output: {spec_output.outputs[0].text}")
    return matches / len(ref_outputs)
