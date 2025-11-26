# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
from typing import Any

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
        prompt: str | list[dict[str, Any]] = ""
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
            placeholders = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{VLLM_S3_BUCKET_URL}/{VLM_IMAGES_DIR}/stop_sign.jpg"
                    },
                }
            ]
            prompt = [
                *placeholders,
                {"type": "text", "text": "The meaning of the image is"},
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


@pytest.fixture(autouse=True)
def reset_torch_dynamo():
    """Reset torch dynamo cache before each test"""
    yield
    # Cleanup after test
    torch._dynamo.reset()


@pytest.mark.parametrize(
    "speculative_config",
    [
        {
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": 3,
        },
        {
            "method": "suffix",
            "suffix_decoding_max_spec_factor": 2.0,
        },
    ],
)
def test_ngram_and_suffix_correctness(
    speculative_config: dict,
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    """
    Compare the outputs of an original LLM and a speculative LLM
    should be the same when using ngram speculative decoding.
    """
    test_prompts = get_test_prompts(mm_enabled=False)

    ref_llm = LLM(model=model_name, max_model_len=1024)
    ref_outputs = ref_llm.chat(test_prompts, sampling_config)
    del ref_llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    spec_llm = LLM(
        model=model_name,
        speculative_config=speculative_config,
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


def test_suffix_decoding_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    """
    Check that suffix decoding caching takes effect and improves acceptance
    lengths and acceptance rates over multiple runs of the same prompts.
    """
    test_prompts = get_test_prompts(mm_enabled=False)

    spec_llm = LLM(
        model=model_name,
        speculative_config={
            "method": "suffix",
            "suffix_decoding_max_spec_factor": 2.0,
            "suffix_decoding_max_cached_requests": 1000,
        },
        max_model_len=1024,
        disable_log_stats=False,
    )

    # Run several times and check that the accepted tokens increase.
    num_draft = []
    num_accept = []
    for i in range(10):  # Run multiple times to warm up the cache.
        spec_llm.chat(test_prompts, sampling_config)
        # Collect draft and acceptance stats.
        metrics = spec_llm.get_metrics()
        for metric in metrics:
            if metric.name == "vllm:spec_decode_num_draft_tokens":
                num_draft.append(metric.value)
            if metric.name == "vllm:spec_decode_num_accepted_tokens":
                num_accept.append(metric.value)

    # Calculate the acceptance rates for the first and last runs.
    first_accept_tokens = num_accept[0]
    first_draft_tokens = num_draft[0]
    first_accept_rate = first_accept_tokens / first_draft_tokens

    # Take the diff since the stats are cumulative.
    last_accept_tokens = num_accept[-1] - num_accept[-2]
    last_draft_tokens = num_draft[-1] - num_draft[-2]
    last_accept_rate = last_accept_tokens / last_draft_tokens

    # Expect the acceptance length to improve.
    assert first_accept_tokens < last_accept_tokens

    # Expect the acceptance rate to improve.
    assert first_accept_rate < last_accept_rate

    # Heuristic: expect at least 85% acceptance rate at the end.
    assert last_accept_rate > 0.85

    del spec_llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "model_path",
    [
        "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        "RedHatAI/Qwen3-8B-speculator.eagle3",
    ],
    ids=["llama3_eagle3_speculator", "qwen3_eagle3_speculator"],
)
def test_speculators_model_integration(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_path: str,
):
    """
    Test that speculators models work with the simplified integration.

    This verifies the `vllm serve <speculator-model>` use case where
    speculative config is automatically detected from the model config
    without requiring explicit --speculative-config argument.

    Tests:
    1. Speculator model is correctly detected
    2. Verifier model is extracted from speculator config
    3. Speculative decoding is automatically enabled
    4. Text generation works correctly
    5. Output matches reference (non-speculative) generation
    """
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # Generate test prompts
    test_prompts = get_test_prompts(mm_enabled=False)

    # First run: Direct speculator model (simplified integration)
    spec_llm = LLM(model=model_path, max_model_len=1024)
    spec_outputs = spec_llm.chat(test_prompts, sampling_config)

    # Verify speculative config was auto-detected
    assert spec_llm.llm_engine.vllm_config.speculative_config is not None, (
        f"Speculative config should be auto-detected for {model_path}"
    )

    spec_config = spec_llm.llm_engine.vllm_config.speculative_config
    assert spec_config.num_speculative_tokens > 0, (
        f"Expected positive speculative tokens, "
        f"got {spec_config.num_speculative_tokens}"
    )

    # Verify draft model is set to the speculator model
    assert spec_config.model == model_path, (
        f"Draft model should be {model_path}, got {spec_config.model}"
    )

    # Extract verifier model for reference run
    verifier_model = spec_llm.llm_engine.vllm_config.model_config.model

    del spec_llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    # Second run: Reference without speculative decoding
    ref_llm = LLM(model=verifier_model, max_model_len=1024)
    ref_outputs = ref_llm.chat(test_prompts, sampling_config)
    del ref_llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    # Compare outputs
    matches = sum(
        1
        for ref, spec in zip(ref_outputs, spec_outputs)
        if ref.outputs[0].text == spec.outputs[0].text
    )

    # Heuristic: expect at least 66% of prompts to match exactly
    assert matches >= int(0.66 * len(ref_outputs)), (
        f"Only {matches}/{len(ref_outputs)} outputs matched. "
        f"Expected at least {int(0.66 * len(ref_outputs))} matches."
    )


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "enable_chunked_prefill"],
    [
        (("eagle3", "Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3", 1), False, False),
        pytest.param(
            (
                "eagle3",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "Rayzl/qwen2.5-vl-7b-eagle3-sgl",
                1,
            ),
            False,
            False,
            marks=pytest.mark.skip(
                reason="Skipping due to its head_dim not being a a multiple of 32"
            ),
        ),
        pytest.param(
            (
                "eagle",
                "meta-llama/Llama-3.1-8B-Instruct",
                "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
                1,
            ),
            False,
            True,
            marks=large_gpu_mark(min_gb=40),
        ),  # works on 4x H100
        (
            (
                "eagle3",
                "meta-llama/Llama-3.1-8B-Instruct",
                "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                1,
            ),
            False,
            False,
        ),
        pytest.param(
            (
                "eagle",
                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct",
                4,
            ),
            False,
            False,
            marks=large_gpu_mark(min_gb=80),
        ),  # works on 4x H100
        pytest.param(
            (
                "eagle",
                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct",
                4,
            ),
            True,
            True,
            marks=large_gpu_mark(min_gb=80),
        ),  # works on 4x H100
        (
            (
                "eagle",
                "eagle618/deepseek-v3-random",
                "eagle618/eagle-deepseek-v3-random",
                1,
            ),
            False,
            False,
        ),
    ],
    ids=[
        "qwen3_eagle3",
        "qwen2_5_vl_eagle3",
        "llama3_eagle",
        "llama3_eagle3",
        "llama4_eagle",
        "llama4_eagle_mm",
        "deepseek_eagle",
    ],
)
@pytest.mark.parametrize("attn_backend", get_attn_backend_list_based_on_platform())
def test_eagle_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, str, int],
    mm_enabled: bool,
    enable_chunked_prefill: bool,
    attn_backend: str,
):
    if attn_backend == "TREE_ATTN":
        # TODO: Fix this flaky test
        pytest.skip(
            "TREE_ATTN is flaky in the test disable for now until it can be "
            "resolved (see https://github.com/vllm-project/vllm/issues/22922)"
        )

    # Generate test prompts inside the function instead of using fixture
    test_prompts = get_test_prompts(mm_enabled)
    """
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using eagle speculative decoding.
    model_setup: (method, model_name, eagle_model_name, tp_size)
    """
    with monkeypatch.context() as m:
        if "Llama-4-Scout" in model_setup[1] and attn_backend == "FLASH_ATTN":
            # Scout requires default backend selection
            # because vision encoder has head_dim 88 being incompatible
            #  with FLASH_ATTN and needs to fall back to Flex Attn
            pass
        else:
            m.setenv("VLLM_MLA_DISABLE", "1")
            m.setenv("VLLM_ATTENTION_BACKEND", attn_backend)

        if attn_backend == "TRITON_ATTN" and not current_platform.is_rocm():
            pytest.skip(
                "TRITON_ATTN does not support "
                "multi-token eagle spec decode on current platform"
            )

        if attn_backend == "FLASH_ATTN" and current_platform.is_rocm():
            m.setenv("VLLM_ROCM_USE_AITER", "1")

        method, model_name, spec_model_name, tp_size = model_setup
        max_model_len = 2048
        max_num_batched_tokens = 128 if enable_chunked_prefill else max_model_len

        ref_llm = LLM(
            model=model_name, max_model_len=max_model_len, tensor_parallel_size=tp_size
        )
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
                "max_model_len": max_model_len,
            },
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
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

        # Heuristic: expect at least 60% of the prompts to match exactly
        # Upon failure, inspect the outputs to check for inaccuracy.
        assert matches > int(0.6 * len(ref_outputs))
        del spec_llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled"],
    [
        (("mtp", "XiaomiMiMo/MiMo-7B-Base", 1), False),
        (("mtp", "ZixiQi/DeepSeek-V3-4layers-MTP-FP8", 1), False),
    ],
    ids=["mimo", "deepseek"],
)
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int],
    mm_enabled: bool,
):
    # Generate test prompts inside the function instead of using fixture
    test_prompts = get_test_prompts(mm_enabled)
    """
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using MTP speculative decoding.
    model_setup: (method, model_name, tp_size)
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_MLA_DISABLE", "1")

        method, model_name, tp_size = model_setup

        ref_llm = LLM(
            model=model_name,
            max_model_len=2048,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
        )
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
