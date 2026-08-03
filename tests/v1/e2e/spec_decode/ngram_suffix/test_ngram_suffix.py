# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.utils import single_gpu_only
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

from ..utils import evaluate_llm_for_gsm8k, get_test_prompts


@pytest.fixture
def disable_vllm_compile_cache_on_rocm(request: pytest.FixtureRequest) -> None:
    if current_platform.is_rocm():
        request.getfixturevalue("disable_vllm_compile_cache")


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
@pytest.mark.usefixtures("disable_vllm_compile_cache_on_rocm")
@single_gpu_only
def test_ngram_and_suffix_correctness(
    speculative_config: dict,
    model_name: str,
    vllm_runner,
):
    with vllm_runner(
        model_name,
        # Keep LLM defaults; VllmRunner only provides lifecycle cleanup here.
        trust_remote_code=False,
        enable_chunked_prefill=None,
        speculative_config=speculative_config,
        max_model_len=4096,
        # Preserve LLM's default compilation/cudagraph configuration. Without
        # this, VllmRunner injects its reduced test-only capture sizes.
        compilation_config=CompilationConfig(),
    ) as runner:
        evaluate_llm_for_gsm8k(runner.llm)


@pytest.mark.parametrize("async_scheduling", [True], ids=["async"])
@single_gpu_only
def test_ngram_gpu_default_with_async_scheduling(
    async_scheduling: bool,
):
    """
    Test ngram_gpu speculative decoding (k=3) correctness with and without
    async scheduling, validated via GSM8K accuracy.
    Uses Qwen/Qwen3-8B (ref GSM8K accuracy: 87%-92%).
    """
    qwen3_model = "Qwen/Qwen3-8B"
    spec_llm = LLM(
        model=qwen3_model,
        speculative_config={
            "method": "ngram_gpu",
            "prompt_lookup_max": 3,
            "prompt_lookup_min": 2,
            "num_speculative_tokens": 2,
        },
        max_model_len=4096,
        async_scheduling=async_scheduling,
    )
    # Assert the resolved async_scheduling config matches what was requested.
    assert (
        spec_llm.llm_engine.vllm_config.scheduler_config.async_scheduling
        == async_scheduling
    )
    evaluate_llm_for_gsm8k(spec_llm, expected_accuracy_threshold=0.8)
    del spec_llm
    cleanup_dist_env_and_memory()


@single_gpu_only
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

    # Heuristic: expect at least 80.0% acceptance rate at the end.
    assert last_accept_rate > 0.80

    del spec_llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()
