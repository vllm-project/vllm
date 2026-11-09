# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.platforms import current_platform

from ..utils import (
    evaluate_llm_for_gsm8k,
    get_spec_decode_metric_value,
    get_test_prompts,
)

if TYPE_CHECKING:
    from vllm.v1.spec_decode.dynamic.utils import DynamicSDSchedule


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
        {"method": "suffix"},
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
        block_size=None,
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
    vllm_runner,
):
    """Test ngram_gpu speculative decoding (k=3) correctness with and without
    async scheduling, validated via GSM8K accuracy.
    Uses Qwen/Qwen3-8B (ref GSM8K accuracy: 87%-92%).
    """
    qwen3_model = "Qwen/Qwen3-8B"
    with vllm_runner(
        qwen3_model,
        block_size=None,
        trust_remote_code=False,
        speculative_config={
            "method": "ngram_gpu",
            "prompt_lookup_max": 3,
            "prompt_lookup_min": 2,
            "num_speculative_tokens": 2,
        },
        max_model_len=4096,
        async_scheduling=async_scheduling,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
    ) as spec_runner:
        # Assert the resolved async_scheduling config matches what was requested.
        assert (
            spec_runner.llm.llm_engine.vllm_config.scheduler_config.async_scheduling
            == async_scheduling
        )
        evaluate_llm_for_gsm8k(spec_runner.llm, expected_accuracy_threshold=0.8)


@single_gpu_only
def test_suffix_cross_request_acceptance(
    sampling_config: SamplingParams,
    model_name: str,
    vllm_runner,
):
    """Repeating the same prompts should let suffix decoding draft the earlier
    responses from its corpus of finished requests."""
    test_prompts = get_test_prompts(mm_enabled=False)
    k = 8

    with vllm_runner(
        model_name,
        block_size=None,
        trust_remote_code=False,
        speculative_config={"method": "suffix", "num_speculative_tokens": k},
        max_model_len=1024,
        disable_log_stats=False,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
    ) as spec_runner:
        num_drafts = []
        num_accept = []
        for _ in range(2):
            spec_runner.llm.chat(test_prompts, sampling_config)
            metrics = spec_runner.llm.get_metrics()
            num_drafts.append(
                get_spec_decode_metric_value(metrics, "vllm:spec_decode_num_drafts")
            )
            num_accept.append(
                get_spec_decode_metric_value(
                    metrics, "vllm:spec_decode_num_accepted_tokens"
                )
            )

    # Stats are cumulative; compare per-draft acceptance of the two runs.
    first = num_accept[0] / num_drafts[0]
    second = (num_accept[1] - num_accept[0]) / (num_drafts[1] - num_drafts[0])
    summary = f"accepted per draft: first={first:.2f}, second={second:.2f}"
    # Without the corpus, the repeated pass stays close to the first.
    assert second > 2 * first, summary


def _dynamic_sd_ngram_config(
    schedule: DynamicSDSchedule,
) -> dict[str, str | int | DynamicSDSchedule]:
    return {
        "method": "ngram",
        "prompt_lookup_max": 3,
        "prompt_lookup_min": 2,
        "num_speculative_tokens": 3,
        "num_speculative_tokens_per_batch_size": schedule,
    }


@single_gpu_only
def test_dynamic_sd_invalid_schedule_fails_at_config_time(vllm_runner):
    """An invalid schedule raises the config error itself, not an engine
    startup failure after the model has loaded."""
    with (
        pytest.raises(ValueError, match="non-overlapping"),
        vllm_runner(
            "Qwen/Qwen3-0.6B",
            speculative_config=_dynamic_sd_ngram_config([(1, 16, 3), (16, 32, 2)]),
        ),
    ):
        pass


@single_gpu_only
def test_dynamic_sd_empty_schedule_falls_back_to_static_k(
    sampling_config: SamplingParams,
    vllm_runner,
):
    """An empty schedule serves with the static num_speculative_tokens."""
    with vllm_runner(
        "Qwen/Qwen3-0.6B",
        speculative_config=_dynamic_sd_ngram_config([]),
        disable_log_stats=False,
    ) as runner:
        speculative_config = runner.llm.llm_engine.vllm_config.speculative_config
        assert speculative_config.num_speculative_tokens_per_batch_size is None

        runner.llm.chat(
            get_test_prompts(mm_enabled=False, num_prompts=8), sampling_config
        )
        metrics = runner.llm.get_metrics()
        num_drafts = get_spec_decode_metric_value(
            metrics, "vllm:spec_decode_num_drafts"
        )
        num_draft_tokens = get_spec_decode_metric_value(
            metrics, "vllm:spec_decode_num_draft_tokens"
        )
        assert num_drafts > 0
        assert num_draft_tokens == 3 * num_drafts
