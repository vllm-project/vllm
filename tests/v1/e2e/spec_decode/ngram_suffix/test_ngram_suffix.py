# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.platforms import current_platform
from vllm.v1.metrics.reader import Vector

from ..utils import (
    evaluate_llm_for_gsm8k,
    get_spec_decode_metric_value,
    get_test_prompts,
)


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


def _spec_decode_counts(metrics) -> tuple[float, float, list[int]]:
    per_pos = next(
        m for m in metrics if m.name == "vllm:spec_decode_num_accepted_tokens_per_pos"
    )
    assert isinstance(per_pos, Vector)
    return (
        get_spec_decode_metric_value(metrics, "vllm:spec_decode_num_drafts"),
        get_spec_decode_metric_value(metrics, "vllm:spec_decode_num_draft_tokens"),
        per_pos.values,
    )


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@single_gpu_only
def test_ngram_gpu_max_speculative_tokens(
    async_scheduling: bool,
    model_name: str,
    vllm_runner,
):
    """A per-request cap bounds the drafts verified per step: cap 1 accepts
    only at position 0, cap 0 verifies no drafts, while uncapped requests on
    the same engine accept beyond position 0."""
    num_spec = 3
    texts = [
        " ".join(f"item-{i}-{j} costs {i * j % 97} dollars." for j in range(24))
        for i in range(8)
    ]
    prompts = [f"Repeat this list exactly:\n{t}\nRepeated list:\n{t}" for t in texts]
    with vllm_runner(
        model_name,
        block_size=None,
        trust_remote_code=False,
        speculative_config={
            "method": "ngram_gpu",
            "prompt_lookup_max": 4,
            "prompt_lookup_min": 2,
            "num_speculative_tokens": num_spec,
        },
        max_model_len=2048,
        async_scheduling=async_scheduling,
        disable_log_stats=False,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
    ) as runner:
        llm = runner.llm

        def run(cap: int | None) -> tuple[float, float, list[int]]:
            drafts0, tokens0, pos0 = _spec_decode_counts(llm.get_metrics())
            params = SamplingParams(
                temperature=0, max_tokens=96, max_speculative_tokens=cap
            )
            llm.generate(prompts, params)
            drafts, tokens, pos = _spec_decode_counts(llm.get_metrics())
            return (
                drafts - drafts0,
                tokens - tokens0,
                [b - a for a, b in zip(pos0, pos)],
            )

        _, _, uncapped_pos = run(None)
        capped_drafts, capped_tokens, capped_pos = run(1)
        off_drafts, off_tokens, _ = run(0)

    assert sum(uncapped_pos[1:]) > 0, uncapped_pos
    assert capped_drafts > 0 and capped_tokens == capped_drafts
    assert capped_pos[0] > 0 and sum(capped_pos[1:]) == 0, capped_pos
    assert off_drafts == 0 and off_tokens == 0


@single_gpu_only
def test_suffix_decoding_acceptance(
    sampling_config: SamplingParams,
    model_name: str,
    vllm_runner,
):
    """Check that suffix decoding caching takes effect and improves acceptance
    lengths and acceptance rates over multiple runs of the same prompts.
    """
    test_prompts = get_test_prompts(mm_enabled=False)

    with vllm_runner(
        model_name,
        block_size=None,
        trust_remote_code=False,
        speculative_config={
            "method": "suffix",
            "suffix_decoding_max_spec_factor": 2.0,
            "suffix_decoding_max_cached_requests": 1000,
        },
        max_model_len=1024,
        disable_log_stats=False,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
    ) as spec_runner:
        # Run several times and check that the accepted tokens increase.
        num_draft = []
        num_accept = []
        for _ in range(10):  # Run multiple times to warm up the cache.
            spec_runner.llm.chat(test_prompts, sampling_config)
            # Collect draft and acceptance stats.
            metrics = spec_runner.llm.get_metrics()
            num_draft.append(
                get_spec_decode_metric_value(
                    metrics, "vllm:spec_decode_num_draft_tokens"
                )
            )
            num_accept.append(
                get_spec_decode_metric_value(
                    metrics, "vllm:spec_decode_num_accepted_tokens"
                )
            )

    # Calculate the acceptance rates for the first and last runs.
    first_accept_tokens = num_accept[0]
    first_draft_tokens = num_draft[0]
    assert first_draft_tokens > 0, (
        "Suffix decoder produced no draft tokens on the first run: "
        f"accepted={first_accept_tokens}, drafted={first_draft_tokens}"
    )
    first_accept_rate = first_accept_tokens / first_draft_tokens

    # Take the diff since the stats are cumulative.
    last_accept_tokens = num_accept[-1] - num_accept[-2]
    last_draft_tokens = num_draft[-1] - num_draft[-2]
    assert last_draft_tokens > 0, (
        "Suffix decoder produced no draft tokens on the last run: "
        f"accepted_delta={last_accept_tokens}, drafted_delta={last_draft_tokens}; "
        f"cumulative_drafted={num_draft[-2:]}"
    )
    last_accept_rate = last_accept_tokens / last_draft_tokens
    summary = (
        f"first accepted/drafted={first_accept_tokens}/{first_draft_tokens} "
        f"(rate={first_accept_rate:.3f}); last delta accepted/drafted="
        f"{last_accept_tokens}/{last_draft_tokens} (rate={last_accept_rate:.3f})"
    )

    # Expect the acceptance length to improve.
    assert first_accept_tokens < last_accept_tokens, (
        f"Expected accepted tokens to increase after cache warmup; {summary}"
    )

    # Expect the acceptance rate to improve.
    assert first_accept_rate < last_accept_rate, (
        f"Expected acceptance rate to increase after cache warmup; {summary}"
    )

    # Heuristic: expect at least 80.0% acceptance rate at the end.
    assert last_accept_rate > 0.80, f"Expected final acceptance rate > 0.80; {summary}"
