# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams

from ...utils import assert_request_outputs_match, get_test_prompts
from .._correctness import check_mtp_correctness


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        (
            ("mtp", "google/gemma-4-E4B-it", 1, "google/gemma-4-E4B-it-assistant"),
            False,
            0.50,
        ),  # gemma4 MTP with assistant model, ref: ~62%
    ],
    ids=["gemma4-e4b"],
)
@single_gpu_only
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int, str | None],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    vllm_runner,
):
    check_mtp_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        vllm_runner,
    )


@single_gpu_only
def test_gemma4_mtp_fast_prefill(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    vllm_runner,
):
    """Skipping prompt rows keeps target outputs and draft acceptance intact."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    speculative_config = {
        "method": "mtp",
        "model": "google/gemma-4-E2B-it-assistant",
        "num_speculative_tokens": 2,
    }
    prompts = get_test_prompts(mm_enabled=False)
    long_prompt = (
        "Background: "
        + "The cache stores keys and values. " * 200
        + "Explain why prefill can delay other requests."
    )
    prompts.append([{"role": "user", "content": long_prompt}])
    results = []
    first_pos_acceptance_rates = []
    for fast_prefill in [False, True]:
        with vllm_runner(
            "google/gemma-4-E2B-it",
            speculative_config=speculative_config,
            kv_sharing_fast_prefill=fast_prefill,
            language_model_only=True,
            attention_backend="TRITON_ATTN",
            max_model_len=4096,
            block_size=None,
            enable_chunked_prefill=True,
            max_num_batched_tokens=512,
            enable_prefix_caching=False,
            disable_log_stats=False,
            seed=42,
        ) as runner:
            results.append(runner.llm.chat(prompts, sampling_config))
            metrics = {m.name: m for m in runner.llm.get_metrics()}
            first_pos_acceptance_rates.append(
                metrics["vllm:spec_decode_num_accepted_tokens_per_pos"].values[0]
                / metrics["vllm:spec_decode_num_drafts"].value
            )
    assert_request_outputs_match(
        results[0],
        results[1],
        required_matches=int(0.8 * len(results[0])) + 1,
        context="MTP with fast prefill",
    )
    assert first_pos_acceptance_rates[1] >= 0.9 * first_pos_acceptance_rates[0], (
        f"first-position acceptance rate {first_pos_acceptance_rates[1]:.3f} "
        f"vs {first_pos_acceptance_rates[0]:.3f}"
    )
