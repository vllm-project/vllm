# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import check_logprobs_close
from tests.utils import single_gpu_only
from vllm import SamplingParams

from ...utils import compute_acceptance_rate
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
def test_gemma4_mtp_fast_prefill(monkeypatch, vllm_runner):
    """Skipping prompt rows keeps target outputs and draft acceptance intact."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    speculative_config = {
        "method": "mtp",
        "model": "google/gemma-4-E2B-it-assistant",
        "num_speculative_tokens": 2,
    }
    prompts = [
        "Calculate 17 times 23. Explain your calculation.",
        "Write a Python function that reverses a singly linked list.",
        "Background: "
        + "The cache stores keys and values. " * 200
        + "Explain why prefill can delay other requests.",
    ]
    results = []
    acceptance_rates = []
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
            max_num_seqs=4,
            enable_prefix_caching=False,
            disable_log_stats=False,
            seed=42,
        ) as runner:
            tokenizer = runner.llm.get_tokenizer()
            formatted = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]
            results.append(
                runner.generate_w_logprobs(
                    formatted,
                    SamplingParams(
                        temperature=0.0, max_tokens=64, ignore_eos=True, logprobs=5
                    ),
                )
            )
            acceptance_rates.append(compute_acceptance_rate(runner.llm.get_metrics()))
    check_logprobs_close(
        outputs_0_lst=results[0],
        outputs_1_lst=results[1],
        name_0="MTP",
        name_1="MTP with fast prefill",
    )
    assert all(rate > 0 for rate in acceptance_rates)
    assert acceptance_rates[1] == pytest.approx(acceptance_rates[0], abs=0.05)
