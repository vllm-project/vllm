# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory

from ...utils import compute_acceptance_len, compute_acceptance_rate


def test_gemma4_dspark_correctness_and_acceptance_rate(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    E2E test for Gemma4 DSpark speculative decoding: acceptance rate/length
    regression coverage plus GSM8K correctness, at temperature=1.0 to exercise
    the probabilistic draft-sampling/rejection-sampling path (not just greedy).

    gemma-4-12B is instruct-tuned, so GSM8K is run through the chat template
    (use_chat_completions=True; raw few-shot completion collapses to a few
    percent). Reference: measured over 5 runs of 200 GSM8K questions at
    temperature=1.0 (prefix caching disabled):
      accuracy:        min=0.900 max=0.955 mean=0.937
      acceptance_rate: min=0.578 max=0.595 mean=0.588
      acceptance_len:  min=5.044 max=5.167 mean=5.116
    Thresholds set conservatively to 10% to avoid flaking due to unlucky sampling
    """
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    spec_llm = LLM(
        model="RedHatAI/gemma-4-12B-it-NVFP4",
        trust_remote_code=True,
        speculative_config={
            "method": "dspark",
            "model": "deepseek-ai/dspark_gemma4_12b_block7",
            "num_speculative_tokens": 7,
            "draft_sample_method": "probabilistic",
        },
        max_model_len=4096,
        max_num_seqs=32,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
    )
    try:
        results = evaluate_gsm8k_offline(
            spec_llm, num_questions=200, temperature=1.0, use_chat_completions=True
        )
        gsm8k_accuracy = results["accuracy"]

        metrics = spec_llm.get_metrics()
        acceptance_rate = compute_acceptance_rate(metrics)
        acceptance_len = compute_acceptance_len(metrics)
        print(
            f"Gemma4 DSpark acceptance_rate={acceptance_rate:.2f}, "
            f"acceptance_len={acceptance_len:.2f}, "
            f"gsm8k_accuracy={gsm8k_accuracy:.3f}"
        )

        assert acceptance_rate >= 0.58 * 0.9
        assert acceptance_len >= 5.0 * 0.9
        assert gsm8k_accuracy >= 0.92 * 0.9
    finally:
        del spec_llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()


@pytest.fixture
def dspark_config():
    target_model = "Qwen/Qwen3-4B-FP8"
    draft_model = "deepseek-ai/dspark_qwen3_4b_block7"

    return dict(
        model=target_model,
        trust_remote_code=True,
        speculative_config={
            "method": "dspark",
            "model": draft_model,
            "num_speculative_tokens": 7,
            "attention_backend": "FLASH_ATTN",
            "draft_sample_method": "probabilistic",
        },
        max_model_len=4096,
        disable_log_stats=False,
    )


def test_dspark_correctness_and_acceptance_rate(dspark_config):
    """
    E2E test for DSpark speculative decoding: acceptance rate/length
    regression coverage plus GSM8K correctness, at temperature=1.0 to
    exercise the probabilistic draft-sampling/rejection-sampling path
    (not just greedy).

    Uses Qwen/Qwen3-4B-FP8 as target with the dspark_qwen3_4b_block7 draft
    model. Reference: measured over 12 runs of the full GSM8K set at
    temperature=1.0 (prefix caching disabled to avoid cross-run reuse):
      accuracy:        min=0.782 max=0.814 mean=0.801
      acceptance_rate: min=0.418 max=0.434 mean=0.428
      acceptance_len:  min=3.928 max=4.037 mean=3.994
    Thresholds set conservatively to 10% to avoid flaking due to unlucky sampling
    """
    spec_llm = LLM(**dspark_config)

    results = evaluate_gsm8k_offline(spec_llm, temperature=1.0)
    gsm8k_accuracy = results["accuracy"]

    metrics = spec_llm.get_metrics()
    acceptance_rate = compute_acceptance_rate(metrics)
    acceptance_len = compute_acceptance_len(metrics)

    print(
        f"DSpark acceptance_rate={acceptance_rate:.2f}, "
        f"acceptance_len={acceptance_len:.2f}, "
        f"gsm8k_accuracy={gsm8k_accuracy:.3f}"
    )

    assert acceptance_rate >= 0.428 * 0.9
    assert acceptance_len >= 3.994 * 0.9
    assert gsm8k_accuracy >= 0.801 * 0.9

    del spec_llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()
