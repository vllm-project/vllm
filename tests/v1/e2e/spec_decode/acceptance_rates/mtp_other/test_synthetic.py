# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from tests.utils import single_gpu_only
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

from ...utils import compute_acceptance_len, get_test_prompts


@single_gpu_only
def test_synthetic_acceptance_rate():
    """Verify that synthetic rejection sampling produces an acceptance
    length close to the requested mean acceptance length."""
    num_spec_tokens = 3
    expected_acceptance_len = 1.875
    tolerance = 0.15

    spec_llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        trust_remote_code=True,
        speculative_config={
            "method": "eagle3",
            "model": "nm-testing/Llama3_2_1B_speculator.eagle3",
            "num_speculative_tokens": num_spec_tokens,
            "max_model_len": 2048,
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_length": expected_acceptance_len,
        },
        max_model_len=2048,
        enforce_eager=True,
        disable_log_stats=False,
    )

    test_prompts = get_test_prompts(mm_enabled=False, num_prompts=50)
    spec_llm.chat(
        test_prompts,
        SamplingParams(temperature=0, max_tokens=64, ignore_eos=True),
    )

    metrics = spec_llm.get_metrics()
    acceptance_len = compute_acceptance_len(metrics)

    print(
        f"Synthetic acceptance length: {acceptance_len:.3f}"
        f" (expected={expected_acceptance_len:.3f},"
        f" tolerance=±{tolerance})"
    )
    assert abs(acceptance_len - expected_acceptance_len) <= tolerance, (
        f"Synthetic acceptance length {acceptance_len:.3f} is not within"
        f" ±{tolerance} of expected {expected_acceptance_len:.3f}"
    )

    del spec_llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()
