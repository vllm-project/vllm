# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams

from ...draft_model.test_draft_model import ArgsTest, assert_draft_model_correctness

# Reference values at temperature 0 on one H100.
QWEN3_XPRESS = ArgsTest(
    target_model="Qwen/Qwen3-8B",
    draft_model="UIUC-SSAIL/Qwen3-8B-XPress-b16",
    sampling_config=SamplingParams(temperature=0.0, max_tokens=256),
    num_speculative_tokens=15,
    expected_acceptance_rate=0.387,
    expected_acceptance_len=6.71,
    expected_gsm8k_accuracy=0.80,
    enforce_eager=False,
    max_model_len=4096,
    gpu_memory_utilization=0.92,
    method="dflash",
    # XPress sizes a [max_num_seqs * block, vocab] scratch buffer, so it wants a
    # smaller batch than the 100 the draft-model path captures with.
    max_num_seqs=16,
    # Qwen3's template enables a thinking block by default. Thinking text is
    # high-entropy free reasoning and accepts far worse than a direct answer,
    # which more than halves the acceptance length and leaves the thresholds
    # too close to what a degraded refiner still reaches.
    chat_template_kwargs={"enable_thinking": False},
    # Accepting every draft on every step is a verifier rubber-stamping
    # degenerate output, not a good drafter.
    reject_full_acceptance=True,
)


@pytest.mark.parametrize("args", [QWEN3_XPRESS], ids=["qwen3_xpress"])
def test_xpress_correctness_and_acceptance_rate(args: ArgsTest, vllm_runner):
    """Guard GSM8K accuracy and acceptance metrics for XPress models."""
    assert_draft_model_correctness(args, vllm_runner)
