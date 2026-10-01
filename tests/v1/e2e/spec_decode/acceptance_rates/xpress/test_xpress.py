# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest
from vllm import SamplingParams
from vllm.config import CompilationConfig

from ...draft_model.test_draft_model import ArgsTest
from ...utils import (
    Messages,
    compute_acceptance_len,
    compute_acceptance_rate,
    evaluate_llm_for_gsm8k,
    get_test_prompts,
)


@dataclass
class XPressArgsTest(ArgsTest):
    """ArgsTest with the two knobs an attached refiner needs.

    XPress sizes a [max_num_seqs * block, vocab] scratch buffer, so it wants a
    smaller batch than the 100 the draft-model path captures with.
    """

    max_num_seqs: int = 16
    enable_prefix_caching: bool | None = None


# Reference values at temperature 0 on one H100. 
QWEN3_XPRESS = XPressArgsTest(
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
)


@pytest.mark.parametrize("args", [QWEN3_XPRESS], ids=["qwen3_xpress"])
def test_xpress_correctness_and_acceptance_rate(
    args: XPressArgsTest, vllm_runner
):
    """Guard GSM8K accuracy and acceptance metrics for XPress models.

    Accuracy is the check a refiner producing garbage cannot pass: NaN logits
    argmax to a fixed token, which the verifier then accepts every time, so
    acceptance alone would read as a perfect score.

    This mirrors assert_draft_model_correctness, but cannot call it: that helper
    hardcodes method="draft_model" together with the nested draft-engine config,
    and asserts async scheduling, which an attached refiner does not use.
    """
    test_prompts: list[Messages] = get_test_prompts(
        mm_enabled=False, num_prompts=args.num_prompts
    )

    with vllm_runner(
        args.target_model,
        block_size=None,
        trust_remote_code=False,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
        speculative_config={
            "model": args.draft_model,
            "method": "dflash",
            "num_speculative_tokens": args.num_speculative_tokens,
        },
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
        disable_log_stats=False,  # enables get_metrics()
    ) as spec_runner:
        spec_llm = spec_runner.llm

        # Qwen3's template enables a thinking block by default. Thinking text is
        # high-entropy free reasoning and accepts far worse than a direct answer,
        # which more than halves the acceptance length and leaves the thresholds
        # too close to what a degraded refiner still reaches.
        spec_llm.chat(
            test_prompts,
            args.sampling_config,
            chat_template_kwargs={"enable_thinking": False},
        )
        metrics = spec_llm.get_metrics()
        acceptance_rate: float = compute_acceptance_rate(metrics)
        acceptance_len: float = compute_acceptance_len(metrics)

        # Evaluate after reading the metrics, to not pollute them.
        evaluate_llm_for_gsm8k(
            spec_llm, expected_accuracy_threshold=args.expected_gsm8k_accuracy
        )

        print(
            f"xpress: target={args.target_model}, draft={args.draft_model}, "
            f"acceptance_rate={acceptance_rate:.3f}, "
            f"acceptance_len={acceptance_len:.3f}"
        )

        context = (
            f"xpress target={args.target_model}, draft={args.draft_model}, "
            f"acceptance_rate={acceptance_rate:.3f}, "
            f"acceptance_len={acceptance_len:.3f}"
        )
        assert acceptance_rate >= args.expected_acceptance_rate, (
            f"{context}; expected acceptance_rate >= "
            f"{args.expected_acceptance_rate:.3f}"
        )
        assert acceptance_len >= args.expected_acceptance_len, (
            f"{context}; expected acceptance_len >= "
            f"{args.expected_acceptance_len:.3f}"
        )
        # Accepting every draft on every step is a verifier rubber-stamping
        # degenerate output, not a good drafter.
        assert acceptance_len < args.num_speculative_tokens + 1, (
            f"{context}; acceptance_len is at its ceiling"
        )
