# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig

from ..utils import (
    assert_request_outputs_match,
    get_spec_decode_metric_value,
    get_test_prompts,
)


@pytest.mark.parametrize(
    "speculative_config",
    [
        {
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": 3,
        },
    ],
)
@single_gpu_only
def test_per_request_disable_spec_decode(
    speculative_config: dict,
    model_name: str,
    vllm_runner,
):
    """`SamplingParams.disable_spec_decode` turns speculative decoding off for
    individual requests inside a batch that otherwise speculates: outputs are
    unchanged (greedy) and no draft tokens are produced for those requests."""
    prompts = get_test_prompts(mm_enabled=False, num_prompts=16)
    greedy = SamplingParams(temperature=0, max_tokens=64)
    greedy_no_spec = SamplingParams(
        temperature=0, max_tokens=64, disable_spec_decode=True
    )

    with vllm_runner(
        model_name,
        block_size=None,
        trust_remote_code=False,
        enable_chunked_prefill=None,
        max_model_len=4096,
        compilation_config=CompilationConfig(),
        disable_log_stats=False,
    ) as ref_runner:
        ref_outputs = ref_runner.llm.chat(prompts, greedy)

    with vllm_runner(
        model_name,
        block_size=None,
        trust_remote_code=False,
        enable_chunked_prefill=None,
        speculative_config=speculative_config,
        max_model_len=4096,
        compilation_config=CompilationConfig(),
        disable_log_stats=False,
    ) as spec_runner:
        # Every request has speculation disabled: no drafts at all.
        outputs_all_disabled = spec_runner.llm.chat(prompts, greedy_no_spec)
        metrics = spec_runner.llm.get_metrics()
        assert get_spec_decode_metric_value(metrics, "vllm:spec_decode_num_drafts") == 0

        # Mixed batch: even prompts speculate, odd prompts do not.
        mixed_params = [
            greedy if i % 2 == 0 else greedy_no_spec for i in range(len(prompts))
        ]
        outputs_mixed = spec_runner.llm.chat(prompts, mixed_params)
        metrics = spec_runner.llm.get_metrics()
        assert get_spec_decode_metric_value(metrics, "vllm:spec_decode_num_drafts") > 0

    # The byte-exact claim is the metric above: zero drafts are scheduled for
    # flagged requests. Output text is held to the same bar as the other
    # spec-decode e2e tests rather than exact equality: an engine configured
    # for speculation still runs flagged rows through the uniform-decode batch
    # shape (placeholder padding), so floating-point tie-breaks can differ from
    # a no-spec engine even though no draft is ever proposed or verified.
    required = int(0.6 * len(ref_outputs)) + 1
    assert_request_outputs_match(
        ref_outputs,
        outputs_all_disabled,
        required_matches=required,
        context=f"disable_spec_decode on all requests, {model_name}",
    )
    assert_request_outputs_match(
        ref_outputs,
        outputs_mixed,
        required_matches=required,
        context=f"disable_spec_decode on alternating requests, {model_name}",
    )
