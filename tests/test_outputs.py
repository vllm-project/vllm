# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.outputs import CompletionOutput, RequestOutput, SamplingMask
from vllm.v1.metrics.stats import RequestSpecDecodeMetrics

pytestmark = pytest.mark.cpu_test


def test_request_output_forward_compatible():
    output = RequestOutput(
        request_id="test_request_id",
        prompt="test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=False,
        example_arg_added_in_new_version="some_value",
    )
    assert output is not None


def test_request_output_add_preserves_terminal_metadata():
    accumulated = RequestOutput(
        request_id="request",
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="first",
                token_ids=[1],
                cumulative_logprob=None,
                logprobs=None,
            )
        ],
        finished=False,
    )
    sampling_mask = SamplingMask([[10], [20]])
    spec_decode_metrics = RequestSpecDecodeMetrics.new(num_spec_tokens=2)
    spec_decode_metrics.observe(num_draft_tokens=2, num_accepted=1)
    terminal = RequestOutput(
        request_id="request",
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="second",
                token_ids=[2],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="length",
                sampling_mask=sampling_mask,
                spec_decode_metrics=spec_decode_metrics,
            )
        ],
        finished=True,
    )

    accumulated.add(terminal, aggregate=True)

    output = accumulated.outputs[0]
    assert output.token_ids == [1, 2]
    assert output.sampling_mask is sampling_mask
    assert output.spec_decode_metrics is spec_decode_metrics
