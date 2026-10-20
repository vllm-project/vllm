# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.logprobs import Logprob
from vllm.outputs import (
    ClassificationRequestOutput,
    CompletionOutput,
    EmbeddingRequestOutput,
    PoolingOutput,
    PoolingRequestOutput,
    RequestError,
    RequestOutput,
    ScoringRequestOutput,
)
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


@pytest.mark.parametrize(
    ("output_type", "empty_value"),
    [
        (EmbeddingRequestOutput, []),
        (ClassificationRequestOutput, []),
        (ScoringRequestOutput, 0.0),
    ],
)
@pytest.mark.skip_global_cleanup
def test_specialized_pooling_outputs_preserve_request_error(output_type, empty_value):
    error = RequestError(
        code="multimodal_cache_miss",
        message="Multi-modal processor cache miss.",
        retryable=True,
    )
    base_output = PoolingRequestOutput(
        request_id="request",
        outputs=PoolingOutput(torch.empty(0)),
        prompt_token_ids=[1, 2],
        num_cached_tokens=0,
        finished=True,
        error=error,
    )

    output = output_type.from_base(base_output)

    assert output.error is error
    value = getattr(output.outputs, "embedding", None)
    value = getattr(output.outputs, "probs", value)
    value = getattr(output.outputs, "score", value)
    assert value == empty_value


def test_request_output_add_preserves_terminal_spec_decode_metrics():
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
                spec_decode_metrics=spec_decode_metrics,
            )
        ],
        finished=True,
    )

    accumulated.add(terminal, aggregate=True)

    output = accumulated.outputs[0]
    assert output.token_ids == [1, 2]
    assert output.finish_reason == "length"
    assert accumulated.finished
    assert output.spec_decode_metrics is spec_decode_metrics


def test_request_output_add_keeps_prompt_logprobs_from_either_side():
    prompt_logprobs = [None, {2: Logprob(logprob=-1.0)}]
    token_id_logprobs = np.array([[0.0], [-1.0]], dtype=np.float32)

    def output(index, prompt_logprobs, token_id_logprobs):
        return RequestOutput(
            request_id="request",
            prompt=None,
            prompt_token_ids=[1, 2],
            prompt_logprobs=prompt_logprobs,
            prompt_token_id_logprobs=token_id_logprobs,
            outputs=[
                CompletionOutput(
                    index=index,
                    text="",
                    token_ids=[3],
                    cumulative_logprob=None,
                    logprobs=None,
                )
            ],
            finished=False,
        )

    # A DELTA output whose prompt logprobs were already sent has [] / None.
    merged = output(0, [], None)
    merged.add(output(1, prompt_logprobs, token_id_logprobs), aggregate=True)
    assert merged.prompt_logprobs is prompt_logprobs
    assert merged.prompt_token_id_logprobs is token_id_logprobs

    merged = output(0, prompt_logprobs, token_id_logprobs)
    merged.add(output(1, [], None), aggregate=True)
    assert merged.prompt_logprobs is prompt_logprobs
    assert merged.prompt_token_id_logprobs is token_id_logprobs
