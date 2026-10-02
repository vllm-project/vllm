# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.outputs import (
    ClassificationRequestOutput,
    EmbeddingRequestOutput,
    PoolingOutput,
    PoolingRequestOutput,
    RequestError,
    RequestOutput,
    ScoringRequestOutput,
)

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
