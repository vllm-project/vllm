# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from types import SimpleNamespace

import pytest
import torch

from vllm.entrypoints.pooling.base.serving import PoolingBaseServing
from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.exceptions import RetryableRequestError
from vllm.outputs import PoolingOutput, PoolingRequestOutput, RequestError

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.asyncio
async def test_collect_batch_propagates_retryable_request_error():
    failed = PoolingRequestOutput(
        request_id="request",
        outputs=PoolingOutput(torch.empty(0)),
        prompt_token_ids=[1, 2],
        num_cached_tokens=0,
        finished=True,
        error=RequestError(
            code="multimodal_cache_miss",
            message="Multi-modal processor cache miss.",
            retryable=True,
        ),
    )

    async def results():
        yield 0, failed

    ctx = SimpleNamespace(
        engine_inputs=[{"prompts": {"prompt_token_ids": [1, 2]}}],
        result_generator=results(),
        final_res_batch=None,
    )

    with pytest.raises(RetryableRequestError):
        await PoolingBaseServing._collect_batch(None, ctx)
    assert ctx.final_res_batch is None


def test_retryable_request_error_maps_to_service_unavailable():
    response = create_error_response(
        RetryableRequestError("Multi-modal processor cache miss.")
    )

    assert response.error.code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.error.type == "ServiceUnavailableError"
