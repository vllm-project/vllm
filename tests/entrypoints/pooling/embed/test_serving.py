# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from fastapi.responses import JSONResponse

from vllm import PoolingParams
from vllm.entrypoints.pooling.embed.protocol import (
    CohereEmbedContent,
    CohereEmbedInput,
    CohereEmbedRequest,
)
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding
from vllm.entrypoints.pooling.typing import PoolingServeContext
from vllm.outputs import PoolingOutput, PoolingRequestOutput


def _make_pooling_output(num_prompt_tokens: int) -> PoolingRequestOutput:
    return PoolingRequestOutput(
        request_id="embd-test-0",
        outputs=PoolingOutput(data=torch.tensor([0.1, -0.2, 0.3, -0.4])),
        prompt_token_ids=list(range(num_prompt_tokens)),
        num_cached_tokens=0,
        finished=True,
    )


def _build_cohere_response(request: CohereEmbedRequest) -> dict:
    serving = object.__new__(ServingEmbedding)
    serving.json_response_cls = JSONResponse

    ctx = PoolingServeContext(
        request=request,
        model_name="test",
        request_id="embd-test",
        pooling_params=PoolingParams(),
        final_res_batch=[_make_pooling_output(num_prompt_tokens=3)],
    )

    response = serving._build_cohere_response_from_ctx(ctx)
    return json.loads(response.body)


@pytest.mark.parametrize(
    ("cohere_request", "expected_input_tokens", "expected_image_tokens"),
    [
        (
            CohereEmbedRequest(model="test", texts=["hello"]),
            3,
            0,
        ),
        (
            CohereEmbedRequest(model="test", images=["image-uri"]),
            0,
            3,
        ),
        (
            CohereEmbedRequest(
                model="test",
                inputs=[
                    CohereEmbedInput(
                        content=[CohereEmbedContent(type="text", text="hello")]
                    )
                ],
            ),
            3,
            0,
        ),
        (
            CohereEmbedRequest(
                model="test",
                inputs=[
                    CohereEmbedInput(
                        content=[
                            CohereEmbedContent(type="text", text="hello"),
                            CohereEmbedContent(
                                type="image_url",
                                image_url={"url": "image-uri"},
                            ),
                        ]
                    )
                ],
            ),
            0,
            3,
        ),
    ],
)
def test_cohere_billed_units_account_for_image_input_shapes(
    cohere_request: CohereEmbedRequest,
    expected_input_tokens: int,
    expected_image_tokens: int,
):
    response = _build_cohere_response(cohere_request)

    assert response["meta"]["billed_units"] == {
        "input_tokens": expected_input_tokens,
        "image_tokens": expected_image_tokens,
    }
