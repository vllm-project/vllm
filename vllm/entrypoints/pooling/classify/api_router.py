# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import Request
from fastapi.responses import Response

from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    with_cancellation,
)

from .protocol import ClassificationRequest
from .serving import ServingClassification


def classify(request: Request) -> ServingClassification | None:
    return request.app.state.serving_classification


@with_cancellation
@load_aware_call
async def create_classify(
    request: ClassificationRequest, raw_request: Request
) -> Response:
    handler = classify(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Classification API")

    return await handler(request, raw_request)
