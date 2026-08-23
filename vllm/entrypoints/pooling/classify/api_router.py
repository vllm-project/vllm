# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)

from .protocol import ClassificationRequest
from .serving import ServingClassification

router = APIRouter()


def classify(request: Request) -> ServingClassification:
    handler = getattr(request.app.state, "serving_classification", None)
    if handler is None:
        raise NotImplementedError("The model does not support Classification API")
    return handler


@router.post("/classify", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_classify(
    request: ClassificationRequest, raw_request: Request
) -> Response:
    handler = classify(raw_request)
    return await handler(request, raw_request)
