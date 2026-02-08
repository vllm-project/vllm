# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, Depends, Request
from starlette.responses import Response

from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.pooling.classify.protocol import ClassificationRequest
from vllm.entrypoints.utils import load_aware_call, with_cancellation

router = APIRouter()


@router.post("/classify", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_classify(
    request: ClassificationRequest, raw_request: Request
) -> Response:
    handler = raw_request.app.state.openai_serving_classification
    return await handler(request, raw_request)
