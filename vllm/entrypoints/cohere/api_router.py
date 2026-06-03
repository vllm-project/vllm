# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.cohere.protocol import (
    CohereChatV2Request,
    CohereChatV2Response,
    CohereError,
)
from vllm.entrypoints.cohere.serving import CohereServingChatV2
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import load_aware_call, with_cancellation
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def _serving(request: Request) -> CohereServingChatV2 | None:
    return getattr(request.app.state, "cohere_serving_chat_v2", None)


def _error_response(
    error: ErrorResponse, *, fallback_status: int = HTTPStatus.BAD_REQUEST
) -> JSONResponse:
    """Translate vLLM's internal error envelope into Cohere's error shape."""
    info = error.error
    status = info.code or fallback_status
    return JSONResponse(
        status_code=status,
        content=CohereError(message=info.message).model_dump(exclude_none=True),
    )


@router.post(
    "/v2/chat",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": CohereError},
        HTTPStatus.NOT_FOUND.value: {"model": CohereError},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": CohereError},
    },
)
@with_cancellation
@load_aware_call
async def chat_v2(request: CohereChatV2Request, raw_request: Request):
    handler = _serving(raw_request)
    if handler is None:
        return JSONResponse(
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
            content=CohereError(
                message="The model does not support the Cohere v2 chat API."
            ).model_dump(exclude_none=True),
        )

    try:
        result = await handler.create_chat_v2(request, raw_request)
    except Exception as e:  # noqa: BLE001 - report as 500 for parity w/ siblings
        logger.exception("Error in /v2/chat: %s", e)
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            content=CohereError(message=str(e)).model_dump(exclude_none=True),
        )

    if isinstance(result, ErrorResponse):
        return _error_response(result)

    if isinstance(result, CohereChatV2Response):
        return JSONResponse(content=result.model_dump(exclude_none=True))

    return StreamingResponse(content=result, media_type="text/event-stream")


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
