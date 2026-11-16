# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import Discriminator, Tag

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    create_json_response,
    validate_json_request,
)

from .hash_protocol import HashResponse, HashTokensRequest
from .hash_serving import hash_request

router = APIRouter()


def _request_kind(value: Any) -> str | None:
    if isinstance(value, ChatCompletionRequest):
        return "chat"
    if isinstance(value, CompletionRequest):
        return "completion"
    if isinstance(value, HashTokensRequest):
        return "tokens"
    if isinstance(value, dict):
        kinds = {"messages": "chat", "prompt": "completion", "token_ids": "tokens"}
        present = [kind for key, kind in kinds.items() if key in value]
        if len(present) == 1:
            return present[0]
    return None


HashRequest = Annotated[
    Annotated[ChatCompletionRequest, Tag("chat")]
    | Annotated[CompletionRequest, Tag("completion")]
    | Annotated[HashTokensRequest, Tag("tokens")],
    Discriminator(_request_kind),
]


@router.post(
    "/hash",
    dependencies=[Depends(validate_json_request)],
    response_model=HashResponse,
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
async def create_hash(request: HashRequest, raw_request: Request):
    state = raw_request.app.state
    engine = getattr(state, "engine_client", None)
    renderer = getattr(state, "serving_render", None)
    if engine is None or renderer is None:
        raise NotImplementedError("The hash endpoint requires a running engine")
    return create_json_response(
        await hash_request(request, raw_request, renderer, engine)
    )
