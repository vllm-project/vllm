# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.decisions.state import init_decisions_state
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.renderers.chat_utils import load_chat_template
from vllm.tasks import SupportedTask

from .protocol import StructuredDecisionRequest
from .serving import ServingStructuredDecisions

router = APIRouter()


def structured_decisions(request: Request) -> ServingStructuredDecisions | None:
    return getattr(request.app.state, "serving_structured_decisions", None)


@router.post(
    "/v1/systemone",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_structured_decision(
    request: StructuredDecisionRequest, raw_request: Request
):
    handler = structured_decisions(raw_request)
    if handler is None:
        reason = getattr(
            raw_request.app.state,
            "structured_decisions_disabled_reason",
            "the model does not support it",
        )
        raise NotImplementedError(f"/v1/systemone is disabled: {reason}")
    result = await handler.create_decision(request, raw_request)
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)
    return JSONResponse(content=result.model_dump())


def register_structured_decisions_api_router(app: FastAPI):
    app.include_router(router)


def init_structured_decisions_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    request_logger: RequestLogger | None,
    supported_tasks: tuple[SupportedTask, ...],
    default_chat_template_kwargs: dict[str, Any],
):
    strategy = init_decisions_state(
        engine_client,
        state,
        args,
        request_logger,
        load_chat_template(args.chat_template),
        default_chat_template_kwargs,
        supported_tasks,
    )
    state.serving_structured_decisions = (
        ServingStructuredDecisions(
            state.openai_serving_models, strategy, request_logger=request_logger
        )
        if strategy is not None
        else None
    )
