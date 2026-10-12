# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.logger import init_logger
from vllm.renderers.chat_utils import load_chat_template
from vllm.tasks import SupportedTask

from .protocol import StructuredDecisionRequest
from .serving import ServingStructuredDecisions
from .strategies import ReadContext, select_read_strategy

logger = init_logger(__name__)

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
        raise NotImplementedError(
            "The model does not support the structured decision API"
        )
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
    state.serving_structured_decisions = None
    try:
        strategy_cls = select_read_strategy(engine_client.model_config, supported_tasks)
        strategy = strategy_cls(
            ReadContext(
                engine_client=engine_client,
                online_renderer=state.online_renderer,
                chat_template=load_chat_template(args.chat_template),
                chat_template_content_format=args.chat_template_content_format,
                default_chat_template_kwargs=default_chat_template_kwargs,
            )
        )
    except ValueError as e:
        logger.debug("/v1/systemone is disabled: %s", e)
        return
    state.serving_structured_decisions = ServingStructuredDecisions(
        state.openai_serving_models, strategy, request_logger=request_logger
    )
