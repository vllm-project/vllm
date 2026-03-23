# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.generative_scores.serving import (
    GenerativeScoreResponse,
    OpenAIServingGenerativeScores,
)
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import load_aware_call, with_cancellation
from vllm.logger import init_logger

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger

router = APIRouter()

logger = init_logger(__name__)


def generative_scores(request: Request) -> OpenAIServingGenerativeScores | None:
    return request.app.state.serving_generative_scores


@router.post(
    "/generative_score",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_generative_score(raw_request: Request):
    handler = generative_scores(raw_request)
    if handler is None:
        raise NotImplementedError(
            "The model does not support the Generative Scores API"
        )

    raw_body = await raw_request.json()

    from vllm.entrypoints.openai.generative_scores.serving import (
        GenerativeScoreRequest,
    )

    gen_request = GenerativeScoreRequest(**raw_body)
    result = await handler.create_generative_score(gen_request, raw_request)

    if isinstance(result, ErrorResponse):
        return JSONResponse(
            content=result.model_dump(), status_code=result.error.code
        )
    elif isinstance(result, GenerativeScoreResponse):
        return JSONResponse(content=result.model_dump())

    raise ValueError(f"Unexpected response type: {type(result)}")


def register_generative_scores_api_router(app: FastAPI):
    app.include_router(router)


async def init_generative_scores_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: "RequestLogger | None",
):
    from vllm.entrypoints.openai.generative_scores.serving import (
        OpenAIServingGenerativeScores,
    )

    state.serving_generative_scores = OpenAIServingGenerativeScores(
        engine_client,
        state.openai_serving_models,
        request_logger=request_logger,
    )
