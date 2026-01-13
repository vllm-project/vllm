# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from typing_extensions import assert_never

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.pooling.score.protocol import (
    RerankRequest,
    RerankResponse,
    ScoreRequest,
    ScoreResponse,
)
from vllm.entrypoints.pooling.score.serving import ServingScores
from vllm.entrypoints.utils import load_aware_call, with_cancellation
from vllm.logger import init_logger

router = APIRouter()

logger = init_logger(__name__)


def score(request: Request) -> ServingScores | None:
    return request.app.state.openai_serving_scores


def rerank(request: Request) -> ServingScores | None:
    return request.app.state.openai_serving_scores


@router.post(
    "/score",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_score(request: ScoreRequest, raw_request: Request):
    handler = score(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Score API"
        )

    try:
        generator = await handler.create_score(request, raw_request)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, ScoreResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post(
    "/v1/score",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_score_v1(request: ScoreRequest, raw_request: Request):
    logger.warning(
        "To indicate that Score API is not part of standard OpenAI API, we "
        "have moved it to `/score`. Please update your client accordingly."
    )

    return await create_score(request, raw_request)


@router.post(
    "/rerank",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def do_rerank(request: RerankRequest, raw_request: Request):
    handler = rerank(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Rerank (Score) API"
        )
    try:
        generator = await handler.do_rerank(request, raw_request)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, RerankResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post(
    "/v1/rerank",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def do_rerank_v1(request: RerankRequest, raw_request: Request):
    logger.warning_once(
        "To indicate that the rerank API is not part of the standard OpenAI"
        " API, we have located it at `/rerank`. Please update your client "
        "accordingly. (Note: Conforms to JinaAI rerank API)"
    )

    return await do_rerank(request, raw_request)


@router.post(
    "/v2/rerank",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def do_rerank_v2(request: RerankRequest, raw_request: Request):
    return await do_rerank(request, raw_request)
