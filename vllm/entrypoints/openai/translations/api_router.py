# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, FastAPI, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.translations.protocol import (
    TranscriptionRequest,
    TranscriptionResponseVariant,
    TranslationRequest,
    TranslationResponseVariant,
)
from vllm.entrypoints.openai.translations.serving import (
    OpenAIServingTranscription,
    OpenAIServingTranslation,
)
from vllm.entrypoints.utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger
    from vllm.tasks import SupportedTask
else:
    RequestLogger = Any

logger = init_logger(__name__)

router = APIRouter()


def transcription(request: Request) -> OpenAIServingTranscription:
    return request.app.state.openai_serving_transcription


def translation(request: Request) -> OpenAIServingTranslation:
    return request.app.state.openai_serving_translation


@router.post(
    "/v1/audio/transcriptions",
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.UNPROCESSABLE_ENTITY.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_transcriptions(
    raw_request: Request, request: Annotated[TranscriptionRequest, Form()]
):
    handler = transcription(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Transcriptions API"
        )

    audio_data = await request.file.read()
    try:
        generator = await handler.create_transcription(audio_data, request, raw_request)
    except Exception as e:
        return handler.create_error_response(e)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, TranscriptionResponseVariant):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/audio/translations",
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.UNPROCESSABLE_ENTITY.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_translations(
    request: Annotated[TranslationRequest, Form()], raw_request: Request
):
    handler = translation(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Translations API"
        )

    audio_data = await request.file.read()
    try:
        generator = await handler.create_translation(audio_data, request, raw_request)
    except Exception as e:
        return handler.create_error_response(e)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, TranslationResponseVariant):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


def attach_router(app: FastAPI):
    app.include_router(router)


def init_transcription_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
):
    state.openai_serving_transcription = (
        OpenAIServingTranscription(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            log_error_stack=args.log_error_stack,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "transcription" in supported_tasks
        else None
    )
    state.openai_serving_translation = (
        OpenAIServingTranslation(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            log_error_stack=args.log_error_stack,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "transcription" in supported_tasks
        else None
    )
