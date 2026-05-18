# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger

from .protocol import TranslationRequest, TranslationResponseVariant
from .serving import OpenAIServingTranslation

logger = init_logger(__name__)

router = APIRouter()


def translation(request: Request) -> OpenAIServingTranslation:
    return request.app.state.openai_serving_translation


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
        raise NotImplementedError("The model does not support Translations API")

    audio_data = await request.file.read()

    generator = await handler.create_translation(audio_data, request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, TranslationResponseVariant):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")
