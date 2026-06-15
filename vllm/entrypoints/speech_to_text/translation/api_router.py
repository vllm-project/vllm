# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Annotated

from fastapi import Form, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.entrypoints.speech_to_text.base.utils import read_upload_with_limit
from vllm.logger import init_logger

from .protocol import TranslationRequest, TranslationResponseVariant
from .serving import OpenAIServingTranslation

logger = init_logger(__name__)


def translation(request: Request) -> OpenAIServingTranslation:
    return request.app.state.openai_serving_translation


@with_cancellation
@load_aware_call
async def create_translations(
    request: Annotated[TranslationRequest, Form()], raw_request: Request
):
    handler = translation(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Translations API")

    audio_data = await read_upload_with_limit(request.file)

    generator = await handler.create_translation(audio_data, request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, TranslationResponseVariant):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")
