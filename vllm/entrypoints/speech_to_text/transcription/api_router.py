# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import AsyncGenerator
from http import HTTPStatus
from typing import Annotated

from anyio import CancelScope
from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.types import Receive, Scope, Send

from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.entrypoints.speech_to_text.base.utils import read_upload_with_limit
from vllm.logger import init_logger

from ...serve.engine.protocol import ErrorResponse
from .protocol import TranscriptionRequest, TranscriptionResponseVariant
from .serving import OpenAIServingTranscription

logger = init_logger(__name__)

router = APIRouter()


class _DiarizedStreamingResponse(StreamingResponse):
    def __init__(self, stream: AsyncGenerator[str, None]) -> None:
        super().__init__(content=stream, media_type="text/event-stream")
        self._stream = stream

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # A disconnect during send can leave the generator suspended at yield.
            with CancelScope(shield=True):
                await self._stream.aclose()


def transcription(request: Request) -> OpenAIServingTranscription:
    return request.app.state.openai_serving_transcription


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
        raise NotImplementedError("The model does not support Transcriptions API")

    audio_data = await read_upload_with_limit(request.file)

    generator = await handler.create_transcription(audio_data, request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, str):
        return PlainTextResponse(content=generator)

    elif isinstance(generator, TranscriptionResponseVariant):
        return JSONResponse(content=generator.model_dump())

    if request.response_format == "diarized_json":
        return _DiarizedStreamingResponse(generator)
    return StreamingResponse(content=generator, media_type="text/event-stream")
