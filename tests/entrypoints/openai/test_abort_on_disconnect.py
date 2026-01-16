# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _cancelled_async_iter():
    async def gen():
        raise asyncio.CancelledError()
        if False:
            yield None

    return gen()


@pytest.mark.asyncio
async def test_completion_stream_generator_aborts_on_disconnect():
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion

    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    serving.engine_client = MagicMock()
    serving.engine_client.abort = AsyncMock()

    request = SimpleNamespace(n=None, stream_options=None, echo=False, max_tokens=1)
    request_prompts = ["hi"]  # won't be used (we cancel immediately)

    gen = serving.completion_stream_generator(
        request=request,
        request_prompts=request_prompts,
        result_generator=_cancelled_async_iter(),
        request_id="cmpl-123",
        created_time=0,
        model_name="m",
        num_prompts=1,
        tokenizer=MagicMock(),
        request_metadata=MagicMock(),
        enable_force_include_usage=False,
    )

    async for _ in gen:
        pass

    serving.engine_client.abort.assert_called_once_with("cmpl-123")


@pytest.mark.asyncio
async def test_chat_completion_stream_generator_aborts_on_disconnect():
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    serving = OpenAIServingChat.__new__(OpenAIServingChat)
    serving.engine_client = MagicMock()
    serving.engine_client.abort = AsyncMock()

    # minimal request fields used before the async-for
    request = SimpleNamespace(
        n=None,
        stream_options=None,
    )

    gen = serving.chat_completion_stream_generator(
        request=request,
        result_generator=_cancelled_async_iter(),
        request_id="chat-123",
        created_time=0,
        model_name="m",
        tokenizer=MagicMock(),
        request_metadata=MagicMock(),
    )

    async for _ in gen:
        pass

    serving.engine_client.abort.assert_called_once_with("chat-123")


@pytest.mark.asyncio
async def test_responses_stream_generator_aborts_on_disconnect():
    from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

    serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
    serving.engine_client = MagicMock()
    serving.engine_client.abort = AsyncMock()

    request = SimpleNamespace(request_id="resp-123")

    gen = serving.responses_stream_generator(
        request=request,
        sampling_params=MagicMock(),
        result_generator=_cancelled_async_iter(),
        context=MagicMock(),
        model_name="m",
        tokenizer=MagicMock(),
        request_metadata=MagicMock(),
        created_time=0,
    )

    async for _ in gen:
        pass

    serving.engine_client.abort.assert_called_once_with("resp-123")


@pytest.mark.asyncio
async def test_speech_to_text_stream_generator_aborts_on_disconnect():
    from vllm.entrypoints.openai.speech_to_text import OpenAISpeechToText

    serving = OpenAISpeechToText.__new__(OpenAISpeechToText)
    serving.engine_client = MagicMock()
    serving.engine_client.abort = AsyncMock()
    serving.task_type = "transcribe"

    request = SimpleNamespace(model="m")

    gen = serving._speech_to_text_stream_generator(
        request=request,
        list_result_generator=[_cancelled_async_iter()],
        request_id="transcribe-123",
        request_metadata=MagicMock(),
        audio_duration_s=1.0,
        chunk_object_type="transcription.chunk",
        response_stream_choice_class=MagicMock(),
        stream_response_class=MagicMock(),
    )

    async for _ in gen:
        pass

    serving.engine_client.abort.assert_called_once_with("transcribe-123")
