# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm.entrypoints.openai.speech_to_text.protocol import TranscriptionResponse
from vllm.entrypoints.openai.speech_to_text.speech_to_text import OpenAISpeechToText


async def _never_finishes():
    await asyncio.Event().wait()
    yield


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine_inputs", "expected_request_ids"),
    [
        ([{"prompt": "chunk"}], ["transcribe-outer-request"]),
        (
            [{"prompt": "chunk-0"}, {"prompt": "chunk-1"}],
            ["transcribe-outer-request-0", "transcribe-outer-request-1"],
        ),
    ],
)
async def test_non_streaming_cancel_aborts_engine_requests(
    engine_inputs, expected_request_ids
):
    engine_client = SimpleNamespace(
        errored=False,
        generate=Mock(side_effect=lambda *_args, **_kwargs: _never_finishes()),
        abort=AsyncMock(),
        is_tracing_enabled=AsyncMock(return_value=False),
    )

    server = OpenAISpeechToText.__new__(OpenAISpeechToText)
    server.engine_client = engine_client
    server.task_type = "transcribe"
    server.models = SimpleNamespace(model_name=lambda: "audio")
    server.model_config = SimpleNamespace(max_model_len=1024)
    server.default_sampling_params = {}
    server.asr_config = SimpleNamespace(max_audio_clip_s=30)
    server._check_model = AsyncMock(return_value=None)
    server._maybe_get_adapters = Mock(return_value=None)
    server._preprocess_speech_to_text = AsyncMock(return_value=(engine_inputs, 40.0))
    server._log_inputs = Mock()

    request = SimpleNamespace(
        model="audio",
        response_format="json",
        stream=False,
        use_beam_search=False,
        max_completion_tokens=None,
        prompt="",
        to_sampling_params=Mock(return_value=object()),
    )
    raw_request = SimpleNamespace(
        headers={"X-Request-Id": "outer-request"},
        state=SimpleNamespace(),
    )

    task = asyncio.create_task(
        server._create_speech_to_text(
            audio_data=b"audio",
            request=request,
            raw_request=raw_request,
            response_class=TranscriptionResponse,
            stream_generator_method=Mock(),
        )
    )
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    generated_request_ids = [
        call.args[2] for call in engine_client.generate.call_args_list
    ]
    assert generated_request_ids == expected_request_ids
    for request_id in expected_request_ids:
        engine_client.abort.assert_any_await(request_id)
    assert engine_client.abort.await_count == len(expected_request_ids)
