# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm.entrypoints.speech_to_text.base.serving import OpenAISpeechToText
from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionResponse


async def _never_finishes():
    await asyncio.Event().wait()
    yield


async def _records_start_then_never_finishes(started_request_ids, request_id):
    started_request_ids.append(request_id)
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
    server.model_cls = SimpleNamespace(no_space_languages=set())
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
        language="en",
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
    engine_client.abort.assert_awaited_once_with(expected_request_ids)


@pytest.mark.asyncio
async def test_non_streaming_cancel_advances_all_chunk_generators():
    started_request_ids: list[str] = []
    engine_client = SimpleNamespace(
        errored=False,
        generate=Mock(
            side_effect=lambda *_args, **_kwargs: (
                _records_start_then_never_finishes(started_request_ids, _args[2])
            )
        ),
        abort=AsyncMock(),
        is_tracing_enabled=AsyncMock(return_value=False),
    )

    engine_inputs = [
        {"prompt": "chunk-0"},
        {"prompt": "chunk-1"},
        {"prompt": "chunk-2"},
    ]
    server = OpenAISpeechToText.__new__(OpenAISpeechToText)
    server.engine_client = engine_client
    server.task_type = "transcribe"
    server.models = SimpleNamespace(model_name=lambda: "audio")
    server.model_config = SimpleNamespace(max_model_len=1024)
    server.model_cls = SimpleNamespace(no_space_languages=set())
    server.default_sampling_params = {}
    server.asr_config = SimpleNamespace(max_audio_clip_s=30)
    server._check_model = AsyncMock(return_value=None)
    server._maybe_get_adapters = Mock(return_value=None)
    server._preprocess_speech_to_text = AsyncMock(return_value=(engine_inputs, 90.0))
    server._log_inputs = Mock()

    request = SimpleNamespace(
        model="audio",
        response_format="json",
        stream=False,
        use_beam_search=False,
        max_completion_tokens=None,
        language="en",
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
    await asyncio.sleep(0.01)

    expected_request_ids = [
        "transcribe-outer-request-0",
        "transcribe-outer-request-1",
        "transcribe-outer-request-2",
    ]
    assert set(started_request_ids) == set(expected_request_ids)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_language_detection_cancel_aborts_engine_request():
    engine_client = SimpleNamespace(
        generate=Mock(return_value=_never_finishes()),
        abort=AsyncMock(),
    )

    server = OpenAISpeechToText.__new__(OpenAISpeechToText)
    server.engine_client = engine_client
    server.asr_config = SimpleNamespace()
    server.tokenizer = Mock()
    server.model_cls = SimpleNamespace(
        get_language_detection_prompt=Mock(return_value={"prompt": "detect"}),
        get_language_token_ids=Mock(return_value=[1]),
        parse_language_detection_output=Mock(),
    )

    request_id = "transcribe-outer-request-lang_detect"
    task = asyncio.create_task(server._detect_language(Mock(), request_id))
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    engine_client.abort.assert_awaited_once_with(request_id)
