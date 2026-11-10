# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from starlette.requests import Request

from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionResponse
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams


async def _never_finishes():
    await asyncio.Event().wait()
    yield


async def _finished(text: str):
    yield RequestOutput(
        request_id="",
        prompt=None,
        prompt_token_ids=[1],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[2],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        finished=True,
    )


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
    engine_inputs, expected_request_ids, monkeypatch
):
    engine_client = SimpleNamespace(
        errored=False,
        check_admission=Mock(),
        generate=Mock(side_effect=lambda *_args, **_kwargs: _never_finishes()),
        abort=AsyncMock(),
        is_tracing_enabled=AsyncMock(return_value=False),
    )

    server = SpeechToTextBaseServing.__new__(SpeechToTextBaseServing)
    server.engine_client = engine_client
    server.task_type = "transcribe"  # type: ignore[misc]  # Final, set in __init__
    server.models = SimpleNamespace(model_name=lambda: "audio")
    server.model_config = SimpleNamespace(max_model_len=1024)
    server.model_cls = SimpleNamespace(no_space_languages=set())
    server.default_sampling_params = {}
    server.asr_config = SimpleNamespace(max_audio_clip_s=30)
    monkeypatch.setattr(server, "_check_model", AsyncMock(return_value=None))
    monkeypatch.setattr(server, "_maybe_get_adapters", Mock(return_value=None))
    monkeypatch.setattr(
        server,
        "_preprocess_speech_to_text",
        AsyncMock(
            return_value=(
                engine_inputs,
                40.0,
                [30.0 * i for i in range(len(engine_inputs))],
            )
        ),
    )
    monkeypatch.setattr(server, "_log_inputs", Mock())

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
async def test_non_streaming_cancel_advances_all_chunk_generators(monkeypatch):
    started_request_ids: list[str] = []
    engine_client = SimpleNamespace(
        errored=False,
        check_admission=Mock(),
        generate=Mock(
            side_effect=lambda *_args, **_kwargs: _records_start_then_never_finishes(
                started_request_ids, _args[2]
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
    server = SpeechToTextBaseServing.__new__(SpeechToTextBaseServing)
    server.engine_client = engine_client
    server.task_type = "transcribe"  # type: ignore[misc]  # Final, set in __init__
    server.models = SimpleNamespace(model_name=lambda: "audio")
    server.model_config = SimpleNamespace(max_model_len=1024)
    server.model_cls = SimpleNamespace(no_space_languages=set())
    server.default_sampling_params = {}
    server.asr_config = SimpleNamespace(max_audio_clip_s=30)
    monkeypatch.setattr(server, "_check_model", AsyncMock(return_value=None))
    monkeypatch.setattr(server, "_maybe_get_adapters", Mock(return_value=None))
    monkeypatch.setattr(
        server,
        "_preprocess_speech_to_text",
        AsyncMock(return_value=(engine_inputs, 90.0, [0.0, 29.5, 29.5 + 29.7])),
    )
    monkeypatch.setattr(server, "_log_inputs", Mock())

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

    server = SpeechToTextBaseServing.__new__(SpeechToTextBaseServing)
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

    sampling_params = engine_client.generate.call_args.args[1]
    assert sampling_params.watermarking is False
    engine_client.abort.assert_awaited_once_with(request_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_beam_search", [False, True])
@pytest.mark.parametrize(
    "header,expected_rank", [(None, None), ("invalid", None), ("0", 0), ("3", 3)]
)
async def test_speech_to_text_preserves_data_parallel_rank(
    use_beam_search, header, expected_rank, monkeypatch
):
    """X-data-parallel-rank reaches language detection and every chunk request."""
    ranks = []

    def generate(prompt, params, request_id, *, data_parallel_rank=None, **kwargs):
        ranks.append((request_id, data_parallel_rank))
        return _finished("hello")

    def beam_search(*, request_id, data_parallel_rank=None, **kwargs):
        ranks.append((request_id, data_parallel_rank))
        return _finished("hello")

    server = SpeechToTextBaseServing.__new__(SpeechToTextBaseServing)
    server.engine_client = SimpleNamespace(
        errored=False, check_admission=Mock(), generate=generate
    )
    server.task_type = "transcribe"  # type: ignore[misc]  # Final, set in __init__
    server.models = SimpleNamespace(model_name=lambda: "audio")
    server.model_config = SimpleNamespace(max_model_len=1024)
    server.default_sampling_params = {}
    server.asr_config = SimpleNamespace(max_audio_clip_s=30, sample_rate=16000)
    server.tokenizer = Mock()
    server.model_cls = SimpleNamespace(
        validate_language=lambda language: language,
        supports_explicit_language_detection=True,
        get_language_detection_prompt=Mock(return_value={"prompt": "detect"}),
        get_language_token_ids=Mock(return_value=[1]),
        parse_language_detection_output=Mock(return_value="en"),
        get_generation_prompt=Mock(return_value={"prompt": "transcribe"}),
        post_process_output=lambda text, model_config: text,
        no_space_languages=set(),
    )
    server.renderer = SimpleNamespace(
        render_cmpl_async=AsyncMock(
            return_value=[{"prompt": "chunk-0"}, {"prompt": "chunk-1"}]
        )
    )
    server._decode_and_chunk_speech_async = AsyncMock(
        return_value=([np.zeros(16000), np.zeros(16000)], 2.0)
    )
    monkeypatch.setattr(server, "_check_model", AsyncMock(return_value=None))
    monkeypatch.setattr(server, "_maybe_get_adapters", Mock(return_value=None))
    monkeypatch.setattr(server, "_log_inputs", Mock())
    monkeypatch.setattr(server, "beam_search", beam_search)
    monkeypatch.setattr(
        "vllm.entrypoints.speech_to_text.base.serving.parse_model_prompt",
        lambda model_config, prompt: prompt,
    )

    request = SimpleNamespace(
        model="audio",
        response_format="json",
        stream=False,
        use_beam_search=use_beam_search,
        max_completion_tokens=None,
        language=None,
        to_language=None,
        build_stt_params=Mock(),
        to_sampling_params=Mock(return_value=SamplingParams()),
        to_beam_search_params=Mock(
            return_value=BeamSearchParams(beam_width=1, max_tokens=1)
        ),
    )
    headers = [(b"x-request-id", b"outer-request")]
    if header is not None:
        headers.append((b"x-data-parallel-rank", header.encode()))

    response = await server._create_speech_to_text(
        audio_data=b"audio",
        request=request,
        raw_request=Request({"type": "http", "headers": headers}),
        response_class=TranscriptionResponse,
        stream_generator_method=Mock(),
    )

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "hello hello"
    assert ranks == [
        ("transcribe-outer-request-lang_detect", expected_rank),
        ("transcribe-outer-request-0", expected_rank),
        ("transcribe-outer-request-1", expected_rank),
    ]
