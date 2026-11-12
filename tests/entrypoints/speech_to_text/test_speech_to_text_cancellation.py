# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

import vllm.entrypoints.speech_to_text.base.serving as serving_module
import vllm.envs as envs
from vllm.config.speech_to_text import SpeechToTextConfig, VADConfig
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
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
    server.model_cls = SimpleNamespace(
        no_space_languages=set(), supports_diarized_transcription=False
    )
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
                40.0,
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
    server.model_cls = SimpleNamespace(
        no_space_languages=set(), supports_diarized_transcription=False
    )
    server.default_sampling_params = {}
    server.asr_config = SimpleNamespace(max_audio_clip_s=30)
    monkeypatch.setattr(server, "_check_model", AsyncMock(return_value=None))
    monkeypatch.setattr(server, "_maybe_get_adapters", Mock(return_value=None))
    monkeypatch.setattr(
        server,
        "_preprocess_speech_to_text",
        AsyncMock(return_value=(engine_inputs, 90.0, [0.0, 29.5, 29.5 + 29.7], 90.0)),
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
async def test_vad_without_silero_vad_returns_install_hint(monkeypatch):
    server = SpeechToTextBaseServing.__new__(SpeechToTextBaseServing)
    server.engine_client = SimpleNamespace(errored=False, check_admission=Mock())
    server.models = SimpleNamespace(model_name=lambda: "audio")
    server.model_cls = SimpleNamespace(
        supports_segment_timestamp=True, supports_diarized_transcription=False
    )
    monkeypatch.setattr(server, "_check_model", AsyncMock(return_value=None))
    monkeypatch.setattr(serving_module, "has_silero_vad", lambda: False)
    request = SimpleNamespace(
        model="audio",
        response_format="json",
        stream=False,
        use_beam_search=False,
        build_vad_config=lambda: VADConfig(enabled=True),
    )

    response = await server._create_speech_to_text(
        audio_data=b"audio",
        request=request,
        raw_request=None,
        response_class=TranscriptionResponse,
        stream_generator_method=Mock(),
    )

    assert isinstance(response, ErrorResponse)
    assert "pip install vllm[audio]" in response.error.message


def _make_blocking_vad_server(monkeypatch, decode_workers: int, vad_workers: int):
    """Build the real decode and VAD pools with a decode that records each
    call and a VAD that blocks until the returned event is set."""
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_PREPROCESS_WORKERS", decode_workers)
    monkeypatch.setattr(
        SpeechToTextBaseServing,
        "model_cls",
        SimpleNamespace(
            get_speech_to_text_config=lambda *_: SpeechToTextConfig(),
            get_streaming_post_processor_cls=lambda: None,
            supports_segment_timestamp=False,
            validate_language=lambda language: language,
        ),
    )

    decoded: list[bytes] = []
    vad_released = threading.Event()

    def fake_decode_audio(self, audio_data):
        decoded.append(audio_data)
        return np.zeros(16000, dtype=np.float32), 16000, 1.0

    def fake_chunk_speech(self, y, sr, duration, request):
        vad_released.wait()  # holds the VAD worker until released
        return []

    monkeypatch.setattr(SpeechToTextBaseServing, "_decode_audio", fake_decode_audio)
    monkeypatch.setattr(SpeechToTextBaseServing, "_chunk_speech", fake_chunk_speech)

    engine_client = SimpleNamespace(
        model_config=SimpleNamespace(get_diff_sampling_param=dict),
        renderer=None,
        input_processor=None,
    )
    server = SpeechToTextBaseServing(
        engine_client,
        models=None,
        request_logger=None,
        max_vad_workers=vad_workers,
    )
    return server, decoded, vad_released


def _start_vad_requests(server, num_requests: int) -> list[asyncio.Task]:
    request = SimpleNamespace(
        language="en",
        to_language=None,
        build_vad_config=lambda: VADConfig(enabled=True),
    )
    return [
        asyncio.create_task(
            server._preprocess_speech_to_text(request, b"audio", f"request-{i}")
        )
        for i in range(num_requests)
    ]


async def _wait_for_decodes(decoded: list[bytes], count: int) -> None:
    for _ in range(3000):  # up to 30 s on a slow machine
        if len(decoded) >= count:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)  # let any decodes beyond `count` happen


@pytest.mark.asyncio
async def test_vad_pool_bounds_decoded_audio_waiting_for_vad(monkeypatch):
    """With a dedicated VAD pool, at most decode + VAD workers requests may
    hold decoded audio while VAD is busy."""
    decode_workers, vad_workers, num_requests = 2, 1, 8
    bound = decode_workers + vad_workers
    server, decoded, vad_released = _make_blocking_vad_server(
        monkeypatch, decode_workers, vad_workers
    )
    tasks = _start_vad_requests(server, num_requests)
    try:
        await _wait_for_decodes(decoded, bound)
        assert len(decoded) == bound
    finally:
        vad_released.set()
        await asyncio.gather(*tasks)
        server.shutdown()
    assert len(decoded) == num_requests


@pytest.mark.asyncio
async def test_vad_pool_cancellation_releases_permits(monkeypatch):
    """Cancelling requests that are running or waiting for VAD returns
    promptly and frees their permits, so new requests can decode again."""
    decode_workers, vad_workers, num_requests = 2, 1, 8
    bound = decode_workers + vad_workers
    server, decoded, vad_released = _make_blocking_vad_server(
        monkeypatch, decode_workers, vad_workers
    )
    cancelled = _start_vad_requests(server, num_requests)
    new_tasks: list[asyncio.Task] = []
    try:
        await _wait_for_decodes(decoded, bound)
        # One request is running VAD, the rest wait for VAD or for a permit.
        for task in cancelled:
            task.cancel()
        results = await asyncio.wait_for(
            asyncio.gather(*cancelled, return_exceptions=True), timeout=5
        )
        assert all(isinstance(r, asyncio.CancelledError) for r in results)

        new_tasks = _start_vad_requests(server, num_requests)
        await _wait_for_decodes(decoded, 2 * bound)
        assert len(decoded) == 2 * bound
    finally:
        vad_released.set()
        await asyncio.wait_for(asyncio.gather(*new_tasks), timeout=30)
        server.shutdown()
    assert len(decoded) == bound + num_requests
