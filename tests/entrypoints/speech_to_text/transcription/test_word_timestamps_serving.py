# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Word timestamps in /v1/audio/transcriptions, with a stand-in engine and model."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionResponseVerbose,
)
from vllm.model_executor.models.interfaces import WordTimestamp
from vllm.outputs import CompletionOutput, RequestOutput


def _finished(request_id: str, text: str) -> RequestOutput:
    completion = CompletionOutput(0, text, [1], None, None)
    return RequestOutput(request_id, None, [0], None, [completion], True)


async def _outputs(text: str, request_id: str):
    yield _finished(request_id, text)


def _server(
    monkeypatch,
    aligner,
    *,
    encoder_outputs: bool,
    num_chunks: int = 1,
):
    engine_client = SimpleNamespace(
        errored=False,
        check_admission=Mock(),
        generate=Mock(
            side_effect=lambda _input, _params, rid, **_: _outputs("hi", rid)
        ),
        abort=AsyncMock(),
        is_tracing_enabled=AsyncMock(return_value=False),
    )
    server = SpeechToTextBaseServing.__new__(SpeechToTextBaseServing)
    server.engine_client = engine_client
    server.task_type = "transcribe"  # type: ignore[misc]  # Final, set in __init__
    server.models = SimpleNamespace(model_name=lambda: "audio")
    server.model_config = SimpleNamespace(max_model_len=1024)
    server.model_cls = SimpleNamespace(
        no_space_languages=set(),
        supports_segment_timestamp=False,
        supports_diarized_transcription=False,
        supports_encoder_outputs=encoder_outputs,
        post_process_output=lambda text, model_config: text,
    )
    server.default_sampling_params = {}
    server.asr_config = SimpleNamespace(max_audio_clip_s=30)
    server.word_timestamp_aligner = aligner
    server._preprocess_executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(server, "_check_model", AsyncMock(return_value=None))
    monkeypatch.setattr(server, "_maybe_get_adapters", Mock(return_value=None))
    monkeypatch.setattr(server, "_log_inputs", Mock())
    monkeypatch.setattr(
        server,
        "_preprocess_speech_to_text",
        AsyncMock(
            return_value=(
                [{"prompt": f"chunk-{i}"} for i in range(num_chunks)],
                30.0 * num_chunks,
                [30.0 * i for i in range(num_chunks)],
            )
        ),
    )
    return server


def _request(**overrides):
    fields = dict(
        model="audio",
        response_format="verbose_json",
        timestamp_granularities=["word"],
        stream=False,
        use_beam_search=False,
        max_completion_tokens=None,
        language="en",
        prompt="",
        to_sampling_params=Mock(return_value=object()),
    )
    return SimpleNamespace(**(fields | overrides))


async def _transcribe(server, request):
    return await server._create_speech_to_text(
        audio_data=b"audio",
        request=request,
        raw_request=SimpleNamespace(
            headers={"X-Request-Id": "outer"}, state=SimpleNamespace()
        ),
        response_class=TranscriptionResponseVerbose,
        stream_generator_method=Mock(),
    )


@pytest.mark.asyncio
async def test_words_are_shifted_by_the_chunk_start(monkeypatch):
    aligner = Mock(align=Mock(return_value=[WordTimestamp("hi", 0.5, 0.75)]))
    server = _server(monkeypatch, aligner, encoder_outputs=True, num_chunks=2)

    response = await _transcribe(server, _request())

    assert isinstance(response, TranscriptionResponseVerbose)
    assert response.words is not None
    assert [(w.word, w.start, w.end) for w in response.words] == [
        ("hi", 0.5, 0.75),
        ("hi", 30.5, 30.75),
    ]
    # Each chunk reaches the aligner with its own output and text.
    chunks = sorted(
        (call.args[0] for call in aligner.align.call_args_list),
        key=lambda chunk: chunk.output.request_id,
    )
    assert [chunk.output.request_id for chunk in chunks] == [
        "transcribe-outer-0",
        "transcribe-outer-1",
    ]
    assert {(chunk.text, chunk.language) for chunk in chunks} == {("hi", "en")}


@pytest.mark.asyncio
async def test_a_failed_alignment_fails_the_request(monkeypatch):
    aligner = Mock(align=Mock(side_effect=RuntimeError("no capture")))
    server = _server(monkeypatch, aligner, encoder_outputs=True, num_chunks=2)

    response = await _transcribe(server, _request())

    assert isinstance(response, ErrorResponse)
    assert response.error.code == 500
    assert "Word timestamps failed: no capture" in response.error.message
    server.engine_client.abort.assert_awaited_once_with(
        ["transcribe-outer-0", "transcribe-outer-1"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("encoder_outputs", [True, False])
async def test_beam_search_is_refused_only_for_encoder_outputs(
    monkeypatch, encoder_outputs
):
    server = _server(monkeypatch, Mock(), encoder_outputs=encoder_outputs)
    # Past the checks, the request reaches preprocessing.
    server._preprocess_speech_to_text.side_effect = LookupError("preprocessed")
    request = _request(use_beam_search=True)

    if encoder_outputs:
        response = await _transcribe(server, request)
        assert isinstance(response, ErrorResponse)
        assert response.error.code == 400
        assert "beam search" in response.error.message
    else:
        with pytest.raises(LookupError, match="preprocessed"):
            await _transcribe(server, request)


def _salting_server(*, encoder_outputs: bool) -> SpeechToTextBaseServing:
    """A server whose renderer returns each prompt's extras as its engine input."""
    # Two identical 30 s chunks.
    chunks = [np.zeros(30 * 16000, dtype=np.float32)] * 2
    server = SpeechToTextBaseServing.__new__(SpeechToTextBaseServing)
    server._decode_and_chunk_speech_async = AsyncMock(return_value=(chunks, 60.0))
    server.asr_config = SpeechToTextConfig(
        sample_rate=16000.0,
        max_audio_clip_s=30,
        overlap_chunk_second=1,
        min_energy_split_window_size=1600,
    )
    model_cls = MagicMock(
        supports_encoder_outputs=encoder_outputs,
        supports_segment_timestamp=False,
        supports_explicit_language_detection=False,
    )
    model_cls.validate_language.side_effect = lambda language: language
    server.model_cls = model_cls
    server.model_config = MagicMock()
    server.task_type = "transcribe"  # type: ignore[misc]  # Final, set in __init__
    server.renderer = MagicMock()
    server.renderer.render_cmpl_async = AsyncMock(
        side_effect=lambda prompts, prompt_extras=None: [
            dict(prompt_extras or {}) for _ in prompts
        ]
    )
    return server


async def _salts(server, *, word: bool, request_id: str = "transcribe-outer"):
    request = MagicMock(
        language="en",
        to_language=None,
        response_format="verbose_json",
        timestamp_granularities=["word"] if word else [],
    )
    with patch(
        "vllm.entrypoints.speech_to_text.base.serving.parse_model_prompt",
        return_value={"prompt": "chunk"},
    ):
        engine_inputs, _, _ = await server._preprocess_speech_to_text(
            request=request, audio_data=b"audio", request_id=request_id
        )
    return [engine_input.get("cache_salt") for engine_input in engine_inputs]


@pytest.mark.asyncio
async def test_every_chunk_gets_a_fresh_cache_salt():
    server = _salting_server(encoder_outputs=True)
    # The client sends the same X-Request-Id twice, with identical chunks.
    salts = await _salts(server, word=True) + await _salts(server, word=True)
    assert len(set(salts)) == 4
    assert not any("outer" in salt for salt in salts)


@pytest.mark.asyncio
@pytest.mark.parametrize(("encoder_outputs", "word"), [(False, True), (True, False)])
async def test_no_cache_salt_without_encoder_outputs_for_words(encoder_outputs, word):
    server = _salting_server(encoder_outputs=encoder_outputs)
    assert await _salts(server, word=word) == [None, None]
