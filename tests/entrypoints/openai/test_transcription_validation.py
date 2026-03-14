# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# imports for structured outputs tests
import json
import types

import pytest

from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionResponseStreamChoice,
    TranscriptionStreamResponse,
)
from vllm.entrypoints.openai.speech_to_text.speech_to_text import OpenAISpeechToText

from ...utils import ROCM_ENV_OVERRIDES, ROCM_EXTRA_ARGS, RemoteOpenAIServer
from .conftest import add_attention_backend

MISTRAL_FORMAT_ARGS = [
    "--tokenizer_mode",
    "mistral",
    "--config_format",
    "mistral",
    "--load_format",
    "mistral",
]


async def transcribe_and_check(
    client,
    model_name: str,
    file,
    *,
    language: str,
    expected_text: str,
    expected_seconds: int | None = None,
    case_sensitive: bool = False,
):
    """Run a transcription request and assert the output contains
    *expected_text* and optionally that usage reports *expected_seconds*.

    Provides detailed failure messages with the actual transcription output.
    """
    transcription = await client.audio.transcriptions.create(
        model=model_name,
        file=file,
        language=language,
        response_format="text",
        temperature=0.0,
    )
    out = json.loads(transcription)
    out_text = out["text"]
    out_usage = out["usage"]

    if case_sensitive:
        assert expected_text in out_text, (
            f"Expected {expected_text!r} in transcription output, got: {out_text!r}"
        )
    else:
        assert expected_text.lower() in out_text.lower(), (
            f"Expected {expected_text!r} (case-insensitive) in transcription "
            f"output, got: {out_text!r}"
        )

    if expected_seconds is not None:
        assert out_usage["seconds"] == expected_seconds, (
            f"Expected {expected_seconds}s of audio, "
            f"got {out_usage['seconds']}s. Full usage: {out_usage!r}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name", ["mistralai/Voxtral-Mini-3B-2507", "Qwen/Qwen3-ASR-0.6B"]
)
async def test_basic_audio(mary_had_lamb, model_name, rocm_aiter_fa_attention):
    server_args = ["--enforce-eager", *ROCM_EXTRA_ARGS]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    # Based on https://github.com/openai/openai-cookbook/blob/main/examples/Whisper_prompting_guide.ipynb.
    with RemoteOpenAIServer(
        model_name, server_args, env_dict=ROCM_ENV_OVERRIDES
    ) as remote_server:
        client = remote_server.get_async_client()
        await transcribe_and_check(
            client,
            model_name,
            mary_had_lamb,
            language="en",
            expected_text="Mary had a little lamb",
            expected_seconds=16,
        )


@pytest.mark.asyncio
async def test_basic_audio_with_lora(mary_had_lamb, rocm_aiter_fa_attention):
    """Ensure STT (transcribe) requests can pass LoRA through to generate."""
    # ROCm SPECIFIC CONFIGURATION:
    # To ensure the test passes on ROCm, we modify the max model length to 512.
    # We DO NOT apply this to other platforms to maintain strict upstream parity.
    from vllm.platforms import current_platform

    model_name = "ibm-granite/granite-speech-3.3-2b"
    lora_model_name = "speech"
    server_args = [
        "--enforce-eager",
        "--enable-lora",
        "--max-lora-rank",
        "64",
        "--lora-modules",
        f"{lora_model_name}={model_name}",
        "--max-model-len",
        "512" if current_platform.is_rocm() else "2048",
        "--max-num-seqs",
        "1",
    ]

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    # Based on https://github.com/openai/openai-cookbook/blob/main/examples/Whisper_prompting_guide.ipynb.
    with RemoteOpenAIServer(
        model_name, server_args, env_dict=ROCM_ENV_OVERRIDES
    ) as remote_server:
        client = remote_server.get_async_client()
        await transcribe_and_check(
            client,
            lora_model_name,
            mary_had_lamb,
            language="en",
            expected_text="mary had a little lamb",
            expected_seconds=16,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name", ["google/gemma-3n-E2B-it", "Qwen/Qwen3-ASR-0.6B"]
)
async def test_basic_audio_foscolo(foscolo, rocm_aiter_fa_attention, model_name):
    # Gemma accuracy on some of the audio samples we use is particularly bad,
    # hence we use a different one here. WER is evaluated separately.
    server_args = ["--enforce-eager", *ROCM_EXTRA_ARGS]

    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(
        model_name,
        server_args,
        max_wait_seconds=480,
        env_dict=ROCM_ENV_OVERRIDES,
    ) as remote_server:
        client = remote_server.get_async_client()
        await transcribe_and_check(
            client,
            model_name,
            foscolo,
            language="it",
            expected_text="ove il mio corpo fanciulletto giacque",
        )


def _build_mock_request_output(
    *,
    text: str,
    token_ids: list[int],
    finish_reason: str | None,
    prompt_token_ids: list[int] | None = None,
):
    output = types.SimpleNamespace(
        text=text,
        token_ids=token_ids,
        finish_reason=finish_reason,
    )
    return types.SimpleNamespace(
        prompt_token_ids=prompt_token_ids,
        outputs=[output],
    )


async def _make_result_generator(*items):
    for item in items:
        yield item


def _make_mock_stt_serving() -> OpenAISpeechToText:
    serving = OpenAISpeechToText.__new__(OpenAISpeechToText)
    serving.asr_config = types.SimpleNamespace(max_audio_clip_s=5.0)
    serving.model_cls = types.SimpleNamespace(
        get_num_audio_tokens=lambda *_args, **_kwargs: 0
    )
    serving.model_config = types.SimpleNamespace()
    serving.task_type = "transcribe"
    serving.log_error_stack = False
    return serving


def test_parse_diarized_payload_unit():
    payload = (
        '{"text":"hi there",'
        '"segments":[{"id":0,"start":0.0,"end":0.8,'
        '"text":"hi there","speaker":"speaker_1"}]}'
    )

    parsed = OpenAISpeechToText._try_parse_diarized_response_payload(payload)
    assert parsed is not None

    parsed_text, segments = parsed
    assert parsed_text == "hi there"
    assert len(segments) == 1
    assert segments[0].id == 0
    assert segments[0].speaker == "speaker_1"


def test_parse_diarized_payload_fenced_json_unit():
    payload = (
        "```json\n"
        '{"segments":[{"start":0.0,"end":1.2,'
        '"text":"hello","speaker":"speaker_0"}]}\n'
        "```"
    )

    parsed = OpenAISpeechToText._try_parse_diarized_response_payload(payload)
    assert parsed is not None

    parsed_text, segments = parsed
    assert parsed_text == "hello"
    assert len(segments) == 1
    assert segments[0].id == 0


def test_parse_diarized_payload_invalid_unit():
    assert OpenAISpeechToText._try_parse_diarized_response_payload("not-json") is None


@pytest.mark.asyncio
async def test_diarized_stream_event_contract_unit():
    serving = _make_mock_stt_serving()
    request = types.SimpleNamespace(response_format="diarized_json")
    request_metadata = RequestResponseMetadata(request_id="req-diarized")

    first_chunk = _make_result_generator(
        _build_mock_request_output(
            text="Hel",
            token_ids=[11],
            finish_reason=None,
            prompt_token_ids=[1, 2, 3],
        ),
        _build_mock_request_output(
            text="lo",
            token_ids=[12],
            finish_reason="stop",
        ),
    )
    second_chunk = _make_result_generator(
        _build_mock_request_output(
            text="Bye",
            token_ids=[21, 22],
            finish_reason="stop",
            prompt_token_ids=[4, 5],
        ),
    )

    raw_events = []
    async for event in serving._diarized_stream_event_generator(
        request=request,
        list_result_generator=[first_chunk, second_chunk],
        request_metadata=request_metadata,
        audio_duration_s=9.0,
    ):
        raw_events.append(event)

    assert raw_events[-1] == "data: [DONE]\n\n"

    payloads = [
        json.loads(event.removeprefix("data: "))
        for event in raw_events[:-1]
        if event.startswith("data: {")
    ]

    def _event_kind(payload: dict) -> str:
        if "segment_id" in payload and "delta" in payload:
            return "delta"
        if "id" in payload and "speaker" in payload:
            return "segment"
        return "done"

    assert [_event_kind(payload) for payload in payloads] == [
        "delta",
        "delta",
        "segment",
        "delta",
        "segment",
        "done",
    ]

    assert payloads[0]["segment_id"] == 0
    assert payloads[1]["segment_id"] == 0
    assert payloads[2]["id"] == 0
    assert payloads[2]["start"] == 0.0
    assert payloads[2]["end"] == 5.0
    assert payloads[2]["speaker"] == "speaker_0"
    assert payloads[3]["segment_id"] == 1
    assert payloads[4]["id"] == 1
    assert payloads[4]["start"] == 5.0
    assert payloads[4]["end"] == 9.0
    assert payloads[5]["text"] == "HelloBye"

    assert all("usage" not in payload for payload in payloads)

    assert request_metadata.final_usage_info is not None
    assert request_metadata.final_usage_info.prompt_tokens == 5
    assert request_metadata.final_usage_info.completion_tokens == 4
    assert request_metadata.final_usage_info.total_tokens == 9


@pytest.mark.asyncio
async def test_diarized_stream_routing_unit():
    serving = _make_mock_stt_serving()
    request = types.SimpleNamespace(
        response_format="diarized_json",
        stream_include_usage=True,
        stream_continuous_usage_stats=True,
        model="test-model",
    )
    request_metadata = RequestResponseMetadata(request_id="req-route")

    result_generator = _make_result_generator(
        _build_mock_request_output(
            text="abc",
            token_ids=[1, 2, 3],
            finish_reason="stop",
            prompt_token_ids=[1],
        )
    )

    stream_chunks = []
    async for chunk in serving._speech_to_text_stream_generator(
        request=request,
        list_result_generator=[result_generator],
        request_id="req-route",
        request_metadata=request_metadata,
        audio_duration_s=2.0,
        chunk_object_type="transcription.chunk",
        response_stream_choice_class=TranscriptionResponseStreamChoice,
        stream_response_class=TranscriptionStreamResponse,
    ):
        stream_chunks.append(chunk)

    payload_chunks = [chunk for chunk in stream_chunks if chunk.startswith("data: {")]
    assert payload_chunks
    assert all(
        '"object":"transcription.chunk"' not in chunk for chunk in payload_chunks
    )

    payloads = [json.loads(chunk.removeprefix("data: ")) for chunk in payload_chunks]
    assert any("segment_id" in payload and "delta" in payload for payload in payloads)
    assert any("id" in payload and "speaker" in payload for payload in payloads)
    assert all("usage" not in payload for payload in payloads)
    assert stream_chunks[-1] == "data: [DONE]\n\n"
