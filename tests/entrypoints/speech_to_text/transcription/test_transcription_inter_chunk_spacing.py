# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ASR inter-chunk spacing: ``asr_inter_chunk_separator`` and transcription
serving (mocked).

Unit tests cover the helper and ``SupportsTranscription.no_space_languages``.
Integration-style tests exercise ``OpenAIServingTranscription`` streaming and
``create_transcription`` without loading a model.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.config import ModelConfig
from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.speech_to_text.base.serving import (
    SpeechToTextBaseServing,
    asr_inter_chunk_separator,
)
from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionRequest
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription,
)
from vllm.model_executor.models.interfaces import (
    StreamingTranscriptionPostProcessor,
    SupportsTranscription,
)
from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from vllm.outputs import CompletionOutput, RequestOutput

# --- Unit: helper + protocol -------------------------------------------------


def test_default_no_space_languages_includes_zh_and_ja():
    assert SupportsTranscription.no_space_languages == {"ja", "zh"}


@pytest.mark.parametrize(
    ("language", "expected_sep"),
    [
        ("en", " "),
        ("EN", " "),
        ("zh", ""),
        ("ZH", ""),
        ("ja", ""),
        (None, " "),
    ],
)
def test_asr_inter_chunk_separator_matches_protocol(language, expected_sep):
    sep = asr_inter_chunk_separator(language, SupportsTranscription.no_space_languages)
    assert sep == expected_sep


def test_qwen3_asr_stream_processor_passes_plain_text_without_prefix():
    post_processor = (
        Qwen3ASRForConditionalGeneration.get_streaming_post_processor_cls()()
    )

    assert post_processor.process_delta("Hello", False) == "Hello"
    assert post_processor.process_delta(" world", True) == " world"


def test_qwen3_asr_stream_processor_buffers_prefix_with_leading_space():
    post_processor = (
        Qwen3ASRForConditionalGeneration.get_streaming_post_processor_cls()()
    )

    assert post_processor.process_delta(" language Eng", False) == ""
    assert post_processor.process_delta("lish<asr", False) == ""
    assert post_processor.process_delta("_text>Hello", True) == "Hello"


def test_qwen3_asr_stream_processor_keeps_independent_state():
    processor_cls = Qwen3ASRForConditionalGeneration.get_streaming_post_processor_cls()
    first_processor = processor_cls()
    second_processor = processor_cls()

    assert first_processor.process_delta(" language Eng", False) == ""
    assert second_processor.process_delta("plain text", True) == "plain text"
    assert first_processor.process_delta("lish<asr_text>Hello", True) == "Hello"


def test_qwen3_asr_stream_processor_emits_finished_incomplete_prefix():
    post_processor = (
        Qwen3ASRForConditionalGeneration.get_streaming_post_processor_cls()()
    )

    assert (
        post_processor.process_delta(" language English", True) == " language English"
    )


def test_qwen3_asr_stream_processor_stops_buffering_long_plain_prefix():
    post_processor = (
        Qwen3ASRForConditionalGeneration.get_streaming_post_processor_cls()()
    )
    text = " language " + ("x" * 50)

    assert post_processor.process_delta(text, False) == text


def test_qwen3_asr_stream_processor_stops_buffering_prefix_with_newline():
    post_processor = (
        Qwen3ASRForConditionalGeneration.get_streaming_post_processor_cls()()
    )
    text = " language English\nhello"

    assert post_processor.process_delta(text, False) == text


def test_joined_chunks_english_has_space_between():
    sep = asr_inter_chunk_separator("en", SupportsTranscription.no_space_languages)
    assert sep.join(["hello", "world"]) == "hello world"


def test_joined_chunks_chinese_has_no_space_between():
    sep = asr_inter_chunk_separator("zh", SupportsTranscription.no_space_languages)
    assert sep.join(["你好", "世界"]) == "你好世界"


# --- Integration: serving (no model) -----------------------------------------


class _StubTranscriptionModel:
    """Minimal stand-in for a SupportsTranscription implementation (no torch)."""

    no_space_languages: set[str] = {"ja", "zh"}
    supports_segment_timestamp = False

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        return SpeechToTextConfig(
            sample_rate=16000.0,
            max_audio_clip_s=5.0,
        )

    @classmethod
    def post_process_output(cls, text: str) -> str:
        return text

    @classmethod
    def get_streaming_post_processor_cls(
        cls,
    ) -> type[StreamingTranscriptionPostProcessor]:
        return StreamingTranscriptionPostProcessor


def _request_output(text: str, finish_reason: str | None = "stop") -> RequestOutput:
    return RequestOutput(
        request_id="rid",
        prompt=None,
        prompt_token_ids=None,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=(1, 2, 3),
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=finish_reason,
            )
        ],
        finished=True,
    )


def _sse_delta_contents(sse_body: str) -> list[str]:
    """Extract ``choices[0].delta.content`` from each ``data:`` line (streaming API)."""
    contents: list[str] = []
    for line in sse_body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ").strip()
        if payload == "[DONE]":
            continue
        obj = json.loads(payload)
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            if "content" in delta:
                contents.append(delta["content"])
    return contents


@pytest.mark.asyncio
async def test_transcription_stream_generator_english_inserts_space_between_chunks():
    """Online streaming: first output per audio chunk is prefixed with *separator*."""

    async def gen_hello() -> AsyncGenerator[RequestOutput, None]:
        yield _request_output("hello")

    async def gen_world() -> AsyncGenerator[RequestOutput, None]:
        yield _request_output("world")

    serving = OpenAIServingTranscription.__new__(OpenAIServingTranscription)
    serving.enable_force_include_usage = False
    serving.model_cls = _StubTranscriptionModel
    serving.streaming_post_processor_cls = (
        _StubTranscriptionModel.get_streaming_post_processor_cls()
    )
    serving.task_type = "transcribe"
    request = SimpleNamespace(
        model="stub-model",
        stream_include_usage=False,
        stream_continuous_usage_stats=False,
    )
    sep = asr_inter_chunk_separator("en", _StubTranscriptionModel.no_space_languages)
    assert sep == " "

    out_lines: list[str] = []
    agen = OpenAIServingTranscription.transcription_stream_generator(
        serving,
        request=request,
        result_generator=[gen_hello(), gen_world()],
        request_id="test-req",
        request_metadata=RequestResponseMetadata(request_id="test-req"),
        audio_duration_s=1.0,
        separator=sep,
    )
    async for line in agen:
        out_lines.append(line)
    sse = "".join(out_lines)
    combined = "".join(_sse_delta_contents(sse))
    assert combined.strip() == "hello world"


@pytest.mark.asyncio
async def test_transcription_stream_generator_chinese_no_space_between_chunks():
    async def gen_a() -> AsyncGenerator[RequestOutput, None]:
        yield _request_output("你好")

    async def gen_b() -> AsyncGenerator[RequestOutput, None]:
        yield _request_output("世界")

    serving = OpenAIServingTranscription.__new__(OpenAIServingTranscription)
    serving.enable_force_include_usage = False
    serving.model_cls = _StubTranscriptionModel
    serving.streaming_post_processor_cls = (
        _StubTranscriptionModel.get_streaming_post_processor_cls()
    )
    serving.task_type = "transcribe"
    request = SimpleNamespace(
        model="stub-model",
        stream_include_usage=False,
        stream_continuous_usage_stats=False,
    )
    sep = asr_inter_chunk_separator("zh", _StubTranscriptionModel.no_space_languages)
    assert sep == ""

    out_lines: list[str] = []
    agen = OpenAIServingTranscription.transcription_stream_generator(
        serving,
        request=request,
        result_generator=[gen_a(), gen_b()],
        request_id="test-req-zh",
        request_metadata=RequestResponseMetadata(request_id="test-req-zh"),
        audio_duration_s=1.0,
        separator=sep,
    )
    async for line in agen:
        out_lines.append(line)
    combined = "".join(_sse_delta_contents("".join(out_lines)))
    assert combined == "你好世界"


@pytest.mark.asyncio
async def test_transcription_stream_generator_strips_qwen3_asr_prefix_per_chunk():
    async def gen_hello() -> AsyncGenerator[RequestOutput, None]:
        yield _request_output("language Eng", finish_reason=None)
        yield _request_output("lish<asr", finish_reason=None)
        yield _request_output("_text>Hello", finish_reason=None)
        yield _request_output("")

    async def gen_world() -> AsyncGenerator[RequestOutput, None]:
        yield _request_output(" language Eng", finish_reason=None)
        yield _request_output("lish<asr_text>world")

    serving = OpenAIServingTranscription.__new__(OpenAIServingTranscription)
    serving.enable_force_include_usage = False
    serving.model_cls = Qwen3ASRForConditionalGeneration
    serving.streaming_post_processor_cls = (
        Qwen3ASRForConditionalGeneration.get_streaming_post_processor_cls()
    )
    serving.task_type = "transcribe"
    request = SimpleNamespace(
        model="stub-qwen3-asr",
        stream_include_usage=False,
        stream_continuous_usage_stats=False,
    )

    out_lines: list[str] = []
    agen = OpenAIServingTranscription.transcription_stream_generator(
        serving,
        request=request,
        result_generator=[gen_hello(), gen_world()],
        request_id="test-qwen3-asr",
        request_metadata=RequestResponseMetadata(request_id="test-qwen3-asr"),
        audio_duration_s=1.0,
        separator=" ",
    )
    async for line in agen:
        out_lines.append(line)

    combined = "".join(_sse_delta_contents("".join(out_lines)))
    assert combined == "Hello world"


@pytest.mark.asyncio
async def test_create_transcription_non_streaming_joins_chunks_by_language():
    """``create_transcription`` uses the same separator logic as the helper."""

    async def gen_hello() -> AsyncGenerator[RequestOutput, None]:
        yield _request_output("hello")

    async def gen_world() -> AsyncGenerator[RequestOutput, None]:
        yield _request_output("world")

    engine_client = MagicMock()
    engine_client.model_config = MagicMock()
    engine_client.model_config.get_diff_sampling_param.return_value = {
        "max_tokens": 256,
        "temperature": 0.0,
    }
    engine_client.model_config.max_model_len = 8192
    engine_client.errored = False
    engine_client.generate.side_effect = [gen_hello(), gen_world()]

    models = MagicMock(spec=OpenAIServingModels)
    models.lora_requests = {}
    models.is_base_model.return_value = True

    preprocess_mock = AsyncMock(return_value=([MagicMock(), MagicMock()], 1.0))

    with (
        patch(
            "vllm.model_executor.model_loader.get_model_cls",
            return_value=_StubTranscriptionModel,
        ),
        patch.object(
            SpeechToTextBaseServing, "_preprocess_speech_to_text", preprocess_mock
        ),
    ):
        serving = OpenAIServingTranscription(engine_client, models, request_logger=None)

        req_en = TranscriptionRequest.model_construct(
            file=MagicMock(),
            model="stub-model",
            language="en",
            stream=False,
            response_format="json",
        )
        out_en = await serving.create_transcription(
            b"\x00\x00", req_en, raw_request=None
        )
        assert not isinstance(out_en, ErrorResponse)
        assert out_en.text == "hello world"

        async def gen_nihao() -> AsyncGenerator[RequestOutput, None]:
            yield _request_output("你好")

        async def gen_shijie() -> AsyncGenerator[RequestOutput, None]:
            yield _request_output("世界")

        engine_client.generate.side_effect = [gen_nihao(), gen_shijie()]

        req_zh = TranscriptionRequest.model_construct(
            file=MagicMock(),
            model="stub-model",
            language="zh",
            stream=False,
            response_format="json",
        )
        out_zh = await serving.create_transcription(
            b"\x00\x00", req_zh, raw_request=None
        )
        assert not isinstance(out_zh, ErrorResponse)
        assert out_zh.text == "你好世界"
