# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for realtime output post-processing (no GPU / server needed).

The realtime WebSocket path must route model output through the same
post-processing hooks as the REST transcription path, so that models with a
structured output format (e.g. Qwen3-ASR's ``language Chinese<asr_text>``
header, repeated on every 5 s segment) do not leak the header to clients.
"""

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection
from vllm.model_executor.models.interfaces import (
    StreamingTranscriptionPostProcessor,
    SupportsRealtime,
)
from vllm.model_executor.models.qwen3_asr import Qwen3ASRStreamingPostProcessor


@dataclass
class _FakeCompletion:
    text: str
    token_ids: list[int]
    finish_reason: str | None = None


@dataclass
class _FakeOutput:
    outputs: list[_FakeCompletion]
    prompt_token_ids: list[int] | None = None


@dataclass
class _FakeWebSocket:
    sent: list[dict] = field(default_factory=list)

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))


class _FakeEngineClient:
    def __init__(self, outputs: list[_FakeOutput]):
        self._outputs = outputs
        self.kwargs: dict = {}

    def generate(self, **kwargs):
        self.kwargs = kwargs

        async def gen():
            for out in self._outputs:
                yield out

        return gen()


class _FakeServing:
    def __init__(self, model_cls, outputs: list[_FakeOutput]):
        self.model_cls = model_cls
        self.engine_client = _FakeEngineClient(outputs)


class _PlainRealtimeModel(SupportsRealtime):
    realtime_max_tokens = 8

    @classmethod
    async def buffer_realtime_audio(cls, audio_stream, input_stream, model_config):
        raise NotImplementedError


class _StructuredRealtimeModel(_PlainRealtimeModel):
    @classmethod
    def get_streaming_post_processor_cls(
        cls,
    ) -> type[StreamingTranscriptionPostProcessor]:
        return Qwen3ASRStreamingPostProcessor


def _tokens(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def _segment(pieces: list[str], finish: bool = True) -> list[_FakeOutput]:
    """One generation segment streamed as per-token deltas."""
    outs = []
    for i, piece in enumerate(pieces):
        last = i == len(pieces) - 1
        outs.append(
            _FakeOutput(
                outputs=[
                    _FakeCompletion(
                        text=piece,
                        token_ids=_tokens(piece),
                        finish_reason="stop" if (finish and last) else None,
                    )
                ],
                prompt_token_ids=[1, 2, 3] if i == 0 else None,
            )
        )
    return outs


async def _run(model_cls, outputs: list[_FakeOutput]):
    ws = _FakeWebSocket()
    serving = _FakeServing(model_cls, outputs)
    conn = RealtimeConnection(ws, serving)  # type: ignore[arg-type]
    conn._is_connected = True
    input_stream: asyncio.Queue[list[int]] = asyncio.Queue()

    async def audio_gen():
        if False:
            yield None

    await conn._run_generation(audio_gen(), input_stream)
    fed = []
    while not input_stream.empty():
        fed.append(input_stream.get_nowait())
    return ws.sent, fed


@pytest.mark.asyncio
async def test_plain_model_output_is_passed_through():
    outputs = _segment(["hello", " world"])
    sent, fed = await _run(_PlainRealtimeModel, outputs)

    deltas = [e["delta"] for e in sent if e["type"] == "transcription.delta"]
    done = [e for e in sent if e["type"] == "transcription.done"]
    assert deltas == ["hello", " world"]
    assert done[0]["text"] == "hello world"
    assert done[0]["usage"]["completion_tokens"] == len("hello world")
    # raw token ids are still fed back to the input stream untouched
    assert fed == [_tokens("hello"), _tokens(" world")]


@pytest.mark.asyncio
async def test_structured_header_is_stripped_per_segment():
    # Two 5 s segments; each generation repeats the Qwen3-ASR header.
    outputs = _segment(
        ["language", " Chinese", "<asr_text>", "今天", "天气"]
    ) + _segment(["language", " Chinese", "<asr_text>", "很好", "。"])
    sent, fed = await _run(_StructuredRealtimeModel, outputs)

    deltas = [e["delta"] for e in sent if e["type"] == "transcription.delta"]
    done = [e for e in sent if e["type"] == "transcription.done"]
    assert "".join(deltas) == "今天天气很好。"
    assert all("<asr_text>" not in d and "language" not in d for d in deltas)
    assert done[0]["text"] == "今天天气很好。"
    assert "<asr_text>" not in done[0]["text"]
    # header tokens are still fed back to the engine as context
    assert fed[0] == _tokens("language")
    # empty deltas (while the header is being buffered) are not sent
    assert "" not in deltas
    # usage counts raw tokens, not the cleaned text
    raw = "language Chinese<asr_text>今天天气language Chinese<asr_text>很好。"
    assert done[0]["usage"]["completion_tokens"] == len(raw)


@pytest.mark.asyncio
async def test_unstructured_output_from_structured_model_is_kept():
    # The model may skip the header; the post-processor must not drop text.
    outputs = _segment(["hello", " there"])
    sent, _ = await _run(_StructuredRealtimeModel, outputs)
    done = [e for e in sent if e["type"] == "transcription.done"]
    assert done[0]["text"] == "hello there"
