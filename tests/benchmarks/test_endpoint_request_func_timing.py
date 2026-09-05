# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Timing-accounting invariants for the streaming benchmark request functions.

serve.py derives TPOT as (latency - ttft) / (output_len - 1), which only means
"average inter-token latency" if latency - ttft == sum(itl). These tests pin
that identity, and pin that the trailing choice-less usage chunk does not
extend the measured end of the request.

Driven against a fake session and a scripted clock: no GPU, model, or socket.
"""

import asyncio
import json
from typing import Any

import numpy as np
import pytest

import vllm.benchmarks.lib.endpoint_request_func as request_func_module
from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    async_request_openai_audio,
    async_request_openai_chat_completions,
    async_request_openai_completions,
)

pytestmark = pytest.mark.skip_global_cleanup

AUDIO_SAMPLE_RATE = 16_000


class _ScriptedClock:
    """A deterministic stand-in for time.perf_counter.

    Each read advances the clock by exactly one tick, so what a code path
    observes depends only on how many times it reads the clock. A stray or
    misattributed read then shows up as a whole tick of drift rather than as
    sub-microsecond jitter, which keeps these assertions free of flakiness.
    """

    def __init__(self, tick: float = 1.0, start: float = 1000.0) -> None:
        self.tick = tick
        self._now = start

    def __call__(self) -> float:
        value = self._now
        self._now += self.tick
        return value


class _FakeTime:
    """Namespace substituted for the module-level time import."""

    def __init__(self, perf_counter) -> None:
        self.perf_counter = perf_counter


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _token_chunk(index: int, *, chat: bool) -> bytes:
    if chat:
        choice = {"index": 0, "delta": {"content": f"tok{index} "}}
    else:
        choice = {"index": 0, "text": f"tok{index} "}
    return _sse({"id": "cmpl-test", "object": "x", "choices": [choice]})


def _usage_chunk(n_tokens: int) -> bytes:
    return _sse(
        {
            "id": "cmpl-test",
            "object": "x",
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": n_tokens,
                "total_tokens": 7 + n_tokens,
            },
        }
    )


def _build_stream(
    n_tokens: int,
    *,
    chat: bool,
    ttft_delay: float = 0.0,
    itl_delay: float = 0.0,
    usage_delay: float = 0.0,
) -> list[tuple[float, bytes]]:
    """Build ``(delay_before_chunk, payload)`` pairs for one streamed response.

    ``vllm bench serve`` always requests ``stream_options.include_usage``, so a
    choice-less usage chunk always trails the final token.
    """
    chunks = [
        (ttft_delay if i == 0 else itl_delay, _token_chunk(i, chat=chat))
        for i in range(n_tokens)
    ]
    chunks.append((usage_delay, _usage_chunk(n_tokens)))
    chunks.append((0.0, b"data: [DONE]\n\n"))
    return chunks


class _FakeContent:
    def __init__(self, chunks: list[tuple[float, bytes]]) -> None:
        self._chunks = chunks

    async def iter_any(self):
        for delay, payload in self._chunks:
            if delay:
                await asyncio.sleep(delay)
            yield payload


class _FakeResponse:
    def __init__(self, chunks: list[tuple[float, bytes]]) -> None:
        self.status = 200
        self.reason = "OK"
        self.content = _FakeContent(chunks)

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeSession:
    """Returns a scripted SSE stream for any request, without touching a socket."""

    def __init__(self, chunks: list[tuple[float, bytes]]) -> None:
        self._chunks = chunks

    def post(self, **kwargs: Any) -> _FakeResponse:
        del kwargs
        return _FakeResponse(self._chunks)


# (id, request function, api url, is_chat_payload)
STREAMING_ENDPOINTS = [
    (
        "completions",
        async_request_openai_completions,
        "http://test/v1/completions",
        False,
    ),
    (
        "chat",
        async_request_openai_chat_completions,
        "http://test/v1/chat/completions",
        True,
    ),
    (
        "audio",
        async_request_openai_audio,
        "http://test/v1/audio/transcriptions",
        True,
    ),
]


def _make_input(api_url: str, n_tokens: int, endpoint_id: str) -> RequestFuncInput:
    multi_modal_content = None
    if endpoint_id == "audio":
        # A short silent clip; only its duration is read by the request func.
        multi_modal_content = {
            "audio": (
                np.zeros(AUDIO_SAMPLE_RATE // 10, dtype=np.float32),
                AUDIO_SAMPLE_RATE,
            )
        }
    return RequestFuncInput(
        prompt="hello",
        api_url=api_url,
        prompt_len=7,
        output_len=n_tokens,
        model="test-model",
        multi_modal_content=multi_modal_content,
    )


def _run(request_func, api_url: str, endpoint_id: str, chunks, n_tokens: int):
    output = asyncio.run(
        request_func(
            _make_input(api_url, n_tokens, endpoint_id),
            _FakeSession(chunks),
        )
    )
    assert output.success, output.error
    return output


@pytest.mark.parametrize(
    "endpoint_id,request_func,api_url,chat",
    STREAMING_ENDPOINTS,
    ids=[e[0] for e in STREAMING_ENDPOINTS],
)
def test_decode_span_identity_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_id: str,
    request_func,
    api_url: str,
    chat: bool,
) -> None:
    """latency - ttft must equal sum(itl) exactly.

    On a scripted clock this can only drift if a code path reads the clock an
    extra time or attributes a read to the wrong event.
    """
    tick = 1.0
    monkeypatch.setattr(
        request_func_module, "time", _FakeTime(_ScriptedClock(tick=tick))
    )

    n_tokens = 5
    chunks = _build_stream(n_tokens, chat=chat)
    output = _run(request_func, api_url, endpoint_id, chunks, n_tokens)

    assert len(output.itl) == n_tokens - 1
    assert output.output_tokens == n_tokens

    residual = (output.latency - output.ttft) - sum(output.itl)
    assert residual == pytest.approx(0.0, abs=tick / 1000), (
        f"{endpoint_id}: (latency - ttft) - sum(itl) = {residual!r}; "
        f"TPOT is derived from latency - ttft, so it must match the measured "
        f"inter-token latencies"
    )


@pytest.mark.parametrize(
    "endpoint_id,request_func,api_url,chat",
    STREAMING_ENDPOINTS,
    ids=[e[0] for e in STREAMING_ENDPOINTS],
)
def test_trailing_usage_chunk_does_not_extend_latency(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_id: str,
    request_func,
    api_url: str,
    chat: bool,
) -> None:
    """E2E latency must end at the last token, not at the usage trailer.

    The usage chunk carries no token, so counting it stretches E2EL (and
    therefore TPOT) by the delivery time of one extra SSE frame. The [DONE]
    sentinel is already excluded on the same grounds.
    """
    tick = 1.0
    clock = _ScriptedClock(tick=tick)
    monkeypatch.setattr(request_func_module, "time", _FakeTime(clock))

    n_tokens = 5
    chunks = _build_stream(n_tokens, chat=chat)
    output = _run(request_func, api_url, endpoint_id, chunks, n_tokens)

    # Clock reads: 1 for the start time, then 1 per non-[DONE] chunk. The last
    # token is therefore read at start + n_tokens ticks; the usage chunk one
    # tick later.
    expected_latency = n_tokens * tick
    assert output.latency == pytest.approx(expected_latency, abs=tick / 1000), (
        f"{endpoint_id}: latency {output.latency} should stop at the final "
        f"token ({expected_latency}), not run on to the usage chunk "
        f"({expected_latency + tick})"
    )


@pytest.mark.parametrize(
    "endpoint_id,request_func,api_url,chat",
    STREAMING_ENDPOINTS,
    ids=[e[0] for e in STREAMING_ENDPOINTS],
)
def test_decode_span_identity_holds_with_real_clock(
    endpoint_id: str,
    request_func,
    api_url: str,
    chat: bool,
) -> None:
    """The identity also holds against the real clock, with real gaps.

    The residual differences out the same recorded timestamps, so wall clock
    jitter cancels and only float rounding remains; a microsecond bound is
    ample rather than tight.
    """
    n_tokens = 4
    chunks = _build_stream(
        n_tokens, chat=chat, ttft_delay=0.02, itl_delay=0.01, usage_delay=0.02
    )
    output = _run(request_func, api_url, endpoint_id, chunks, n_tokens)

    residual = (output.latency - output.ttft) - sum(output.itl)
    assert abs(residual) < 1e-6, (
        f"{endpoint_id}: (latency - ttft) - sum(itl) drifted by {residual:.9f}s"
    )
