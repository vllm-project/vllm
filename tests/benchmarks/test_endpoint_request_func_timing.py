# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Timing-accounting invariants for the streaming benchmark request functions.

serve.py derives TPOT as (latency - ttft) / (output_len - 1), which only means
"average inter-token latency" if latency - ttft == sum(itl). These tests pin
that identity, pin that the trailing choice-less usage chunk does not extend
the measured end of the request, and pin that chat chunks carrying no token
(the opening role chunk, an empty delta) are skipped for TTFT and ITL while a
token that renders as an empty string is still counted.

Driven against a fake session and a scripted clock: no GPU, model, or socket.
"""

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import aiohttp
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


def _role_chunk(role: str = "assistant") -> bytes:
    choice = {"index": 0, "delta": {"role": role, "content": ""}}
    return _sse(
        {"id": "cmpl-test", "object": "chat.completion.chunk", "choices": [choice]}
    )


def _empty_chunk() -> bytes:
    choice = {"index": 0, "delta": {}}
    return _sse(
        {"id": "cmpl-test", "object": "chat.completion.chunk", "choices": [choice]}
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

    # The legacy script iterates ``response.content`` directly.
    def __aiter__(self):
        return self.iter_any()


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

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


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


def _run(
    request_func,
    api_url: str,
    endpoint_id: str,
    chunks,
    n_tokens: int,
    *,
    expect_success: bool = True,
):
    output = asyncio.run(
        request_func(
            _make_input(api_url, n_tokens, endpoint_id),
            _FakeSession(chunks),
        )
    )
    if expect_success:
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
    """Latency - ttft must equal sum(itl) exactly.

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


@pytest.mark.parametrize(
    "endpoint_id,request_func,api_url,chat",
    STREAMING_ENDPOINTS,
    ids=[e[0] for e in STREAMING_ENDPOINTS],
)
def test_usage_only_stream_is_not_reported_as_success(
    endpoint_id: str,
    request_func,
    api_url: str,
    chat: bool,
) -> None:
    """A stream with no token chunk must fail rather than report 0 ms.

    Only token chunks move the end of the request, so a degenerate stream
    carrying nothing but a usage trailer leaves latency at zero. Reporting that
    as a success would feed a zero-duration request into the benchmark
    aggregates.
    """
    chunks = [(0.0, _usage_chunk(0)), (0.0, b"data: [DONE]\n\n")]
    output = _run(request_func, api_url, endpoint_id, chunks, 0, expect_success=False)

    assert not output.success
    assert output.itl == []
    assert "TTFT" in output.error


# Significant whitespace at every edge, plus a multibyte character so that a
# split can land inside a UTF-8 sequence.
_BOUNDARY_TEXT = " leading 中 trailing "


@pytest.mark.parametrize(
    "endpoint_id,request_func,api_url,chat",
    STREAMING_ENDPOINTS,
    ids=[e[0] for e in STREAMING_ENDPOINTS],
)
def test_streamed_output_is_invariant_to_transport_chunk_boundaries(
    endpoint_id: str,
    request_func,
    api_url: str,
    chat: bool,
) -> None:
    """iter_any() yields arbitrary transport fragments, so parsing must
    give the same result no matter where the byte stream is split."""
    if chat:
        choice: dict[str, Any] = {"index": 0, "delta": {"content": _BOUNDARY_TEXT}}
    else:
        choice = {"index": 0, "text": _BOUNDARY_TEXT}
    # Not _sse(): its ensure_ascii default would escape the multibyte char.
    event = json.dumps(
        {"id": "cmpl-test", "object": "x", "choices": [choice]}, ensure_ascii=False
    )
    stream = f"data: {event}\n\ndata: [DONE]\n\n".encode()

    async def run_all_splits() -> list[tuple[int, bool, str, str]]:
        failures = []
        for split_at in range(1, len(stream)):
            chunks = [(0.0, stream[:split_at]), (0.0, stream[split_at:])]
            output = await request_func(
                _make_input(api_url, 1, endpoint_id),
                _FakeSession(chunks),
            )
            if not output.success or output.generated_text != _BOUNDARY_TEXT:
                error = output.error.splitlines()[-1] if output.error else ""
                failures.append(
                    (split_at, output.success, output.generated_text, error)
                )
        return failures

    assert asyncio.run(run_all_splits()) == []


# ``benchmarks/backend_request_func.py`` is a standalone script rather than a
# package module, so it is loaded by path. It also builds its own ClientSession,
# so the scripted stream is injected by swapping the module's aiohttp reference.
_LEGACY_PATH = Path(__file__).parents[2] / "benchmarks" / "backend_request_func.py"


class _FakeAiohttp:
    """Stands in for the legacy module's aiohttp import."""

    def __init__(self, chunks: list[tuple[float, bytes]]) -> None:
        self._chunks = chunks

    def __getattr__(self, name: str) -> Any:
        return getattr(aiohttp, name)

    def ClientSession(self, **kwargs: Any) -> _FakeSession:  # noqa: N802
        del kwargs
        return _FakeSession(self._chunks)


def _load_legacy():
    spec = importlib.util.spec_from_file_location(
        "legacy_backend_request_func", _LEGACY_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY_ENDPOINTS = [
    (
        "completions",
        "async_request_openai_completions",
        "http://test/v1/completions",
        False,
    ),
    (
        "chat",
        "async_request_openai_chat_completions",
        "http://test/v1/chat/completions",
        True,
    ),
    (
        "audio",
        "async_request_openai_audio",
        "http://test/v1/audio/transcriptions",
        True,
    ),
]


def _run_legacy(
    monkeypatch, func_name, api_url, endpoint_id, chunks, n_tokens, clock=None
):
    module = _load_legacy()
    monkeypatch.setattr(module, "aiohttp", _FakeAiohttp(chunks))
    if clock is not None:
        monkeypatch.setattr(module, "time", _FakeTime(clock))
    request_func = getattr(module, func_name)
    return asyncio.run(request_func(_make_input(api_url, n_tokens, endpoint_id)))


@pytest.mark.parametrize(
    "endpoint_id,func_name,api_url,chat",
    LEGACY_ENDPOINTS,
    ids=[e[0] for e in LEGACY_ENDPOINTS],
)
def test_legacy_decode_span_identity_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_id: str,
    func_name: str,
    api_url: str,
    chat: bool,
) -> None:
    """benchmarks/backend_request_func.py must honour the same identity."""
    tick = 1.0
    n_tokens = 5
    chunks = _build_stream(n_tokens, chat=chat)
    output = _run_legacy(
        monkeypatch,
        func_name,
        api_url,
        endpoint_id,
        chunks,
        n_tokens,
        clock=_ScriptedClock(tick=tick),
    )

    assert output.success, output.error
    assert len(output.itl) == n_tokens - 1

    residual = (output.latency - output.ttft) - sum(output.itl)
    assert residual == pytest.approx(0.0, abs=tick / 1000), (
        f"legacy {endpoint_id}: (latency - ttft) - sum(itl) = {residual!r}"
    )
    assert output.latency == pytest.approx(n_tokens * tick, abs=tick / 1000)


@pytest.mark.parametrize(
    "endpoint_id,func_name,api_url,chat",
    LEGACY_ENDPOINTS,
    ids=[e[0] for e in LEGACY_ENDPOINTS],
)
def test_legacy_usage_only_stream_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_id: str,
    func_name: str,
    api_url: str,
    chat: bool,
) -> None:
    """A usage-only stream must fail rather than report a zero-duration success."""
    chunks = [(0.0, _usage_chunk(0)), (0.0, b"data: [DONE]\n\n")]
    output = _run_legacy(monkeypatch, func_name, api_url, endpoint_id, chunks, 0)

    assert not output.success
    assert output.itl == []


def _run_chat(
    monkeypatch: pytest.MonkeyPatch,
    client: str,
    chunks: list[tuple[float, bytes]],
    n_tokens: int,
    *,
    clock: _ScriptedClock | None = None,
    expect_success: bool = True,
):
    if client == "packaged":
        if clock is not None:
            monkeypatch.setattr(request_func_module, "time", _FakeTime(clock))
        return _run(
            async_request_openai_chat_completions,
            "http://test/v1/chat/completions",
            "chat",
            chunks,
            n_tokens,
            expect_success=expect_success,
        )
    elif client == "legacy":
        output = _run_legacy(
            monkeypatch,
            "async_request_openai_chat_completions",
            "http://test/v1/chat/completions",
            "chat",
            chunks,
            n_tokens,
            clock=clock,
        )
        if expect_success:
            assert output.success, output.error
        return output
    raise ValueError(f"Unknown client: {client}")


@pytest.mark.parametrize("client", ["packaged", "legacy"])
def test_chat_role_chunk_is_not_streamed_output(
    monkeypatch: pytest.MonkeyPatch, client: str
) -> None:
    """Opening role chunk is not a streamed output: TTFT is measured to tok0."""
    tick = 1.0
    clock = _ScriptedClock(tick=tick)
    n_tokens = 5
    chunks = [
        (0.0, _role_chunk()),
        *[(0.0, _token_chunk(i, chat=True)) for i in range(n_tokens)],
        (0.0, _usage_chunk(n_tokens)),
        (0.0, b"data: [DONE]\n\n"),
    ]
    output = _run_chat(monkeypatch, client, chunks, n_tokens, clock=clock)

    assert output.success, output.error
    assert len(output.itl) == n_tokens - 1
    assert output.output_tokens == n_tokens

    # Clock read 0 is start_time (1000.0).
    # Clock read 1 is role chunk (1001.0); skipped because delta has no content.
    # Clock read 2 is tok0 chunk (1002.0); sets TTFT = 1002.0 - 1000.0 = 2 * tick.
    expected_ttft = 2.0 * tick
    assert output.ttft == pytest.approx(expected_ttft, abs=tick / 1000), (
        f"{client}: TTFT {output.ttft} should be measured to the first content chunk "
        f"({expected_ttft}), not the opening role chunk ({1.0 * tick})"
    )

    residual = (output.latency - output.ttft) - sum(output.itl)
    assert residual == pytest.approx(0.0, abs=tick / 1000), (
        f"{client}: (latency - ttft) - sum(itl) = {residual!r}"
    )


@pytest.mark.parametrize("client", ["packaged", "legacy"])
def test_chat_midstream_content_free_chunk_adds_no_interval(
    monkeypatch: pytest.MonkeyPatch, client: str
) -> None:
    """Mid-stream content-free chunk adds no ITL and does not advance timestamp."""
    tick = 1.0
    clock = _ScriptedClock(tick=tick)
    n_tokens = 3
    chunks = [
        (0.0, _role_chunk()),
        (0.0, _token_chunk(0, chat=True)),
        (0.0, _token_chunk(1, chat=True)),
        (0.0, _empty_chunk()),
        (0.0, _token_chunk(2, chat=True)),
        (0.0, _usage_chunk(n_tokens)),
        (0.0, b"data: [DONE]\n\n"),
    ]
    output = _run_chat(monkeypatch, client, chunks, n_tokens, clock=clock)

    assert output.success, output.error
    assert len(output.itl) == n_tokens - 1
    # tok0 to tok1 is 1 tick. empty_chunk is at tick 4 but does not advance
    # most_recent_timestamp (remains tick 3 from tok1). tok2 is at tick 5, so
    # interval is 5 - 3 = 2 ticks.
    assert output.itl == pytest.approx([1.0 * tick, 2.0 * tick], abs=tick / 1000), (
        f"{client}: empty chunk should not add interval or advance "
        f"most_recent_timestamp: {output.itl}"
    )

    residual = (output.latency - output.ttft) - sum(output.itl)
    assert residual == pytest.approx(0.0, abs=tick / 1000), (
        f"{client}: (latency - ttft) - sum(itl) = {residual!r}"
    )


@pytest.mark.parametrize("client", ["packaged", "legacy"])
def test_chat_tokenless_stream_is_rejected(
    monkeypatch: pytest.MonkeyPatch, client: str
) -> None:
    """A stream with no streamed output is rejected (#54708)."""
    chunks = [
        (0.0, _role_chunk()),
        (0.0, _empty_chunk()),
        (0.0, _usage_chunk(0)),
        (0.0, b"data: [DONE]\n\n"),
    ]
    output = _run_chat(monkeypatch, client, chunks, 0, expect_success=False)

    assert not output.success, f"{client}: tokenless stream should have failed"
    assert output.itl == []
    assert "TTFT" in output.error


@pytest.mark.parametrize("client", ["packaged", "legacy"])
def test_chat_empty_string_token_is_counted(
    monkeypatch: pytest.MonkeyPatch, client: str
) -> None:
    """A token that renders as an empty string is counted as streamed output."""
    tick = 1.0
    clock = _ScriptedClock(tick=tick)
    n_tokens = 3

    def _chunk(content: str) -> bytes:
        choice = {"index": 0, "delta": {"content": content}}
        return _sse(
            {"id": "cmpl-test", "object": "chat.completion.chunk", "choices": [choice]}
        )

    chunks = [
        (0.0, _role_chunk()),
        (0.0, _chunk("")),
        (0.0, _chunk("a")),
        (0.0, _chunk("b")),
        (0.0, _usage_chunk(n_tokens)),
        (0.0, b"data: [DONE]\n\n"),
    ]
    output = _run_chat(monkeypatch, client, chunks, n_tokens, clock=clock)

    assert output.success, output.error
    assert len(output.itl) == 2
    assert output.output_tokens == n_tokens

    # Clock read 0 is start_time (1000.0).
    # Clock read 1 is role chunk (1001.0); skipped because delta carries role.
    # Clock read 2 is empty token chunk (1002.0); sets TTFT = 2 * tick.
    expected_ttft = 2.0 * tick
    assert output.ttft == pytest.approx(expected_ttft, abs=tick / 1000), (
        f"{client}: TTFT {output.ttft} should be measured to the empty token chunk "
        f"({expected_ttft}), not the next chunk ({3.0 * tick})"
    )

    assert output.itl == pytest.approx([1.0 * tick, 1.0 * tick], abs=tick / 1000)

    residual = (output.latency - output.ttft) - sum(output.itl)
    assert residual == pytest.approx(0.0, abs=tick / 1000), (
        f"{client}: (latency - ttft) - sum(itl) = {residual!r}"
    )
