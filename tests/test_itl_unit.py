# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# tests/test_itl_unit.py
import asyncio
import json
import types
from dataclasses import dataclass

import pytest

# Under test: your patched function
import vllm.benchmarks.lib.endpoint_request_func as ep

# ----------------- fakes -----------------


class DummyStreamedResponseHandler:
    """Minimal stand-in for the real SSE handler.
    Returns the incoming SSE frames split on blank lines.
    """

    def add_chunk(self, chunk_bytes: bytes) -> list[str]:
        s = chunk_bytes.decode("utf-8")
        out: list[str] = []
        for part in s.split("\n\n"):
            part = part.strip()
            if part:
                out.append(part)
        return out


class _FakeContent:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def iter_any(self):
        for c in self._chunks:
            await asyncio.sleep(0)
            yield c


class _FakeResponse:
    def __init__(self, chunks: list[bytes], status: int = 200):
        self.content = _FakeContent(chunks)
        self.status = status
        self.reason = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self, chunks: list[bytes], status: int = 200):
        self._chunks = chunks
        self._status = status

    def post(self, url, json=None, headers=None):
        return _FakeResponse(self._chunks, status=self._status)


@dataclass
class _RFI:
    api_url: str = "http://localhost/v1/chat/completions"
    model_name: str | None = None
    model: str = "dummy-model"
    output_len: int = 128
    prompt_len: int = 16
    multi_modal_content: dict | None = None


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


class _Clock:
    """Deterministic perf_counter() so we can assert exact values."""

    def __init__(self, times: list[float]):
        self.t = times
        self.i = 0

    def perf_counter(self):
        val = self.t[min(self.i, len(self.t) - 1)]
        self.i += 1
        return val


# ----------------- fixtures -----------------


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # Avoid header logic failing due to missing key
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    # Keep helpers from doing anything external
    monkeypatch.setattr(ep, "StreamedResponseHandler", DummyStreamedResponseHandler)
    monkeypatch.setattr(ep, "_get_chat_content", lambda rfi, mm_position="last": "hi")
    monkeypatch.setattr(ep, "_update_payload_common", lambda payload, rfi: None)
    monkeypatch.setattr(ep, "_update_headers_common", lambda headers, rfi: None)


# ----------------- tests -----------------


@pytest.mark.asyncio
async def test_usage_continuous_coalesced_equal_split(monkeypatch):
    """Usage deltas: 1 (sets TTFT), then Δ=3 in one chunk, then Δ=2.
    We expect ITL seconds: [0.1, 0.1, 0.1, 0.15, 0.15]
    """
    # perf_counter timeline (seconds):
    # start=0.0, then usage curr=1 at 0.1 (TTFT=0.1)
    # next usage curr=4 at 0.4 (Δ=3 → dt=0.3 -> 0.1 each)
    # next usage curr=6 at 0.7 (Δ=2 → dt=0.3 -> 0.15 each)
    clk = _Clock([0.0, 0.1, 0.4, 0.7])
    monkeypatch.setattr(
        ep, "time", types.SimpleNamespace(perf_counter=clk.perf_counter)
    )

    chunks = [
        _sse({"usage": {"completion_tokens": 1}}),
        _sse({"usage": {"completion_tokens": 4}}),
        _sse({"usage": {"completion_tokens": 6}}),
    ]
    rfi = _RFI()

    out = await ep.async_request_openai_chat_completions(
        rfi, _FakeSession(chunks), None
    )
    assert out.success
    assert out.ttft == pytest.approx(0.1, rel=0, abs=1e-9)
    assert out.output_tokens == 6
    assert out.itl == pytest.approx([0.1, 0.1, 0.1, 0.15, 0.15], rel=0, abs=1e-9)
    assert out.latency == pytest.approx(0.7, rel=0, abs=1e-9)


@pytest.mark.asyncio
async def test_end_only_usage_first_chunk_extras_zero_ms(monkeypatch):
    """Content comes first (sets TTFT). Usage only once at end (curr=N).
    Expect TTFT from first content; ITL has (N-1) zeros.
    """
    # Times: start=0.0, content at 0.1 (TTFT), content at 0.2, usage at 0.5 (curr=4)
    clk = _Clock([0.0, 0.1, 0.2, 0.5])
    monkeypatch.setattr(
        ep, "time", types.SimpleNamespace(perf_counter=clk.perf_counter)
    )

    chunks = [
        _sse({"choices": [{"delta": {"content": "H"}}]}),
        _sse({"choices": [{"delta": {"content": "i"}}]}),
        _sse({"usage": {"completion_tokens": 4}}),
    ]
    rfi = _RFI()

    out = await ep.async_request_openai_chat_completions(
        rfi, _FakeSession(chunks), None
    )
    assert out.success
    assert out.ttft == pytest.approx(0.1, rel=0, abs=1e-9)
    assert out.itl == pytest.approx([0.0, 0.0, 0.0], rel=0, abs=1e-9)
    assert out.output_tokens == 4
    assert out.latency == pytest.approx(0.5, rel=0, abs=1e-9)
    assert out.generated_text == "Hi"


@pytest.mark.asyncio
async def test_zero_or_negative_deltas_do_not_extend(monkeypatch):
    """Repeated usage counters (Δ<=0) must not add ITL entries."""
    # start=0.0, curr=1 at 0.2, curr=1 again at 0.3 (Δ=0 ignore),
    # curr=2 at 0.4 → Δ=1, dt=0.1
    clk = _Clock([0.0, 0.2, 0.3, 0.4])
    monkeypatch.setattr(
        ep, "time", types.SimpleNamespace(perf_counter=clk.perf_counter)
    )

    chunks = [
        _sse({"usage": {"completion_tokens": 1}}),
        _sse({"usage": {"completion_tokens": 1}}),  # Δ=0 → ignore
        _sse({"usage": {"completion_tokens": 2}}),  # Δ=1 → 0.1s
    ]
    rfi = _RFI()

    out = await ep.async_request_openai_chat_completions(
        rfi, _FakeSession(chunks), None
    )
    assert out.success
    assert out.ttft == pytest.approx(0.2, rel=0, abs=1e-9)
    assert out.itl == pytest.approx([0.1], rel=0, abs=1e-9)
