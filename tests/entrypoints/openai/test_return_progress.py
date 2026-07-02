# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the streaming ``return_progress`` / ``prompt_progress`` feature.

The feature reports prefill progress on the OpenAI-compatible streaming
endpoints. It is opt-in per request (``return_progress=True``) and must be
invisible to every other request, including on the engine hot path.
"""

import json

import httpx
import pytest

from ...utils import RemoteOpenAIServer

# Any small model with a chat template works here.
MODEL_NAME = "Qwen/Qwen3-0.6B"

# Long prompt so that, with a small ``--max-num-batched-tokens``, prefill is
# split into several chunks and therefore reports progress more than once.
LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 200

# A cacheable prefix followed by a long uncached tail. Priming with the prefix
# then sending ``CACHE_PREFIX + CACHE_TAIL`` yields a request whose prefill both
# hits the prefix cache (``cache > 0``) and still chunks over the tail (so at
# least one progress chunk is emitted while the cache hit is visible).
CACHE_PREFIX = "Background. " + ("vLLM serves large language models. " * 40)
CACHE_TAIL = "Now continue the detailed analysis. " * 200

_PROGRESS_KEYS = {"total", "cache", "processed", "time_ms"}

BASE_ARGS = [
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "8192",
    "--enforce-eager",
    "--max-num-seqs",
    "8",
    "--enable-chunked-prefill",
    "--max-num-batched-tokens",
    "256",
    "--enable-prefix-caching",
]


@pytest.fixture(scope="module")
def server():
    with RemoteOpenAIServer(MODEL_NAME, BASE_ARGS) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def server_with_token_details():
    # ``--enable-prompt-tokens-details`` is the server-side gate that allows
    # disclosing cached-token counts.
    args = BASE_ARGS + ["--enable-prompt-tokens-details"]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


async def _collect_sse(
    server: RemoteOpenAIServer, path: tuple[str, ...], payload: dict
) -> list[dict]:
    """Stream a raw SSE response and return the parsed ``data:`` chunks.

    The raw stream is used (rather than the typed OpenAI client) so that the
    non-standard top-level ``prompt_progress`` field is preserved as-is.
    """
    headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}
    chunks: list[dict] = []
    async with (
        httpx.AsyncClient(timeout=120.0) as client,
        client.stream(
            "POST", server.url_for(*path), json=payload, headers=headers
        ) as response,
    ):
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data.strip() == "[DONE]":
                break
            chunks.append(json.loads(data))
    return chunks


def _progress_reports(chunks: list[dict]) -> list[dict]:
    return [
        c["prompt_progress"] for c in chunks if c.get("prompt_progress") is not None
    ]


def _assert_progress_shape(reports: list[dict], *, expect_cache_hidden: bool) -> None:
    assert reports, "expected at least one prompt_progress chunk"
    total = reports[0]["total"]
    last_processed = -1
    for report in reports:
        assert set(report) == _PROGRESS_KEYS
        assert all(isinstance(value, int) for value in report.values())
        assert report["total"] == total > 0
        assert 0 <= report["processed"] <= total
        # ``processed`` advances monotonically across prefill chunks.
        assert report["processed"] >= last_processed
        last_processed = report["processed"]
        assert report["time_ms"] >= 0
        if expect_cache_hidden:
            assert report["cache"] == 0


@pytest.mark.asyncio
async def test_chat_progress_streaming(server: RemoteOpenAIServer):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": LONG_PROMPT}],
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": True,
        "return_progress": True,
    }
    chunks = await _collect_sse(server, ("v1", "chat", "completions"), payload)
    reports = _progress_reports(chunks)
    _assert_progress_shape(reports, expect_cache_hidden=True)
    # Progress chunks must carry no choices.
    for chunk in chunks:
        if chunk.get("prompt_progress") is not None:
            assert chunk["choices"] == []


@pytest.mark.asyncio
async def test_completion_progress_streaming(server: RemoteOpenAIServer):
    payload = {
        "model": MODEL_NAME,
        "prompt": LONG_PROMPT,
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": True,
        "return_progress": True,
    }
    chunks = await _collect_sse(server, ("v1", "completions"), payload)
    reports = _progress_reports(chunks)
    _assert_progress_shape(reports, expect_cache_hidden=True)


@pytest.mark.asyncio
async def test_chat_no_progress_by_default(server: RemoteOpenAIServer):
    # Regression guard for the engine-side opt-in gate: without return_progress
    # the scheduler must not emit partial-prefill outputs, so no prompt_progress
    # chunk may appear.
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": LONG_PROMPT}],
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": True,
    }
    chunks = await _collect_sse(server, ("v1", "chat", "completions"), payload)
    assert _progress_reports(chunks) == []


@pytest.mark.asyncio
async def test_completion_no_progress_by_default(server: RemoteOpenAIServer):
    payload = {
        "model": MODEL_NAME,
        "prompt": LONG_PROMPT,
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": True,
    }
    chunks = await _collect_sse(server, ("v1", "completions"), payload)
    assert _progress_reports(chunks) == []


@pytest.mark.asyncio
async def test_cache_field_respects_enable_prompt_tokens_details(
    server: RemoteOpenAIServer,
    server_with_token_details: RemoteOpenAIServer,
):
    def payload(prompt: str) -> dict:
        return {
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 5,
            "temperature": 0.0,
            "stream": True,
            "return_progress": True,
        }

    hit_prompt = CACHE_PREFIX + CACHE_TAIL

    # Default server: cached tokens must stay hidden (0) even on a cache hit.
    await _collect_sse(server, ("v1", "completions"), payload(CACHE_PREFIX))
    chunks = await _collect_sse(server, ("v1", "completions"), payload(hit_prompt))
    default_reports = _progress_reports(chunks)
    assert default_reports
    assert all(report["cache"] == 0 for report in default_reports)

    # With --enable-prompt-tokens-details the cache hit may be disclosed.
    await _collect_sse(
        server_with_token_details, ("v1", "completions"), payload(CACHE_PREFIX)
    )
    chunks = await _collect_sse(
        server_with_token_details, ("v1", "completions"), payload(hit_prompt)
    )
    details_reports = _progress_reports(chunks)
    assert details_reports
    assert max(report["cache"] for report in details_reports) > 0
