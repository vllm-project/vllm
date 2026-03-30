# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from vllm.entrypoints.openai.engine.protocol import (
    RequestResponseMetadata,
    UsageInfo,
)
from vllm.entrypoints.openai.request_stats_headers import (
    build_request_stats_headers,
)
from vllm.v1.metrics.stats import RequestStateStats


def test_build_request_stats_headers_basic():
    """Headers are computed correctly from known timestamps."""
    now = time.time()
    stats = RequestStateStats(
        arrival_time=now - 1.0,
        queued_ts=100.0,
        scheduled_ts=100.05,
        first_token_ts=100.15,
        last_token_ts=100.45,
        num_generation_tokens=10,
    )
    usage = UsageInfo(
        prompt_tokens=50,
        completion_tokens=10,
        total_tokens=60,
    )
    headers = build_request_stats_headers(
        metrics=stats,
        usage=usage,
        num_cached_tokens=5,
    )

    # All headers use x-vllm- prefix
    for key in headers:
        assert key.startswith("x-vllm-"), f"Header {key} missing x-vllm- prefix"

    assert float(headers["x-vllm-queue-time"]) == round((100.05 - 100.0) * 1000, 2)
    assert float(headers["x-vllm-prefill-time"]) == round((100.15 - 100.05) * 1000, 2)
    assert float(headers["x-vllm-decode-time"]) == round((100.45 - 100.15) * 1000, 2)
    assert float(headers["x-vllm-inference-time"]) == round((100.45 - 100.05) * 1000, 2)

    assert headers["x-vllm-prompt-tokens"] == "50"
    assert headers["x-vllm-completion-tokens"] == "10"
    assert headers["x-vllm-cached-tokens"] == "5"

    # tokens-per-second: 10 tokens / 0.3s decode = 33.33
    decode_time_s = 100.45 - 100.15
    expected_tps = round(10 / decode_time_s, 2)
    assert float(headers["x-vllm-tokens-per-second"]) == expected_tps

    total_time = float(headers["x-vllm-total-time"])
    assert 900 < total_time < 1500


def test_build_request_stats_headers_zero_timestamps():
    """When timestamps are 0 (not set), timing headers show 0."""
    stats = RequestStateStats(
        arrival_time=time.time(),
        queued_ts=0.0,
        scheduled_ts=0.0,
        first_token_ts=0.0,
        last_token_ts=0.0,
    )
    usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    headers = build_request_stats_headers(
        metrics=stats, usage=usage, num_cached_tokens=0
    )

    assert headers["x-vllm-queue-time"] == "0.00"
    assert headers["x-vllm-prefill-time"] == "0.00"
    assert headers["x-vllm-decode-time"] == "0.00"
    assert headers["x-vllm-inference-time"] == "0.00"
    assert headers["x-vllm-tokens-per-second"] == "0.00"


def test_build_request_stats_headers_partial_timestamps():
    """When scheduled but cancelled before tokens, timing values clamp to 0."""
    stats = RequestStateStats(
        arrival_time=time.time() - 0.5,
        queued_ts=100.0,
        scheduled_ts=100.05,
        first_token_ts=0.0,
        last_token_ts=0.0,
    )
    usage = UsageInfo(prompt_tokens=20, completion_tokens=0, total_tokens=20)
    headers = build_request_stats_headers(
        metrics=stats, usage=usage, num_cached_tokens=0
    )

    assert float(headers["x-vllm-prefill-time"]) == 0.0
    assert float(headers["x-vllm-decode-time"]) == 0.0
    assert float(headers["x-vllm-inference-time"]) == 0.0
    assert float(headers["x-vllm-tokens-per-second"]) == 0.0
    assert float(headers["x-vllm-queue-time"]) == 50.0


def test_build_request_stats_headers_zero_decode_time_with_tokens():
    """Division by zero guard: tokens exist but decode time is 0."""
    stats = RequestStateStats(
        arrival_time=time.time() - 0.1,
        queued_ts=100.0,
        scheduled_ts=100.05,
        first_token_ts=100.10,
        last_token_ts=100.10,  # same as first_token_ts
    )
    usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    headers = build_request_stats_headers(
        metrics=stats, usage=usage, num_cached_tokens=0
    )

    assert headers["x-vllm-decode-time"] == "0.00"
    assert headers["x-vllm-tokens-per-second"] == "0.00"


def _create_test_app(enable_headers: bool) -> FastAPI:
    """Create a minimal FastAPI app with the stats middleware."""
    from vllm.entrypoints.openai.request_stats_headers import (
        build_request_stats_headers,
    )

    app = FastAPI()

    class Args:
        enable_request_stats_headers = enable_headers

    app.state.args = Args()

    @app.middleware("http")
    async def request_stats_headers_middleware(request: Request, call_next):
        response = await call_next(request)
        if not getattr(
            request.app.state.args,
            "enable_request_stats_headers",
            False,
        ):
            return response
        metadata = getattr(request.state, "request_metadata", None)
        if (
            metadata is None
            or metadata.request_stats is None
            or metadata.final_usage_info is None
        ):
            return response
        headers = build_request_stats_headers(
            metrics=metadata.request_stats,
            usage=metadata.final_usage_info,
            num_cached_tokens=metadata.num_cached_tokens,
        )
        for key, value in headers.items():
            response.headers[key] = value
        return response

    @app.get("/test-with-stats")
    async def test_with_stats(request: Request):
        metadata = RequestResponseMetadata(request_id="test-123")
        metadata.final_usage_info = UsageInfo(
            prompt_tokens=50, completion_tokens=10, total_tokens=60
        )
        metadata.request_stats = RequestStateStats(
            arrival_time=time.time() - 1.0,
            queued_ts=100.0,
            scheduled_ts=100.05,
            first_token_ts=100.15,
            last_token_ts=100.45,
            num_generation_tokens=10,
        )
        metadata.num_cached_tokens = 5
        request.state.request_metadata = metadata
        return JSONResponse(content={"ok": True})

    @app.get("/test-no-stats")
    async def test_no_stats(request: Request):
        return JSONResponse(content={"ok": True})

    @app.get("/test-partial-stats")
    async def test_partial_stats(request: Request):
        metadata = RequestResponseMetadata(request_id="test-456")
        metadata.request_stats = RequestStateStats(
            arrival_time=time.time(),
            queued_ts=100.0,
            scheduled_ts=100.05,
        )
        # final_usage_info is None
        request.state.request_metadata = metadata
        return JSONResponse(content={"ok": True})

    return app


@pytest.mark.asyncio
async def test_middleware_flag_disabled():
    """No headers when flag is disabled."""
    app = _create_test_app(enable_headers=False)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/test-with-stats")
    assert resp.status_code == 200
    assert "x-vllm-total-time" not in resp.headers


@pytest.mark.asyncio
async def test_middleware_no_metadata():
    """No headers when request_metadata is not set."""
    app = _create_test_app(enable_headers=True)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/test-no-stats")
    assert resp.status_code == 200
    assert "x-vllm-total-time" not in resp.headers


@pytest.mark.asyncio
async def test_middleware_missing_usage():
    """No headers when final_usage_info is None."""
    app = _create_test_app(enable_headers=True)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/test-partial-stats")
    assert resp.status_code == 200
    assert "x-vllm-total-time" not in resp.headers


@pytest.mark.asyncio
async def test_middleware_full_stats():
    """All headers present when flag enabled and stats available."""
    app = _create_test_app(enable_headers=True)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/test-with-stats")
    assert resp.status_code == 200
    assert "x-vllm-total-time" in resp.headers
    assert "x-vllm-queue-time" in resp.headers
    assert "x-vllm-inference-time" in resp.headers
    assert "x-vllm-prefill-time" in resp.headers
    assert "x-vllm-decode-time" in resp.headers
    assert "x-vllm-prompt-tokens" in resp.headers
    assert "x-vllm-completion-tokens" in resp.headers
    assert "x-vllm-cached-tokens" in resp.headers
    assert "x-vllm-tokens-per-second" in resp.headers
    assert resp.headers["x-vllm-prompt-tokens"] == "50"
    assert resp.headers["x-vllm-completion-tokens"] == "10"
    assert resp.headers["x-vllm-cached-tokens"] == "5"
