# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.request_stats_headers import (
    build_request_stats_headers,
    request_stats_headers_middleware,
)
from vllm.v1.engine import FinishReason
from vllm.v1.metrics.stats import FinishedRequestStats


def _stats(**overrides) -> FinishedRequestStats:
    base = dict(
        finish_reason=FinishReason.STOP,
        request_id="req-1",
        e2e_latency=1.0,
        num_prompt_tokens=50,
        num_generation_tokens=10,
        max_tokens_param=None,
        queued_time=0.05,
        prefill_time=0.10,
        inference_time=0.40,
        decode_time=0.30,
        mean_time_per_output_token=0.030,
        is_corrupted=False,
        num_cached_tokens=5,
    )
    base.update(overrides)
    return FinishedRequestStats(**base)


def test_build_headers_basic():
    headers = build_request_stats_headers(_stats())

    for key in headers:
        assert key.startswith("x-vllm-"), f"{key} missing x-vllm- prefix"

    assert headers["x-vllm-total-time"] == "1000.00"
    assert headers["x-vllm-queue-time"] == "50.00"
    assert headers["x-vllm-prefill-time"] == "100.00"
    assert headers["x-vllm-inference-time"] == "400.00"
    assert headers["x-vllm-decode-time"] == "300.00"
    assert headers["x-vllm-prompt-tokens"] == "50"
    assert headers["x-vllm-completion-tokens"] == "10"
    assert headers["x-vllm-cached-tokens"] == "5"
    assert headers["x-vllm-time-per-output-token"] == "30.00"


def test_build_headers_zero_decode():
    """Single-token completion: mean_time_per_output_token is 0."""
    headers = build_request_stats_headers(
        _stats(num_generation_tokens=1, decode_time=0.0, mean_time_per_output_token=0.0)
    )
    assert headers["x-vllm-time-per-output-token"] == "0.00"
    assert headers["x-vllm-completion-tokens"] == "1"


def _create_test_app() -> FastAPI:
    app = FastAPI()
    app.middleware("http")(request_stats_headers_middleware)

    @app.get("/with-stats")
    async def with_stats(request: Request) -> JSONResponse:
        meta = RequestResponseMetadata(request_id="r")
        meta._finished_stats = _stats()
        request.state.request_metadata = meta
        return JSONResponse({"ok": True})

    @app.get("/no-stats")
    async def no_stats(request: Request) -> JSONResponse:
        meta = RequestResponseMetadata(request_id="r")
        request.state.request_metadata = meta
        return JSONResponse({"ok": True})

    @app.get("/no-metadata")
    async def no_metadata() -> JSONResponse:
        return JSONResponse({"ok": True})

    return app


@pytest.mark.asyncio
async def test_middleware_attaches_headers_when_stats_present():
    app = _create_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/with-stats")
    assert resp.status_code == 200
    assert resp.headers["x-vllm-decode-time"] == "300.00"
    assert resp.headers["x-vllm-prompt-tokens"] == "50"


@pytest.mark.asyncio
async def test_middleware_passes_through_when_finished_stats_missing():
    app = _create_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/no-stats")
    assert resp.status_code == 200
    assert "x-vllm-decode-time" not in resp.headers


@pytest.mark.asyncio
async def test_middleware_passes_through_when_metadata_missing():
    app = _create_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/no-metadata")
    assert resp.status_code == 200
    assert "x-vllm-decode-time" not in resp.headers


def test_finished_stats_is_private_not_a_schema_field():
    # The stats carrier must not be a validated/serialized pydantic field:
    # that is what dragged FinishedRequestStats into pydantic introspection.
    assert "finished_stats" not in RequestResponseMetadata.model_fields
    assert "_finished_stats" in RequestResponseMetadata.__private_attributes__

    meta = RequestResponseMetadata(request_id="r")
    assert meta._finished_stats is None
    assert "finished_stats" not in meta.model_dump()

    meta._finished_stats = _stats()
    assert meta._finished_stats.request_id == "req-1"
