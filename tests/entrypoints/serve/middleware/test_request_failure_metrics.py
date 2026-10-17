# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Requests that fail in the API server never reach the engine, so they must
be counted by ``vllm:request_failure`` to be observable."""

from argparse import Namespace

import httpx
import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse
from prometheus_client import REGISTRY

from vllm.entrypoints.launchers.api_server.entry import build_app
from vllm.entrypoints.serve.middleware.request_failures import (
    init_request_failure_metrics,
    mark_generation_started,
)
from vllm.exceptions import VLLMValidationError


@pytest.fixture(scope="module")
def should_do_global_cleanup_after_test() -> bool:
    # This suite never initializes distributed/accelerator state.
    return False


@pytest.fixture(scope="module")
def app():
    args = Namespace(
        disable_fastapi_docs=True,
        enable_offline_docs=False,
        root_path=None,
        allowed_origins=["*"],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=None,
        enable_request_id_headers=False,
        enable_fault_tolerance=False,
        middleware=[],
        log_error_stack=False,
    )
    app = build_app(args, supported_tasks=())
    init_request_failure_metrics(model_name="test-model")

    @app.get("/success")
    async def success():
        return {"status": "ok"}

    @app.get("/invalid_input")
    async def invalid_input():
        raise VLLMValidationError("bad parameter", parameter="temperature")

    @app.get("/unhandled_error")
    async def unhandled_error():
        raise RuntimeError("unexpected server error")

    @app.get("/fails_after_generation_started")
    async def fails_after_generation_started(raw_request: Request):
        mark_generation_started(raw_request)
        raise RuntimeError("unexpected server error")

    @app.get("/stream_breaks_off")
    async def stream_breaks_off():
        async def generate():
            yield "data: first\n\n"
            raise RuntimeError("unexpected server error")

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app


def _num_failures(stage: str, code: int) -> float:
    return sum(
        sample.value
        for metric in REGISTRY.collect()
        for sample in metric.samples
        if sample.name == "vllm:request_failure_total"
        and sample.labels["stage"] == stage
        and sample.labels["code"] == str(code)
    )


async def _get(app, path: str) -> httpx.Response:
    # raise_app_exceptions=False lets ServerErrorMiddleware turn unhandled
    # exceptions into responses, like a real server would.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        return await client.get(path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,code,num_recorded",
    [
        ("/invalid_input", 400, 1),
        ("/unhandled_error", 500, 1),
        ("/success", 200, 0),
        # Unmatched paths and probes are not requests for the model.
        ("/does_not_exist", 404, 0),
        # Not an input processing failure.
        ("/fails_after_generation_started", 500, 0),
    ],
)
async def test_input_processing_failures_are_counted(app, path, code, num_recorded):
    before = _num_failures("input_processing", code)

    response = await _get(app, path)

    assert response.status_code == code
    assert _num_failures("input_processing", code) - before == num_recorded


@pytest.mark.asyncio
async def test_stream_breaking_off_is_counted(app):
    """The status was already sent as 200 when the stream generator raised."""
    before = _num_failures("streaming", 500)

    response = await _get(app, "/stream_breaks_off")

    assert response.status_code == 200
    assert _num_failures("streaming", 500) - before == 1
