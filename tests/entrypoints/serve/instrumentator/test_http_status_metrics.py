# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that http_requests_total metric records correct status codes.

Regression test for: Prometheus http_requests_total records 4xx exceptions
(ValueError, TypeError, etc.) as 5xx because they propagate through the
PrometheusInstrumentatorMiddleware before being caught by ServerErrorMiddleware.
"""

from argparse import Namespace
from http import HTTPStatus

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry
from prometheus_fastapi_instrumentator import Instrumentator

from vllm.entrypoints.serve.utils.server_utils import exception_handler
from vllm.exceptions import VLLMNotFoundError, VLLMValidationError


@pytest.fixture
def registry():
    """Create a fresh Prometheus registry for each test."""
    return CollectorRegistry()


@pytest.fixture
def app(registry):
    """Create a minimal FastAPI app that mirrors vLLM's exception handler
    and Prometheus middleware setup."""

    app = FastAPI()

    # Mock app state that exception_handler needs
    app.state.args = Namespace(log_error_stack=False)

    # Register exception handlers exactly as vLLM does in build_app()
    app.exception_handler(HTTPException)(_http_exception_handler)
    app.exception_handler(RequestValidationError)(_validation_exception_handler)
    app.exception_handler(ValueError)(exception_handler)
    app.exception_handler(TypeError)(exception_handler)
    app.exception_handler(OverflowError)(exception_handler)
    app.exception_handler(NotImplementedError)(exception_handler)
    app.exception_handler(VLLMValidationError)(exception_handler)
    app.exception_handler(VLLMNotFoundError)(exception_handler)
    app.exception_handler(Exception)(exception_handler)

    # Instrument with Prometheus (same as vLLM's attach_router)
    Instrumentator(
        excluded_handlers=["/metrics"],
        registry=registry,
    ).add().instrument(app)

    # Test routes that raise different exception types
    @app.get("/raise_value_error")
    async def raise_value_error():
        raise ValueError("invalid input value")

    @app.get("/raise_type_error")
    async def raise_type_error():
        raise TypeError("wrong type")

    @app.get("/raise_overflow_error")
    async def raise_overflow_error():
        raise OverflowError("number too large")

    @app.get("/raise_not_implemented_error")
    async def raise_not_implemented_error():
        raise NotImplementedError("feature not supported")

    @app.get("/raise_vllm_validation_error")
    async def raise_vllm_validation_error():
        raise VLLMValidationError("bad parameter", parameter="temperature")

    @app.get("/raise_vllm_not_found_error")
    async def raise_vllm_not_found_error():
        raise VLLMNotFoundError("model not found")

    @app.get("/raise_http_exception_400")
    async def raise_http_exception_400():
        raise HTTPException(status_code=400, detail="bad request")

    @app.get("/raise_http_exception_404")
    async def raise_http_exception_404():
        raise HTTPException(status_code=404, detail="not found")

    @app.get("/raise_runtime_error")
    async def raise_runtime_error():
        raise RuntimeError("unexpected server error")

    @app.get("/success")
    async def success():
        return {"status": "ok"}

    return app


async def _http_exception_handler(req: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


async def _validation_exception_handler(req: Request, exc: RequestValidationError):
    return JSONResponse({"error": str(exc)}, status_code=HTTPStatus.BAD_REQUEST)


def _get_http_requests_total(registry, method: str, handler: str):
    """Extract the http_requests_total metric values grouped by status.

    Returns a dict like {"2xx": 1.0, "5xx": 1.0} for the given handler.
    """
    results: dict[str, float] = {}
    for metric in registry.collect():
        if metric.name == "http_requests":
            for sample in metric.samples:
                if (
                    sample.name == "http_requests_total"
                    and sample.labels.get("method") == method
                    and sample.labels.get("handler") == handler
                ):
                    status = sample.labels.get("status")
                    results[status] = results.get(status, 0) + sample.value
    return results


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint,expected_status_group,expected_http_code",
    [
        # These should record as 4xx in Prometheus
        ("/raise_value_error", "4xx", 400),
        ("/raise_type_error", "4xx", 400),
        ("/raise_overflow_error", "4xx", 400),
        ("/raise_vllm_validation_error", "4xx", 400),
        ("/raise_vllm_not_found_error", "4xx", 404),
        ("/raise_http_exception_400", "4xx", 400),
        ("/raise_http_exception_404", "4xx", 404),
        # NotImplementedError returns 501 which is still 5xx group
        ("/raise_not_implemented_error", "5xx", 501),
        # These should record as 5xx in Prometheus (genuine server errors)
        ("/raise_runtime_error", "5xx", 500),
        # Successful requests should record as 2xx
        ("/success", "2xx", 200),
    ],
    ids=[
        "ValueError->4xx",
        "TypeError->4xx",
        "OverflowError->4xx",
        "VLLMValidationError->4xx",
        "VLLMNotFoundError->4xx",
        "HTTPException(400)->4xx",
        "HTTPException(404)->4xx",
        "NotImplementedError->5xx",
        "RuntimeError->5xx",
        "success->2xx",
    ],
)
async def test_http_requests_total_records_correct_status(
    app,
    registry,
    endpoint,
    expected_status_group,
    expected_http_code,
):
    """Verify that http_requests_total records the correct status group.

    The Prometheus metric should reflect the actual HTTP status code returned
    to the client, not a default 500 for all exceptions.
    """
    # raise_app_exceptions=False allows the full ASGI middleware stack
    # (including ServerErrorMiddleware) to handle exceptions and generate
    # proper HTTP responses, just like a real server would.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.get(endpoint)

    # Verify the HTTP response code returned to the client is correct
    assert response.status_code == expected_http_code, (
        f"Expected HTTP {expected_http_code} for {endpoint}, got {response.status_code}"
    )

    # Verify Prometheus recorded the correct status group
    metrics = _get_http_requests_total(registry, "GET", endpoint)
    assert expected_status_group in metrics, (
        f"Expected Prometheus to record '{expected_status_group}' for "
        f"{endpoint}, but got: {metrics}"
    )
    assert metrics[expected_status_group] == 1.0, (
        f"Expected 1 request recorded as '{expected_status_group}' for "
        f"{endpoint}, but got {metrics[expected_status_group]}"
    )

    # For endpoints that should be recorded as 4xx, verify they are NOT
    # incorrectly recorded as 5xx
    if expected_status_group == "4xx":
        assert "5xx" not in metrics, (
            f"Expected NO '5xx' recording for {endpoint} "
            f"(should be '4xx'), but found: {metrics}"
        )
