# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that http_requests_total metric records correct status codes.

Regression test for: Prometheus http_requests_total records 4xx exceptions
(ValueError, TypeError, etc.) as 5xx because they propagate through the
PrometheusInstrumentatorMiddleware before being caught by ServerErrorMiddleware.
"""

from argparse import Namespace

import httpx
import pytest
from fastapi import HTTPException
from prometheus_client import CollectorRegistry

from vllm.entrypoints.launchers.api_server.entry import build_app
from vllm.exceptions import (
    VLLMNotFoundError,
    VLLMServerError,
    VLLMValidationError,
)


@pytest.fixture(scope="module")
def should_do_global_cleanup_after_test() -> bool:
    # This suite never initializes distributed/accelerator state.
    return False


def _build_args() -> Namespace:
    """Minimal args for ``build_app``; avoids ``make_arg_parser`` device probing."""
    return Namespace(
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


@pytest.fixture(scope="module")
def registry():
    """Shared Prometheus registry for the module-scoped app."""
    return CollectorRegistry()


@pytest.fixture(scope="module")
def app(registry):
    """Build the real vLLM FastAPI app once and attach probe routes that raise.

    Patch the name used by ``attach_router`` (imported into the instrumentator
    metrics module), not ``vllm.v1.metrics.prometheus`` alone — that binding is
    captured at import time.
    """
    import vllm.entrypoints.serve.instrumentator.metrics as metrics_mod

    original = metrics_mod.get_prometheus_registry
    metrics_mod.get_prometheus_registry = lambda: registry
    try:
        app = build_app(_build_args(), supported_tasks=())
    finally:
        metrics_mod.get_prometheus_registry = original

    @app.get("/raise_http_exception_400")
    async def raise_http_exception_400():
        raise HTTPException(status_code=400, detail="bad request")

    @app.get("/raise_http_exception_404")
    async def raise_http_exception_404():
        raise HTTPException(status_code=404, detail="not found")

    @app.get("/raise_request_validation_error")
    async def raise_request_validation_error(n: int):
        # Invalid ``n`` triggers FastAPI's RequestValidationError.
        return {"n": n}

    @app.get("/raise_vllm_validation_error")
    async def raise_vllm_validation_error():
        raise VLLMValidationError("bad parameter", parameter="temperature")

    @app.get("/raise_vllm_not_found_error")
    async def raise_vllm_not_found_error():
        raise VLLMNotFoundError("model not found")

    @app.get("/raise_vllm_server_error")
    async def raise_vllm_server_error():
        # Bare VLLMServerError goes through vllm_error_handler → 500.
        # EngineGenerateError / EngineDeadError are not used here: they call
        # terminate_if_errored and need engine/server state.
        raise VLLMServerError("internal server failure")

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

    @app.get("/raise_runtime_error")
    async def raise_runtime_error():
        raise RuntimeError("unexpected server error")

    @app.get("/success")
    async def success():
        return {"status": "ok"}

    return app


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
    "endpoint,expected_status_group,expected_http_code,request_kwargs",
    [
        ("/raise_http_exception_400", "4xx", 400, {}),
        ("/raise_http_exception_404", "4xx", 404, {}),
        ("/raise_request_validation_error", "4xx", 400, {"params": {"n": "x"}}),
        ("/raise_vllm_validation_error", "4xx", 400, {}),
        ("/raise_vllm_not_found_error", "4xx", 404, {}),
        ("/raise_vllm_server_error", "5xx", 500, {}),
        ("/raise_value_error", "4xx", 400, {}),
        ("/raise_type_error", "4xx", 400, {}),
        ("/raise_overflow_error", "4xx", 400, {}),
        ("/raise_not_implemented_error", "5xx", 501, {}),
        ("/raise_runtime_error", "5xx", 500, {}),
        ("/success", "2xx", 200, {}),
    ],
    ids=[
        "HTTPException(400)->4xx",
        "HTTPException(404)->4xx",
        "RequestValidationError->4xx",
        "VLLMValidationError->4xx",
        "VLLMNotFoundError->4xx",
        "VLLMServerError->5xx",
        "ValueError->4xx",
        "TypeError->4xx",
        "OverflowError->4xx",
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
    request_kwargs,
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
        response = await client.get(endpoint, **request_kwargs)

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
