# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import prometheus_client
from fastapi import FastAPI, Request, Response
from prometheus_client.openmetrics import exposition as om_exposition
from prometheus_fastapi_instrumentator import Instrumentator

from vllm.v1.metrics.prometheus import get_prometheus_registry


async def metrics_handler(request: Request) -> Response:
    """Custom metrics handler that supports OpenMetrics format with exemplars."""
    # Check if exemplars are enabled - if so, always use OpenMetrics format
    # Exemplars are only supported in OpenMetrics format
    exemplars_enabled = False
    try:
        # Check app state for vllm_config
        if hasattr(request.app.state, "vllm_config"):
            vllm_config = request.app.state.vllm_config
            exemplars_enabled = vllm_config.observability_config.enable_exemplars
    except Exception:
        pass

    # Get registry with exemplars_enabled flag to disable multiprocess mode if needed
    registry = get_prometheus_registry(exemplars_enabled=exemplars_enabled)

    # Check Accept header for OpenMetrics format
    accept_header = request.headers.get("accept", "")
    use_openmetrics = (
        exemplars_enabled
        or "application/openmetrics-text" in accept_header
        or "openmetrics" in accept_header.lower()
    )

    if use_openmetrics:
        # Use OpenMetrics format which supports exemplars
        output = om_exposition.generate_latest(registry)
        # generate_latest returns bytes, Response accepts bytes
        return Response(
            content=output,
            media_type="application/openmetrics-text; version=1.0.0; charset=utf-8",
        )
    else:
        # Fall back to standard Prometheus text format
        output = prometheus_client.generate_latest(registry)
        return Response(
            content=output,
            media_type=prometheus_client.CONTENT_TYPE_LATEST,
        )


def attach_router(app: FastAPI):
    """Mount prometheus metrics to a FastAPI app."""

    registry = get_prometheus_registry()

    # Instrument the app to track HTTP request metrics (latency, count, etc.)
    # We don't call .expose() because we use a custom /metrics handler below
    # that supports OpenMetrics format with exemplars.
    Instrumentator(
        excluded_handlers=[
            "/metrics",
            "/health",
            "/load",
            "/ping",
            "/version",
            "/server_info",
        ],
        registry=registry,
    ).add().instrument(app)

    # Custom metrics endpoint that supports OpenMetrics format with exemplars
    @app.get("/metrics")
    async def metrics_endpoint(request: Request) -> Response:
        return await metrics_handler(request)
