# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

import prometheus_client
import regex as re
from fastapi import FastAPI, Request, Response
from prometheus_client import generate_latest, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator import routing as _pfi_routing
from starlette.responses import PlainTextResponse
from starlette.routing import Match, Mount
from starlette.types import Scope

from vllm.logger import init_logger
from vllm.v1.metrics.prometheus import (
    MultiprocessMetricsReadError,
    get_prometheus_registry,
)

logger = init_logger(__name__)


def _patch_instrumentator_route_walk() -> None:
    """Make prometheus-fastapi-instrumentator's route walk tolerate routes
    without a `.path`.

    FastAPI >= 0.137 stores lazy `_IncludedRouter` objects in `app.routes`;
    these are `BaseRoute` subclasses with no `.path` attribute. The
    instrumentator's `_get_route_name` (up to 8.0.0) reads `route.path`
    unconditionally, so every request raises `AttributeError` in the metrics
    middleware and the server returns 500 (e.g. `/health` never goes ready).
    Skip path-less routes; this only affects the metric handler label, not
    request routing. Idempotent.
    """

    def _get_route_name(scope: Scope, routes, route_name=None):
        for route in routes:
            if getattr(route, "path", None) is None:
                continue
            match, child_scope = route.matches(scope)
            if match == Match.FULL:
                route_name = route.path
                child_scope = {**scope, **child_scope}
                if isinstance(route, Mount) and route.routes:
                    child = _get_route_name(child_scope, route.routes, route_name)
                    route_name = None if child is None else route_name + child
                return route_name
            elif match == Match.PARTIAL and route_name is None:
                route_name = route.path
        return None

    _pfi_routing._get_route_name = _get_route_name


_patch_instrumentator_route_walk()


class PrometheusResponse(Response):
    media_type = prometheus_client.CONTENT_TYPE_LATEST


def attach_router(app: FastAPI):
    """Mount prometheus metrics to a FastAPI app."""
    registry = get_prometheus_registry()

    # `response_class=PrometheusResponse` is needed to return an HTTP response
    # with header "Content-Type: text/plain; version=0.0.4; charset=utf-8"
    # instead of the default "application/json" which is incorrect.
    # See https://github.com/trallnag/prometheus-fastapi-instrumentator/issues/163#issue-1296092364
    instrumentator = (
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
        )
        .add()
        .instrument(app)
    )

    if registry is prometheus_client.REGISTRY:
        instrumentator.expose(app, response_class=PrometheusResponse)
    else:
        # Instrumentator.expose builds its own multiprocess registry on every
        # request. Use vLLM's registry so both routes get the same diagnostics.
        @app.get("/metrics", response_class=PrometheusResponse)
        def metrics() -> Response:
            return PrometheusResponse(generate_latest(registry))

        @app.exception_handler(MultiprocessMetricsReadError)
        async def multiprocess_metrics_error(
            _request: Request, exc: MultiprocessMetricsReadError
        ) -> Response:
            logger.error(
                "Prometheus metrics scrape failed while reading multiprocess files.",
                exc_info=exc,
            )
            return PlainTextResponse(
                "Prometheus multiprocess metrics are unavailable. Check that "
                "PROMETHEUS_MULTIPROC_DIR is writable, local, and cleared "
                "between vLLM runs.\n",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            )

    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
