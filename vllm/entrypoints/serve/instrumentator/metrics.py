# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import Iterable
from typing import TYPE_CHECKING

import regex as re
from fastapi import FastAPI
from prometheus_client import CollectorRegistry, make_asgi_app
from prometheus_client.core import GaugeMetricFamily
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator import routing as _pfi_routing
from starlette.routing import Match, Mount
from starlette.types import Receive, Scope, Send

from vllm.v1.metrics.prometheus import get_prometheus_registry

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient


def _patch_instrumentator_route_walk() -> None:
    """Make prometheus-fastapi-instrumentator's route walk tolerate routes
    without a ``.path``.

    FastAPI >= 0.137 stores lazy ``_IncludedRouter`` objects in ``app.routes``;
    these are ``BaseRoute`` subclasses with no ``.path`` attribute. The
    instrumentator's ``_get_route_name`` (up to 8.0.0) reads ``route.path``
    unconditionally, so every request raises ``AttributeError`` in the metrics
    middleware and the server returns 500 (e.g. ``/health`` never goes ready).
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


class _EngineHealthCollector:
    def __init__(self, engines: list[dict], errored: bool):
        self.engines = engines
        self.errored = errored

    def collect(self) -> Iterable[GaugeMetricFamily]:
        metric = GaugeMetricFamily(
            "vllm:engine_healthy",
            "Whether the engine reports healthy under fault tolerance.",
            labels=["engine"],
        )
        for engine in self.engines:
            healthy = engine["status"] == "healthy" and not self.errored
            metric.add_metric([str(engine["id"])], int(healthy))
        yield metric


def attach_router(app: FastAPI):
    """Mount prometheus metrics to a FastAPI app."""

    registry = get_prometheus_registry()

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

    async def metrics(scope: Scope, receive: Receive, send: Send):
        scrape_registry = registry
        client: EngineClient | None = getattr(app.state, "engine_client", None)
        if client and client.vllm_config.parallel_config.enable_fault_tolerance:
            status = await client.get_status()
            # Keep rank health local to this scrape, outside shared metric files.
            scrape_registry = CollectorRegistry(auto_describe=True)
            scrape_registry.register(registry)
            scrape_registry.register(
                _EngineHealthCollector(status["engines"], client.errored)
            )
        await make_asgi_app(registry=scrape_registry)(scope, receive, send)

    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", metrics)

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
