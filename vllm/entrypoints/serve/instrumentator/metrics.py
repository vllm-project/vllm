# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import time

import prometheus_client
import regex as re
from fastapi import FastAPI, Response
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator import routing as _pfi_routing
from starlette.routing import Match, Mount
from starlette.types import Scope

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.core.sched.forecast import curve_value
from vllm.v1.metrics.prometheus import get_prometheus_registry

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


class _RefreshBeforeMetrics:
    """ASGI middleware: run `refresh` before a `/metrics` request is served.

    Scrapes that arrive within `min_interval` seconds of the last refresh reuse
    its values, so any number of scrapers costs the engine one refresh per
    interval. A scrape waits at most `timeout` seconds for the engine. Past
    that it is served the values of the last refresh, the refresh in flight is
    left to finish, and no other one starts until it has. `set_age` is given
    the age of the served values at every scrape.
    """

    def __init__(
        self,
        app,
        refresh,
        set_age=None,
        min_interval: float = 0.02,
        timeout: float = 0.1,
    ) -> None:
        self.app = app
        self.refresh = refresh
        self.set_age = set_age
        self.min_interval = min_interval
        self.timeout = timeout
        self._last = float("-inf")
        self._refreshed_at: float | None = None
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        try:
            if await self.refresh():
                self._refreshed_at = time.monotonic()
        except Exception:
            logger.debug("TTFT forecast refresh failed", exc_info=True)
        finally:
            self._task = None

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and scope["path"].rstrip("/") == "/metrics":
            task = self._task
            if task is None and time.monotonic() - self._last >= self.min_interval:
                self._last = time.monotonic()
                task = self._task = asyncio.ensure_future(self._run())
            if task is not None:
                await asyncio.wait({task}, timeout=self.timeout)
            if self.set_age is not None and self._refreshed_at is not None:
                self.set_age(time.monotonic() - self._refreshed_at)
        await self.app(scope, receive, send)


def forecast_new_token_knots(max_len: int, token_budget: int) -> list[int]:
    """New-token values the forecast gauges are published at.

    Powers of two up to the model length, plus both sides of the first eight
    step boundaries: a prompt one token over the per-step token budget takes
    a whole extra step, and interpolating across that jump would hide it.
    """
    knots = {1 << k for k in range(4, max_len.bit_length())} | {max_len}
    if token_budget > 0:
        for k in range(1, 9):
            knots.update((k * token_budget, k * token_budget + 1))
    return sorted(n for n in knots if 0 < n <= max_len)


def _add_ttft_forecast_gauges(app: FastAPI) -> None:
    """Publish the engine's TTFT forecast as gauges, refreshed at every scrape.

    One series per (new tokens, cached tokens) knot: new tokens at
    `forecast_new_token_knots`, cached tokens at 0, a quarter and half of
    the model length. A consumer interpolates between knots.
    `vllm:ttft_forecast_idle_seconds` is the same request on this engine with
    nothing queued.
    """
    labelnames = ["new_tokens", "cached_tokens"]
    gauge_forecast = prometheus_client.Gauge(
        name="vllm:ttft_forecast_seconds",
        documentation="Forecast seconds to first token for a request arriving now.",
        multiprocess_mode="mostrecent",
        labelnames=labelnames,
    )
    gauge_idle = prometheus_client.Gauge(
        name="vllm:ttft_forecast_idle_seconds",
        documentation="Forecast seconds to first token with nothing queued.",
        multiprocess_mode="mostrecent",
        labelnames=labelnames,
    )

    gauge_restore = prometheus_client.Gauge(
        name="vllm:ttft_forecast_restore_seconds",
        documentation=(
            "Seconds a request waits for a KV connector to load its prefix "
            "before it can be scheduled. Added to the forecast of a request "
            "whose prefix is held in another tier."
        ),
        multiprocess_mode="mostrecent",
    )
    gauge_busy = prometheus_client.Gauge(
        name="vllm:ttft_forecast_busy_share",
        documentation=(
            "Share of recent time the engine had a step in flight. A router "
            "uses it to price the delay a request causes to later arrivals."
        ),
        multiprocess_mode="mostrecent",
    )

    gauge_age = prometheus_client.Gauge(
        name="vllm:ttft_forecast_age_seconds",
        documentation=(
            "Seconds since the forecast gauges were computed. A scrape does not "
            "wait for an engine that is slow to answer: it is served the last "
            "values, and this gauge grows until the engine answers again."
        ),
        multiprocess_mode="mostrecent",
    )

    async def refresh() -> bool:
        client = getattr(app.state, "engine_client", None)
        if client is None:
            return False
        max_len = client.model_config.max_model_len
        cached_knots = [0, max_len // 4, max_len // 2]
        result = await client.forecast_ttft(cached_knots, None)
        if not result.get("ready"):
            return False
        gauge_restore.set(float(result.get("restore_seconds") or 0.0))
        gauge_busy.set(float(result.get("busy_share") or 0.0))
        new_knots = forecast_new_token_knots(
            max_len, int(result.get("token_budget") or 0)
        )
        for cached in cached_knots:
            curve = result["curves"][str(cached)]
            idle = result["floor"][str(cached)]
            for new in new_knots:
                if new + cached > max_len:
                    break
                seconds = curve_value(curve, new)
                idle_seconds = curve_value(idle, new)
                if seconds is None or idle_seconds is None:
                    break
                gauge_forecast.labels(str(new), str(cached)).set(seconds)
                gauge_idle.labels(str(new), str(cached)).set(idle_seconds)
        return True

    app.add_middleware(_RefreshBeforeMetrics, refresh=refresh, set_age=gauge_age.set)


def attach_router(app: FastAPI):
    """Mount prometheus metrics to a FastAPI app."""
    registry = get_prometheus_registry()

    # `response_class=PrometheusResponse` is needed to return an HTTP response
    # with header "Content-Type: text/plain; version=0.0.4; charset=utf-8"
    # instead of the default "application/json" which is incorrect.
    # See https://github.com/trallnag/prometheus-fastapi-instrumentator/issues/163#issue-1296092364
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
    ).add().instrument(app).expose(app, response_class=PrometheusResponse)

    if envs.VLLM_TTFT_FORECAST:
        _add_ttft_forecast_gauges(app)

    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
