# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
from typing import get_args

import pytest
import regex as re
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from vllm.entrypoints.launchers.api_server.routers import register_api_routers
from vllm.entrypoints.serve.middleware.authenticate import (
    UNGUARDED_PATHS,
    AuthenticationMiddleware,
)
from vllm.tasks import POOLING_TASKS, SupportedTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_all_http_routes(app: FastAPI) -> list[tuple[str, list[str]]]:
    """Extract all HTTP routes (path, methods) from the FastAPI app."""
    routes = []
    for route in app.routes:
        if not isinstance(route, Route):
            continue
        path = route.path
        methods = list(route.methods or {"GET"})
        routes.append((path, methods))
    return routes


def generate_test_path(path_template: str) -> str:
    """Replace path parameters (e.g. {response_id}) with 'test'."""
    return re.sub(r"\{[^}]+\}", "test", path_template)


def _create_app_with_mock_routes(routes: list[tuple[str, list[str]]]) -> FastAPI:
    """Create a FastAPI app with AuthenticationMiddleware and mock endpoints."""
    app = FastAPI()
    app.add_middleware(AuthenticationMiddleware, tokens=["valid-token"])

    async def mock_endpoint():
        return JSONResponse({"status": "ok"})

    for path_template, methods in routes:
        allowed_methods = list(set(methods + ["OPTIONS"]))
        app.add_api_route(
            path_template,
            mock_endpoint,
            methods=allowed_methods,
            include_in_schema=False,
        )
    return app


class MockModelConfig:
    def __init__(self):
        self.hf_config = Namespace()
        self.hf_config.num_labels = 1

    def get_pooling_task(self, supported_tasks: tuple["SupportedTask", ...]):
        pooling_tasks = [s for s in supported_tasks if s in POOLING_TASKS]
        return pooling_tasks[0] if len(pooling_tasks) > 0 else None


@pytest.fixture(params=get_args(SupportedTask))
def task_routes(request, monkeypatch) -> tuple[str, list[tuple[str, list[str]]]]:
    """For each supported task, build an app with only that task's routers,
    extract all routes, and return the task name and routes."""
    task = request.param
    # Enable development mode to register all routes (including dev-only routes).
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "1")

    app = FastAPI()
    args = Namespace()
    app.state = Namespace()
    app.state.args = args

    # Register routers for this specific task (development mode already enabled).
    register_api_routers(
        args, app, supported_tasks=(task,), model_config=MockModelConfig()
    )

    routes = get_all_http_routes(app)
    return task, routes


# ---------------------------------------------------------------------------
# Tests for auto-discovered routes
# ---------------------------------------------------------------------------


def test_auto_discovered_protected_routes_require_auth(task_routes):
    """For every auto-discovered route that is not in the liveness allowlist,
    verify that authentication is enforced."""
    task, routes = task_routes
    app = _create_app_with_mock_routes(routes)
    client = TestClient(app)

    for path_template, methods in routes:
        if path_template in UNGUARDED_PATHS:
            continue

        test_path = generate_test_path(path_template)
        test_method = methods[0] if methods else "GET"

        resp = client.request(test_method, test_path)
        assert resp.status_code == 401, (
            f"[{task}] {test_method} {test_path} should reject missing token"
        )

        resp = client.request(
            test_method, test_path, headers={"Authorization": "Bearer wrong"}
        )
        assert resp.status_code == 401, (
            f"[{task}] {test_method} {test_path} should reject invalid token"
        )

        resp = client.request(
            test_method, test_path, headers={"Authorization": "Bearer valid-token"}
        )
        assert resp.status_code == 200, (
            f"[{task}] {test_method} {test_path} should accept valid token"
        )


def test_auto_discovered_unprotected_routes_no_auth(task_routes):
    """For every auto-discovered route in the liveness allowlist, verify that
    no authentication is required."""
    task, routes = task_routes
    app = _create_app_with_mock_routes(routes)
    client = TestClient(app)

    for path_template, methods in routes:
        if path_template not in UNGUARDED_PATHS:
            continue

        test_path = generate_test_path(path_template)
        test_method = methods[0] if methods else "GET"

        resp = client.request(test_method, test_path)
        assert resp.status_code == 200, (
            f"[{task}] {test_method} {test_path} should be accessible without token"
        )


# ---------------------------------------------------------------------------
# Trailing slashes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(UNGUARDED_PATHS))
def test_unguarded_path_with_trailing_slash_needs_no_token(path):
    """A liveness probe configured with a trailing slash must still answer.

    The middleware runs ahead of the router, so "/health/" never reaches
    FastAPI's redirect_slashes: an exact-match allowlist test rejects it with
    401 before routing ever happens.
    """
    routes = [(path, ["GET"]), (path + "/", ["GET"])]
    client = TestClient(_create_app_with_mock_routes(routes))

    assert client.get(path + "/").status_code == 200


def test_trailing_slash_does_not_unguard_a_protected_path():
    """Normalizing the trailing slash must not widen the allowlist."""
    routes = [("/v1/models", ["GET"]), ("/v1/models/", ["GET"])]
    client = TestClient(_create_app_with_mock_routes(routes))

    assert client.get("/v1/models/").status_code == 401

    headers = {"Authorization": "Bearer valid-token"}
    assert client.get("/v1/models/", headers=headers).status_code == 200


# ---------------------------------------------------------------------------
# OPTIONS and CORS preflights
# ---------------------------------------------------------------------------

METRICS_BODY = "vllm:num_requests_running 0\n"


async def _any_method_metrics_app(scope, receive, send):
    """Stand-in for the mounted Prometheus app: it answers any method."""
    await PlainTextResponse(METRICS_BODY)(scope, receive, send)


def _create_app_with_cors_and_metrics_mount() -> FastAPI:
    """The middleware order of the vLLM server: CORSMiddleware is added first,
    so AuthenticationMiddleware runs before it."""
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(AuthenticationMiddleware, tokens=["valid-token"])
    app.mount("/metrics", _any_method_metrics_app)
    return app


def test_bare_options_on_a_guarded_mount_needs_a_token():
    """A bare OPTIONS request is not a preflight, so it must not skip the
    token check. A mount such as /metrics answers any method."""
    client = TestClient(_create_app_with_cors_and_metrics_mount())

    resp = client.options("/metrics")
    assert resp.status_code == 401
    assert METRICS_BODY not in resp.text

    headers = {"Authorization": "Bearer valid-token"}
    assert client.options("/metrics", headers=headers).text == METRICS_BODY


def test_options_without_origin_is_not_a_preflight():
    """Without Origin, CORSMiddleware sends the request through to the app,
    so the request-method header alone must not skip the token check."""
    client = TestClient(_create_app_with_cors_and_metrics_mount())

    resp = client.options("/metrics", headers={"Access-Control-Request-Method": "GET"})
    assert resp.status_code == 401
    assert METRICS_BODY not in resp.text


def test_cors_preflight_needs_no_token_and_returns_no_data():
    """Browsers send a preflight without credentials. CORSMiddleware answers
    it, and the request does not get to the mount."""
    client = TestClient(_create_app_with_cors_and_metrics_mount())

    resp = client.options(
        "/metrics",
        headers={
            "Origin": "http://dashboard.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers
    assert METRICS_BODY not in resp.text
