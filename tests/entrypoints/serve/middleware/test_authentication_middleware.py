# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
from typing import get_args

import pytest
import regex as re
from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from vllm.entrypoints.launchers.api_server.routers import register_api_routers
from vllm.entrypoints.serve.middleware.authenticate import (
    GUARDED_PREFIX,
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
    """For every auto-discovered route that starts with a guarded prefix,
    verify that authentication is enforced."""
    task, routes = task_routes
    app = _create_app_with_mock_routes(routes)
    client = TestClient(app)

    for path_template, methods in routes:
        if not path_template.startswith(GUARDED_PREFIX):
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
    """For every auto-discovered route that does NOT start with a guarded
    prefix, verify that no authentication is required."""
    task, routes = task_routes
    app = _create_app_with_mock_routes(routes)
    client = TestClient(app)

    for path_template, methods in routes:
        if path_template.startswith(GUARDED_PREFIX):
            continue

        test_path = generate_test_path(path_template)
        test_method = methods[0] if methods else "GET"

        resp = client.request(test_method, test_path)
        assert resp.status_code == 200, (
            f"[{task}] {test_method} {test_path} should be accessible without token"
        )


# ---------------------------------------------------------------------------
# Regression tests: root_path must not bypass auth
# ---------------------------------------------------------------------------

GUARDED_ROUTES: list[tuple[str, list[str]]] = [
    ("/v1/models", ["GET"]),
    ("/v1/chat/completions", ["POST"]),
    ("/v2/embed", ["POST"]),
    ("/inference/v1/generate", ["POST"]),
    ("/cohere/v2/chat", ["POST"]),
    ("/health", ["GET"]),
]


def _create_app_with_root_path(
    root_path: str,
    routes: list[tuple[str, list[str]]] | None = None,
) -> FastAPI:
    """Create a FastAPI app with AuthenticationMiddleware and a root_path."""
    app = FastAPI()
    app.add_middleware(AuthenticationMiddleware, tokens=["valid-token"])
    app.root_path = root_path

    async def mock_endpoint():
        return JSONResponse({"status": "ok"})

    for path_template, methods in routes or GUARDED_ROUTES:
        allowed_methods = list(set(methods + ["OPTIONS"]))
        app.add_api_route(
            path_template,
            mock_endpoint,
            methods=allowed_methods,
            include_in_schema=False,
        )
    return app


@pytest.mark.parametrize(
    "root_path, request_path",
    [
        ("/v", "/v1/models"),
        ("/v", "/v1/chat/completions"),
        ("/v", "/v2/embed"),
        ("/in", "/inference/v1/generate"),
        ("/co", "/cohere/v2/chat"),
        ("/", "/v1/models"),
    ],
    ids=[
        "root=/v,path=/v1/models",
        "root=/v,path=/v1/chat/completions",
        "root=/v,path=/v2/embed",
        "root=/in,path=/inference/v1/generate",
        "root=/co,path=/cohere/v2/chat",
        "root=/,path=/v1/models",
    ],
)
def test_root_path_non_segment_prefix_does_not_bypass_auth(
    root_path: str, request_path: str
):
    """When root_path is a non-segment prefix of a guarded route, the auth
    check must still fire. This is a regression test for the removeprefix
    bypass where '/v1/models'.removeprefix('/v') yielded '1/models' which
    did not match GUARDED_PREFIX."""
    app = _create_app_with_root_path(root_path)
    client = TestClient(app, root_path=root_path)

    resp = client.request("GET" if "models" in request_path else "POST", request_path)
    assert resp.status_code == 401, (
        f"root_path={root_path!r} request={request_path!r} "
        "should be rejected without a token"
    )

    resp = client.request(
        "GET" if "models" in request_path else "POST",
        request_path,
        headers={"Authorization": "Bearer valid-token"},
    )
    assert resp.status_code == 200, (
        f"root_path={root_path!r} request={request_path!r} "
        "should succeed with a valid token"
    )


@pytest.mark.parametrize(
    "root_path, request_path",
    [
        ("/api", "/api/v1/models"),
        ("/proxy", "/proxy/v1/chat/completions"),
    ],
    ids=[
        "root=/api,path=/api/v1/models",
        "root=/proxy,path=/proxy/v1/chat/completions",
    ],
)
def test_root_path_segment_boundary_still_guards(root_path: str, request_path: str):
    """When root_path falls on a real segment boundary, stripping it must
    still leave a guarded path that requires auth."""
    routes = [
        ("/v1/models", ["GET"]),
        ("/v1/chat/completions", ["POST"]),
    ]
    app = _create_app_with_root_path(root_path, routes)
    client = TestClient(app, root_path=root_path)

    resp = client.request("GET" if "models" in request_path else "POST", request_path)
    assert resp.status_code == 401, (
        f"root_path={root_path!r} request={request_path!r} "
        "should be rejected without a token"
    )

    resp = client.request(
        "GET" if "models" in request_path else "POST",
        request_path,
        headers={"Authorization": "Bearer valid-token"},
    )
    assert resp.status_code == 200, (
        f"root_path={root_path!r} request={request_path!r} "
        "should succeed with a valid token"
    )


def test_root_path_unguarded_route_still_accessible():
    """Unguarded routes (e.g. /health) must remain accessible without a
    token regardless of root_path."""
    app = _create_app_with_root_path("/v")
    client = TestClient(app, root_path="/v")

    resp = client.get("/health")
    assert resp.status_code == 200, "/health should not require auth"
