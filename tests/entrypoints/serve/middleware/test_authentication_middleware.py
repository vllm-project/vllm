# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
from typing import get_args

import pytest
import regex as re
from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.testclient import TestClient

from vllm.entrypoints.launchers.api_server.routers import register_api_routers
from vllm.entrypoints.serve.middleware.authenticate import (
    PUBLIC_ENDPOINTS,
    AuthenticationMiddleware,
    is_public_path,
)
from vllm.tasks import POOLING_TASKS, SupportedTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_all_http_routes(app: FastAPI) -> list[tuple[str, list[str]]]:
    """Extract all HTTP routes (path, methods) from the FastAPI app."""
    routes = []
    for route in app.routes:
        if isinstance(route, Route):
            methods = list(route.methods or {"GET"})
        elif isinstance(route, Mount):
            # e.g. the /metrics ASGI mount; reachable via any method
            methods = ["GET"]
        else:
            continue
        routes.append((route.path, methods))
    return routes


def generate_test_path(path_template: str) -> str:
    """Replace path parameters (e.g. {response_id}) with 'test'."""
    return re.sub(r"\{[^}]+\}", "test", path_template)


def is_public(path_template: str) -> bool:
    """Whether a route path is exempt from authentication."""
    return any(is_public_path(path_template, p) for p in PUBLIC_ENDPOINTS)


def _create_app_with_mock_routes(routes: list[tuple[str, list[str]]]) -> FastAPI:
    """Create a FastAPI app with AuthenticationMiddleware and mock endpoints.

    The middleware is installed before any mock route is registered and
    only receives a provider for the app's route table — mirroring how
    `init_entrypoints_middleware` wires it and proving that routes
    registered after the middleware are still guarded.
    """
    app = FastAPI()
    app.add_middleware(
        AuthenticationMiddleware,
        tokens=["valid-token"],
        routes_provider=lambda: app.routes,
    )

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
        args,
        app,
        supported_tasks=(task,),
        model_config=MockModelConfig(),
    )

    routes = get_all_http_routes(app)
    return task, routes


# ---------------------------------------------------------------------------
# Tests for auto-discovered routes
# ---------------------------------------------------------------------------


def test_auto_discovered_protected_routes_require_auth(task_routes):
    """Every registered route that is not public enforces authentication.

    The guarded set is derived from the route table, so this covers the
    whole served API surface of each task — including endpoints that the
    old hand-maintained prefix list missed (e.g. /invocations,
    /generative_scoring, /tokenize, /score, /rerank).
    """
    task, routes = task_routes
    app = _create_app_with_mock_routes(routes)
    client = TestClient(app)

    for path_template, methods in routes:
        if is_public(path_template):
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


def test_auto_discovered_public_routes_no_auth(task_routes):
    """Routes covered by PUBLIC_ENDPOINTS stay reachable without a token."""
    task, routes = task_routes
    app = _create_app_with_mock_routes(routes)
    client = TestClient(app)

    for path_template, methods in routes:
        if not is_public(path_template):
            continue

        test_path = generate_test_path(path_template)
        test_method = methods[0] if methods else "GET"

        resp = client.request(test_method, test_path)
        assert resp.status_code == 200, (
            f"[{task}] {test_method} {test_path} should be accessible without token"
        )


# ---------------------------------------------------------------------------
# Regression tests for the guarded-prefix drift (IN-03)
# ---------------------------------------------------------------------------


def test_inference_equivalent_endpoints_require_auth():
    """Endpoints outside the legacy /v1-style prefixes guard inference.

    /invocations (SageMaker), /generative_scoring, /tokenize and the
    non-/v1 pooling variants expose the same capabilities as guarded /v1
    endpoints; the derived guarded set protects them automatically.
    """
    routes = [
        ("/invocations", ["POST"]),
        ("/generative_scoring", ["POST"]),
        ("/tokenize", ["POST"]),
        ("/pooling", ["POST"]),
        ("/score", ["POST"]),
        ("/rerank", ["POST"]),
    ]
    app = _create_app_with_mock_routes(routes)
    client = TestClient(app)

    for path, methods in routes:
        resp = client.request(methods[0], path)
        assert resp.status_code == 401, f"{path} should reject missing token"

        resp = client.request(
            methods[0], path, headers={"Authorization": "Bearer valid-token"}
        )
        assert resp.status_code == 200, f"{path} should accept valid token"


def test_route_registered_after_middleware_is_guarded():
    """A route registered after the middleware still requires auth."""
    app = _create_app_with_mock_routes([])
    client = TestClient(app)

    async def mock_endpoint():
        return JSONResponse({"status": "ok"})

    app.add_api_route(
        "/brand_new_endpoint", mock_endpoint, methods=["GET"], include_in_schema=False
    )

    resp = client.get("/brand_new_endpoint")
    assert resp.status_code == 401, "new routes must be guarded by default"

    resp = client.get(
        "/brand_new_endpoint", headers={"Authorization": "Bearer valid-token"}
    )
    assert resp.status_code == 200


def test_public_endpoint_matching_is_segment_exact():
    """Public matching must not whiten paths that only share a prefix."""
    routes = [("/health", ["GET"]), ("/healthcheck", ["GET"])]
    app = _create_app_with_mock_routes(routes)
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/healthcheck").status_code == 401

    resp = client.get("/healthcheck", headers={"Authorization": "Bearer valid-token"})
    assert resp.status_code == 200


def test_param_only_template_guards_everything():
    """A route template with no static part guards the whole site.

    A template such as "/{path:path}" matches every URL, so deriving no
    prefix from it would leave it unauthenticated; the derivation falls
    back to guarding "/" instead.
    """
    app = _create_app_with_mock_routes([])
    client = TestClient(app)

    async def mock_endpoint():
        return JSONResponse({"status": "ok"})

    app.add_api_route(
        "/{path:path}", mock_endpoint, methods=["GET"], include_in_schema=False
    )

    resp = client.get("/anything/here")
    assert resp.status_code == 401, "catch-all routes must guard every path"

    resp = client.get("/anything/here", headers={"Authorization": "Bearer valid-token"})
    assert resp.status_code == 200
