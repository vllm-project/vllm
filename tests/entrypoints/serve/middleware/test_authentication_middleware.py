# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from argparse import Namespace
from typing import get_args

import pytest
from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from vllm.entrypoints.openai.api_server import register_api_routers
from vllm.entrypoints.serve.middleware.authenticate import (
    GUARDED_PREFIX,
    AuthenticationMiddleware,
)
from vllm.tasks import SupportedTask

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


# ---------------------------------------------------------------------------
# Explicit endpoint lists (from documentation)
# ---------------------------------------------------------------------------

PROTECTED_ENDPOINTS: list[tuple[str, list[str]]] = [
    ("/v1/models", ["GET"]),
    ("/v1/chat/completions", ["POST"]),
    ("/v1/chat/completions/batch", ["POST"]),
    ("/v1/chat/completions/render", ["POST"]),
    ("/v1/chat/completions/derender", ["POST"]),
    ("/v1/completions", ["POST"]),
    ("/v1/completions/render", ["POST"]),
    ("/v1/completions/derender", ["POST"]),
    ("/v1/embeddings", ["POST"]),
    ("/v1/audio/transcriptions", ["POST"]),
    ("/v1/audio/translations", ["POST"]),
    ("/v1/messages", ["POST"]),
    ("/v1/messages/count_tokens", ["POST"]),
    ("/v1/responses", ["POST"]),
    ("/v1/responses/{response_id}", ["GET"]),
    ("/v1/responses/{response_id}/cancel", ["POST"]),
    ("/v1/score", ["POST"]),
    ("/v1/rerank", ["POST"]),
    ("/v1/load_lora_adapter", ["POST"]),
    ("/v1/unload_lora_adapter", ["POST"]),
    ("/inference/v1/generate", ["POST"]),
    ("/v2/embed", ["POST"]),
    ("/v2/rerank", ["POST"]),
]

UNPROTECTED_ENDPOINTS: list[tuple[str, list[str]]] = [
    ("/invocations", ["POST"]),
    ("/generative_scoring", ["POST"]),
    ("/pooling", ["POST"]),
    ("/classify", ["POST"]),
    ("/score", ["POST"]),
    ("/rerank", ["POST"]),
    ("/pause", ["POST"]),
    ("/resume", ["POST"]),
    ("/is_paused", ["GET"]),
    ("/abort_requests", ["POST"]),
    ("/scale_elastic_ep", ["POST"]),
    ("/is_scaling_elastic_ep", ["GET"]),
    ("/init_weight_transfer_engine", ["POST"]),
    ("/update_weights", ["POST"]),
    ("/get_world_size", ["GET"]),
    ("/tokenize", ["POST"]),
    ("/detokenize", ["POST"]),
    ("/health", ["GET"]),
    ("/ping", ["GET"]),
    ("/version", ["GET"]),
    ("/load", ["GET"]),
    ("/tokenizer_info", ["GET"]),
    ("/server_info", ["GET"]),
    ("/reset_prefix_cache", ["POST"]),
    ("/reset_mm_cache", ["POST"]),
    ("/reset_encoder_cache", ["POST"]),
    ("/sleep", ["POST"]),
    ("/wake_up", ["POST"]),
    ("/is_sleeping", ["GET"]),
    ("/collective_rpc", ["POST"]),
    ("/start_profile", ["POST"]),
    ("/stop_profile", ["POST"]),
]

missing_explicit_endpoints = {
    "/redoc",
    "/weight_info",
    "/start_draft_weight_update",
    "/openapi.json",
    "/update_weight_version",
    "/docs/oauth2-redirect",
    "/docs",
    "/finish_weight_update",
    "/start_weight_update",
    "/metrics",
}
missing_auto_discovered_endpoints = {
    "/score",
    "/v1/load_lora_adapter",
    "/v2/rerank",
    "/start_profile",
    "/rerank",
    "/v1/score",
    "/v2/embed",
    "/stop_profile",
    "/pooling",
    "/v1/unload_lora_adapter",
    "/v1/embeddings",
    "/classify",
    "/tokenizer_info",
    "/v1/rerank",
}

# ---------------------------------------------------------------------------
# Fixture: auto-discovered routes per task
# ---------------------------------------------------------------------------


@pytest.fixture(params=get_args(SupportedTask))
def task_routes(request, monkeypatch) -> tuple[str, list[tuple[str, list[str]]]]:
    """For each supported task, build an app with only that task's routers,
    extract all routes, and return the task name and routes."""
    task = request.param
    # Enable development mode to register all routes (including dev-only routes).
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "1")

    app = FastAPI()
    app.state = Namespace()
    app.state.args = Namespace()

    # Register routers for this specific task (development mode already enabled).
    register_api_routers(app, supported_tasks=(task,))

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
# Tests for explicit endpoint lists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path_template,methods", PROTECTED_ENDPOINTS)
def test_explicit_protected_endpoints_require_auth(path_template, methods):
    app = _create_app_with_mock_routes([(path_template, methods)])
    client = TestClient(app)
    test_path = generate_test_path(path_template)
    test_method = methods[0] if methods else "GET"

    resp = client.request(test_method, test_path)
    assert resp.status_code == 401, (
        f"{test_method} {test_path} should reject missing token"
    )

    resp = client.request(
        test_method, test_path, headers={"Authorization": "Bearer wrong"}
    )
    assert resp.status_code == 401, (
        f"{test_method} {test_path} should reject invalid token"
    )

    resp = client.request(
        test_method, test_path, headers={"Authorization": "Bearer valid-token"}
    )
    assert resp.status_code == 200, (
        f"{test_method} {test_path} should accept valid token"
    )


@pytest.mark.parametrize("path_template,methods", UNPROTECTED_ENDPOINTS)
def test_explicit_unprotected_endpoints_no_auth(path_template, methods):
    app = _create_app_with_mock_routes([(path_template, methods)])
    client = TestClient(app)
    test_path = generate_test_path(path_template)
    test_method = methods[0] if methods else "GET"

    resp = client.request(test_method, test_path)
    assert resp.status_code == 200, (
        f"{test_method} {test_path} should be accessible without token"
    )


# ---------------------------------------------------------------------------
# Coverage comparison: explicit vs. auto-discovered (across all tasks)
# ---------------------------------------------------------------------------


def _collect_all_auto_routes(monkeypatch) -> set[str]:
    """Collect all auto-discovered route paths across all SupportedTasks."""
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "1")
    all_paths: set[str] = set()
    for task in get_args(SupportedTask):
        app = FastAPI()
        app.state = Namespace()
        app.state.args = Namespace()
        register_api_routers(app, supported_tasks=(task,))
        routes = get_all_http_routes(app)
        all_paths.update({path for path, _ in routes})
    return all_paths


def test_no_missing_explicit_endpoints(monkeypatch):
    """Verify that every explicitly documented endpoint is registered by at
    least one SupportedTask under development mode."""
    auto_paths = _collect_all_auto_routes(monkeypatch)
    documented = {path for path, _ in PROTECTED_ENDPOINTS + UNPROTECTED_ENDPOINTS}

    missing = documented - auto_paths - missing_auto_discovered_endpoints
    assert not missing, (
        "The following documented endpoints were not registered by any task: "
        f"{sorted(missing)}"
    )


def test_report_extra_auto_discovered_endpoints(monkeypatch):
    """
    Report (and fail if any) auto-discovered endpoints that are not present
    in the explicit lists. This helps keep documentation up-to-date.
    """
    auto_paths = _collect_all_auto_routes(monkeypatch)
    documented = {path for path, _ in PROTECTED_ENDPOINTS + UNPROTECTED_ENDPOINTS}

    extra = auto_paths - documented - missing_explicit_endpoints
    if extra:
        print("\nExtra auto-discovered endpoints not in explicit lists:")
        for path in sorted(extra):
            print(f"  {path}")

    # Make the test fail if there are extra endpoints, forcing documentation update.
    assert not extra, (
        f"Auto-discovered routes not covered by explicit lists: {sorted(extra)}"
    )
