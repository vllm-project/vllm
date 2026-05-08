# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression tests for ``AuthenticationMiddleware``.

These tests check the *path-matching* behaviour of the middleware in
isolation. They do not exercise an actual model and do not depend on a
running engine, which keeps them fast and safe to run in CI.

The motivating issue is that the disaggregated-serving generate route is
mounted at ``/inference/v1/generate`` and performs full text generation.
Before the fix, the middleware only required auth for paths starting
with ``/v1``, so this versioned inference endpoint was reachable
without an API key whenever ``--api-key`` was set.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.server_utils import AuthenticationMiddleware


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    @app.post("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions():
        return {"ok": True}

    @app.post("/inference/v1/generate")
    async def generate():
        return {"ok": True}

    @app.get("/tokenize")
    async def tokenize():
        return {"ok": True}

    return app


def _make_client(token: str = "secret") -> TestClient:
    app = _build_app()
    app.add_middleware(AuthenticationMiddleware, tokens=[token])
    return TestClient(app)


def test_health_does_not_require_auth():
    client = _make_client()
    resp = client.get("/health")
    assert resp.status_code == 200


def test_v1_completions_requires_auth():
    client = _make_client()
    resp = client.post("/v1/completions")
    assert resp.status_code == 401


def test_v1_completions_with_correct_token_succeeds():
    client = _make_client(token="secret")
    resp = client.post(
        "/v1/completions",
        headers={"Authorization": "Bearer secret"},
    )
    assert resp.status_code == 200


def test_v1_completions_with_wrong_token_is_rejected():
    client = _make_client(token="secret")
    resp = client.post(
        "/v1/completions",
        headers={"Authorization": "Bearer not-the-token"},
    )
    assert resp.status_code == 401


def test_inference_v1_generate_requires_auth():
    """Regression test: ``/inference/v1/generate`` performs full text
    generation and must be authenticated even though its path does not
    *start* with ``/v1``."""
    client = _make_client()
    resp = client.post("/inference/v1/generate")
    assert resp.status_code == 401, (
        "/inference/v1/generate must require an API key when --api-key is set; "
        "previously it was reachable unauthenticated because the middleware "
        "only checked startswith('/v1')."
    )


def test_inference_v1_generate_with_correct_token_succeeds():
    client = _make_client(token="secret")
    resp = client.post(
        "/inference/v1/generate",
        headers={"Authorization": "Bearer secret"},
    )
    assert resp.status_code == 200


def test_options_request_skips_auth():
    """CORS preflights must always pass through."""
    client = _make_client()
    resp = client.options("/v1/completions")
    # FastAPI's default may 405 OPTIONS on a POST-only route; what matters
    # here is the middleware did not short-circuit with 401.
    assert resp.status_code != 401


def test_unrelated_path_does_not_require_auth():
    """Paths outside the auth-required prefixes (e.g. ``/tokenize``) keep
    their existing behaviour and remain unauthenticated. This matches the
    project's stance that non-versioned helper endpoints are not in scope
    for the API-key gate."""
    client = _make_client()
    resp = client.get("/tokenize")
    assert resp.status_code == 200
