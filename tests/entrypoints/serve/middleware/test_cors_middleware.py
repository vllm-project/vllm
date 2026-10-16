# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI
from starlette.testclient import TestClient

from vllm.entrypoints.launchers.cli_args import FrontendArgs
from vllm.entrypoints.serve.middleware.register import init_entrypoints_middleware


def _make_client(allowed_origins: list[str] | None = None) -> TestClient:
    args = FrontendArgs()
    if allowed_origins is not None:
        args.allowed_origins = allowed_origins

    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "private-model"}]}

    @app.post("/v1/chat/completions")
    async def chat():
        return {"message": {"content": "ok"}}

    init_entrypoints_middleware(args, app, ())
    return TestClient(app)


def test_default_origins_prevent_cross_origin_reads_and_json_requests():
    client = _make_client()
    response = client.get("/v1/models", headers={"Origin": "http://attacker.example"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers

    preflight = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "http://attacker.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status_code == 400
    assert "access-control-allow-origin" not in preflight.headers


def test_explicit_origin_allows_browser_requests():
    client = _make_client(["http://trusted.example"])
    response = client.get("/v1/models", headers={"Origin": "http://trusted.example"})
    assert response.headers["access-control-allow-origin"] == "http://trusted.example"

    preflight = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "http://trusted.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://trusted.example"
