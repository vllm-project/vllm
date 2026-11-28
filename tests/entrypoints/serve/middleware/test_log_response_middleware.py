# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI
from starlette.testclient import TestClient

from vllm.entrypoints.common.cli_args import FrontendArgs
from vllm.entrypoints.serve.middleware.register import init_entrypoints_middleware


def _make_client(*, debug_log_api_server_response: bool = False) -> TestClient:
    args = FrontendArgs()
    args.debug_log_api_server_response = debug_log_api_server_response

    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "test-model"}]}

    init_entrypoints_middleware(args, app, ())
    return TestClient(app)


def test_response_logging_disabled_by_default(caplog):
    client = _make_client()

    with caplog.at_level("INFO"):
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert "response_body=" not in caplog.text


def test_flag_logs_response_body(caplog):
    client = _make_client(debug_log_api_server_response=True)

    with caplog.at_level("INFO"):
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert "response_body=" in caplog.text
    assert "test-model" in caplog.text


def test_deprecated_env_var_still_logs(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_DEBUG_LOG_API_SERVER_RESPONSE", "true")
    client = _make_client()

    with caplog.at_level("INFO"):
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert "response_body=" in caplog.text


def test_flag_overrides_disabling_env_var(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_DEBUG_LOG_API_SERVER_RESPONSE", "false")
    client = _make_client(debug_log_api_server_response=True)

    with caplog.at_level("INFO"):
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert "response_body=" in caplog.text
