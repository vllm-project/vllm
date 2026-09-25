# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.entrypoints.serve.tokenize.api_router import attach_router


def _make_app():
    app = FastAPI()
    init_exception_handler(app)
    app.state.args = Namespace(log_error_stack=False)
    # Diffusion-only engines attach the tokenize router without wiring a handler.
    app.state.serving_tokenization = None
    attach_router(app)
    return app


def test_tokenize_without_handler_returns_501_not_500():
    """Regression: a None tokenization handler must return a controlled 501,
    not raise AttributeError -> HTTP 500 (vllm-omni #8139)."""
    app = _make_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.post("/tokenize", json={"model": "m", "prompt": "hi"})
    assert r.status_code == HTTPStatus.NOT_IMPLEMENTED
    body = r.json()
    assert body["error"]["code"] == HTTPStatus.NOT_IMPLEMENTED
    assert "tokenization" in body["error"]["message"].lower()


def test_detokenize_without_handler_returns_501_not_500():
    app = _make_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.post("/detokenize", json={"model": "m", "tokens": [1, 2, 3]})
    assert r.status_code == HTTPStatus.NOT_IMPLEMENTED
    assert r.json()["error"]["code"] == HTTPStatus.NOT_IMPLEMENTED
