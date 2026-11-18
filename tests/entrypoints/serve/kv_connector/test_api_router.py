# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for `vllm/entrypoints/serve/kv_connector/api_router.py`.

Exercises the HTTP status-code mapping of `POST /v1/kv_connector/rpc` (RFC
#51639) against a fake `EngineClient`:

* 200 with the response bytes when the connector returns `bytes` (including
  empty `b""`);
* 404 when no KV connector is configured;
* 501 when the connector returns `None` (not implemented);
* 500 when the connector handler raises.
"""

from argparse import Namespace
from http import HTTPStatus
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.exception_handling.handlers.exception import (
    exception_handler,
)
from vllm.entrypoints.serve.exception_handling.handlers.http import (
    http_exception_handler,
)
from vllm.entrypoints.serve.kv_connector.api_router import attach_router

URL = "/v1/kv_connector/rpc"


class _FakeEngineClient:
    """Stand-in for `EngineClient` used by the router.

    `result` is either the bytes/None to return, or an `Exception` to
    raise. `has_connector` controls whether a KV connector is "configured"
    (drives the 404 branch).
    """

    def __init__(self, result, has_connector: bool = True):
        self.result = result
        kv_transfer_config = object() if has_connector else None
        self.vllm_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)

    async def invoke_kv_connector(self, payload: bytes):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _build_app(client: _FakeEngineClient) -> FastAPI:
    app = FastAPI()
    attach_router(app)
    app.state.engine_client = client
    app.state.args = Namespace(log_error_stack=False)
    app.exception_handler(HTTPException)(http_exception_handler)
    app.exception_handler(Exception)(exception_handler)
    return app


def _post(client: _FakeEngineClient, body: bytes = b"payload"):
    with TestClient(_build_app(client), raise_server_exceptions=False) as tc:
        return tc.post(URL, content=body)


def test_returns_bytes_as_200():
    resp = _post(_FakeEngineClient(b"echo:payload"))
    assert resp.status_code == HTTPStatus.OK.value
    assert resp.content == b"echo:payload"
    assert resp.headers["content-type"] == "application/octet-stream"


def test_empty_bytes_is_200_ack():
    resp = _post(_FakeEngineClient(b""))
    assert resp.status_code == HTTPStatus.OK.value
    assert resp.content == b""


def test_none_is_501_not_implemented():
    resp = _post(_FakeEngineClient(None))
    assert resp.status_code == HTTPStatus.NOT_IMPLEMENTED.value


def test_no_connector_is_404():
    resp = _post(_FakeEngineClient(b"unused", has_connector=False))
    assert resp.status_code == HTTPStatus.NOT_FOUND.value


def test_handler_exception_is_500():
    resp = _post(_FakeEngineClient(RuntimeError("boom")))
    assert resp.status_code == HTTPStatus.INTERNAL_SERVER_ERROR.value


def test_payload_forwarded_verbatim():
    captured = {}

    class _Capturing(_FakeEngineClient):
        async def invoke_kv_connector(self, payload: bytes):
            captured["payload"] = payload
            return b"ok"

    resp = _post(_Capturing(b"ok"), body=b"\x00\x01\x02raw")
    assert resp.status_code == HTTPStatus.OK.value
    assert captured["payload"] == b"\x00\x01\x02raw"
