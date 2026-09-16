# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.cancel.api_router import attach_router
from vllm.entrypoints.serve.middleware.authenticate import (
    AuthenticationMiddleware,
)


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    active_requests = {"cmpl-123", "chatcmpl-456"}

    def fake_has_request(req_id: str) -> bool:
        return req_id in active_requests

    async def fake_abort(req_id_or_ids, internal=False):
        if isinstance(req_id_or_ids, str):
            req_ids = [req_id_or_ids]
        else:
            req_ids = list(req_id_or_ids)

        aborted = []
        for r in req_ids:
            if r in active_requests:
                active_requests.remove(r)
                aborted.append(r)
        return aborted

    engine.has_request = MagicMock(side_effect=fake_has_request)
    engine.abort = AsyncMock(side_effect=fake_abort)
    return engine


@pytest.fixture
def app(mock_engine):
    fastapi_app = FastAPI()
    fastapi_app.state.args = Namespace()
    fastapi_app.state.engine_client = mock_engine
    attach_router(fastapi_app)
    return fastapi_app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_cancel_single_request_success(client, mock_engine):
    resp = client.post("/v1/requests/cmpl-123/cancel")
    assert resp.status_code == HTTPStatus.OK
    data = resp.json()
    assert data["status"] == "cancelled"
    assert data["request_id"] == "cmpl-123"
    mock_engine.abort.assert_awaited_once_with("cmpl-123")


def test_cancel_single_request_delete_method(client, mock_engine):
    resp = client.delete("/v1/requests/chatcmpl-456")
    assert resp.status_code == HTTPStatus.OK
    data = resp.json()
    assert data["status"] == "cancelled"
    assert data["request_id"] == "chatcmpl-456"
    mock_engine.abort.assert_awaited_once_with("chatcmpl-456")


def test_cancel_single_request_not_found(client, mock_engine):
    resp = client.post("/v1/requests/nonexistent-id/cancel")
    assert resp.status_code == HTTPStatus.NOT_FOUND
    data = resp.json()
    assert data["error"]["type"] == "NotFoundError"
    assert "nonexistent-id" in data["error"]["message"]


def test_cancel_batch_requests(client, mock_engine):
    resp = client.post(
        "/v1/requests/cancel",
        json={"request_ids": ["cmpl-123", "chatcmpl-456", "missing"]},
    )
    assert resp.status_code == HTTPStatus.OK
    data = resp.json()
    assert data["status"] == "cancelled"
    assert "cmpl-123" in data["cancelled_request_ids"]
    assert "chatcmpl-456" in data["cancelled_request_ids"]
    assert "missing" not in data["cancelled_request_ids"]


def test_cancel_batch_empty(client, mock_engine):
    resp = client.post(
        "/v1/requests/cancel",
        json={"request_ids": []},
    )
    assert resp.status_code == HTTPStatus.OK
    data = resp.json()
    assert data["status"] == "cancelled"
    assert data["cancelled_request_ids"] == []


def test_abort_requests_with_ids(client, mock_engine):
    resp = client.post(
        "/abort_requests",
        json={"request_ids": ["cmpl-123"]},
    )
    assert resp.status_code == HTTPStatus.OK
    data = resp.json()
    assert data["status"] == "cancelled"
    assert "cmpl-123" in data["cancelled_request_ids"]


def test_abort_requests_empty_body(client, mock_engine):
    resp = client.post("/abort_requests", json={})
    assert resp.status_code == HTTPStatus.OK
    data = resp.json()
    assert data["status"] == "cancelled"


def test_authentication_middleware_protects_cancel_endpoint(mock_engine):
    fastapi_app = FastAPI()
    fastapi_app.state.args = Namespace()
    fastapi_app.state.engine_client = mock_engine
    attach_router(fastapi_app)
    fastapi_app.add_middleware(
        AuthenticationMiddleware,
        tokens=["secret-token-123"],
    )

    auth_client = TestClient(fastapi_app)

    # Without token -> 401 Unauthorized
    unauth_resp = auth_client.post("/v1/requests/cmpl-123/cancel")
    assert unauth_resp.status_code == HTTPStatus.UNAUTHORIZED

    # With valid token -> 200 OK
    auth_resp = auth_client.post(
        "/v1/requests/cmpl-123/cancel",
        headers={"Authorization": "Bearer secret-token-123"},
    )
    assert auth_resp.status_code == HTTPStatus.OK


def test_cancel_single_request_race_condition(mock_engine):
    fastapi_app = FastAPI()
    fastapi_app.state.args = Namespace()
    mock_engine.has_request = MagicMock(return_value=True)
    # Simulate race condition: has_request was True, but abort finds nothing
    mock_engine.abort = AsyncMock(return_value=[])
    fastapi_app.state.engine_client = mock_engine
    attach_router(fastapi_app)

    test_client = TestClient(fastapi_app)
    resp = test_client.post("/v1/requests/race-req/cancel")
    assert resp.status_code == HTTPStatus.NOT_FOUND
    assert "race-req" in resp.json()["error"]["message"]
