# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.cancel.api_router import attach_router
from vllm.entrypoints.serve.cancel.registry import CancellationRegistry
from vllm.entrypoints.serve.middleware.authenticate import (
    AuthenticationMiddleware,
)


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    active_requests = {"cmpl-123", "chatcmpl-456"}

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

    engine.abort = AsyncMock(side_effect=fake_abort)
    return engine


@pytest.fixture
def app(mock_engine):
    fastapi_app = FastAPI()
    fastapi_app.state.args = Namespace()
    fastapi_app.state.cancellation_registry = CancellationRegistry()
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
    mock_engine.abort.assert_awaited_once_with(["cmpl-123"])


def test_cancel_single_request_delete_method(client, mock_engine):
    resp = client.delete("/v1/requests/chatcmpl-456")
    assert resp.status_code == HTTPStatus.OK
    data = resp.json()
    assert data["status"] == "cancelled"
    assert data["request_id"] == "chatcmpl-456"
    mock_engine.abort.assert_awaited_once_with(["chatcmpl-456"])


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


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("post", "/v1/requests/cmpl-123/cancel", None),
        ("delete", "/v1/requests/cmpl-123", None),
        ("post", "/v1/requests/cancel", {"request_ids": ["cmpl-123"]}),
    ],
)
def test_authentication_middleware_protects_cancel_endpoints(
    mock_engine, method, path, body
):
    """Every cancel route sits under /v1 so that --api-key guards it."""
    fastapi_app = FastAPI()
    fastapi_app.state.args = Namespace()
    fastapi_app.state.cancellation_registry = CancellationRegistry()
    fastapi_app.state.engine_client = mock_engine
    attach_router(fastapi_app)
    fastapi_app.add_middleware(
        AuthenticationMiddleware,
        tokens=["secret-token-123"],
    )

    auth_client = TestClient(fastapi_app)

    unauth_resp = auth_client.request(method, path, json=body)
    assert unauth_resp.status_code == HTTPStatus.UNAUTHORIZED

    auth_resp = auth_client.request(
        method,
        path,
        json=body,
        headers={"Authorization": "Bearer secret-token-123"},
    )
    assert auth_resp.status_code == HTTPStatus.OK


def test_cancel_unknown_request_returns_not_found(mock_engine):
    """abort() returning no ids is the only signal the request is gone."""
    fastapi_app = FastAPI()
    fastapi_app.state.args = Namespace()
    fastapi_app.state.cancellation_registry = CancellationRegistry()
    mock_engine.abort = AsyncMock(return_value=[])
    fastapi_app.state.engine_client = mock_engine
    attach_router(fastapi_app)

    test_client = TestClient(fastapi_app)
    resp = test_client.post("/v1/requests/race-req/cancel")
    assert resp.status_code == HTTPStatus.NOT_FOUND
    assert "race-req" in resp.json()["error"]["message"]


def test_registry_resolves_a_fanned_out_request_id():
    """Completions and multi-prompt chat register one engine id per prompt."""
    registry = CancellationRegistry()
    registry.add("cmpl-abc", "cmpl-abc-0")
    registry.add("cmpl-abc", "cmpl-abc-1")

    assert registry.resolve("cmpl-abc") == ["cmpl-abc-0", "cmpl-abc-1"]


def test_registry_passes_through_an_unregistered_id():
    assert CancellationRegistry().resolve("cmpl-abc") == ["cmpl-abc"]


def test_registry_evicts_oldest_entries_beyond_its_bound():
    registry = CancellationRegistry(max_tracked=2)
    for i in range(3):
        registry.add(f"cmpl-{i}", f"cmpl-{i}-0")

    # Evicted ids fall back to being treated as engine ids.
    assert registry.resolve("cmpl-0") == ["cmpl-0"]
    assert registry.resolve("cmpl-2") == ["cmpl-2-0"]


def test_cancel_resolves_derived_ids_before_aborting(mock_engine):
    """The id a client gets back is not the id the engine knows."""
    fastapi_app = FastAPI()
    fastapi_app.state.args = Namespace()
    registry = CancellationRegistry()
    registry.add("cmpl-abc", "cmpl-abc-0")
    registry.add("cmpl-abc", "cmpl-abc-1")
    fastapi_app.state.cancellation_registry = registry
    mock_engine.abort = AsyncMock(return_value=["cmpl-abc-0", "cmpl-abc-1"])
    fastapi_app.state.engine_client = mock_engine
    attach_router(fastapi_app)

    resp = TestClient(fastapi_app).post("/v1/requests/cmpl-abc/cancel")

    assert resp.status_code == HTTPStatus.OK
    mock_engine.abort.assert_awaited_once_with(["cmpl-abc-0", "cmpl-abc-1"])
