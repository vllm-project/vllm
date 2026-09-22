# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The /finish_weight_update route and its optional checksum snapshot.

An in-process FastAPI TestClient drives the real router, so the request parsing
and the response shape are covered without a model or a GPU.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.dev.rlhf.api_router import attach_router

pytestmark = pytest.mark.cpu_test

_DIGESTS = {"dp0:pp0:pcp0:tp0:ep0:w": "a"}


@pytest.fixture
def client():
    """A TestClient over the real RLHF router with a mock engine."""
    app = FastAPI()
    app.state.engine_client = AsyncMock()
    attach_router(app)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client, app.state.engine_client


def _finish(client, **body):
    return client.post("/finish_weight_update", json=body or None)


def test_finish_without_the_option_is_unchanged(client):
    """The default response must stay exactly what it was before."""
    test_client, engine = client
    engine.finish_weight_update.return_value = None

    response = _finish(test_client)

    assert response.status_code == 200
    assert response.json() == {"message": "Weight update finished"}
    engine.finish_weight_update.assert_awaited_once_with(None, checksum=False)


def test_finish_returns_the_snapshot_when_requested(client):
    test_client, engine = client
    engine.finish_weight_update.return_value = _DIGESTS

    response = _finish(test_client, checksum=True)

    assert response.status_code == 200
    assert response.json() == {
        "message": "Weight update finished",
        "checksums": _DIGESTS,
    }
    engine.finish_weight_update.assert_awaited_once_with(None, checksum=True)


def test_finish_forwards_a_weight_version_with_the_option(client):
    test_client, engine = client
    engine.finish_weight_update.return_value = _DIGESTS

    response = _finish(test_client, weight_version="step-7", checksum=True)

    assert response.status_code == 200
    engine.finish_weight_update.assert_awaited_once_with("step-7", checksum=True)


def test_checksum_defaults_to_false_when_only_a_version_is_sent(client):
    """A body without the option must not turn hashing on."""
    test_client, engine = client
    engine.finish_weight_update.return_value = None

    response = _finish(test_client, weight_version="step-7")

    assert response.status_code == 200
    assert "checksums" not in response.json()
    engine.finish_weight_update.assert_awaited_once_with("step-7", checksum=False)


def test_no_checksums_key_when_the_engine_reports_none(client):
    """None means "not requested", which is not an empty snapshot."""
    test_client, engine = client
    engine.finish_weight_update.return_value = None

    response = _finish(test_client, checksum=True)

    assert response.status_code == 200
    assert response.json() == {"message": "Weight update finished"}
