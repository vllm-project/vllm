# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI
from starlette.testclient import TestClient

from vllm.entrypoints.serve.elastic_ep.api_router import (
    attach_router,
)

AUTH_HEADERS = {"Authorization": "Bearer test-key"}


class _StubEngineClient:
    """Mimics the engine client; records scale calls."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.calls: list[tuple[int, int]] = []

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int
    ) -> None:
        self.calls.append((new_data_parallel_size, drain_timeout))
        if not self.enabled:
            raise AssertionError("Only ray DP backend supports scaling elastic EP")


class _StubParallelConfig:
    def __init__(self, enabled: bool = True):
        self.enable_elastic_ep = enabled


class _StubVllmConfig:
    def __init__(self, enabled: bool = True):
        self.parallel_config = _StubParallelConfig(enabled)


def _make_client(elastic_enabled: bool = True) -> TestClient:
    app = FastAPI()
    attach_router(app)
    app.state.engine_client = _StubEngineClient(enabled=elastic_enabled)
    app.state.vllm_config = _StubVllmConfig(enabled=elastic_enabled)
    return TestClient(app, raise_server_exceptions=False)


def test_rejects_boolean_data_parallel_size():
    client = _make_client()
    resp = client.post(
        "/scale_elastic_ep",
        json={"new_data_parallel_size": True},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 400
    assert "positive integer" in resp.json()["detail"]


def test_rejects_boolean_drain_timeout():
    client = _make_client()
    resp = client.post(
        "/scale_elastic_ep",
        json={"new_data_parallel_size": 2, "drain_timeout": True},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 400
    assert "positive integer" in resp.json()["detail"]


def test_rejects_non_positive_and_non_integer():
    client = _make_client()
    for bad in (0, -3, 2.5, "4"):
        resp = client.post(
            "/scale_elastic_ep",
            json={"new_data_parallel_size": bad},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 400, bad


def test_returns_400_when_elastic_ep_disabled():
    client = _make_client(elastic_enabled=False)
    resp = client.post(
        "/scale_elastic_ep",
        json={"new_data_parallel_size": 2},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 400
    assert "enable-elastic-ep" in resp.json()["detail"]


def test_valid_request_still_scales():
    app_client = _make_client()
    resp = app_client.post(
        "/scale_elastic_ep",
        json={"new_data_parallel_size": 2, "drain_timeout": 30},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()["message"] == "Scaled to 2 data parallel engines"
