# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.dev.runtime_config.api_router import attach_router

pytestmark = pytest.mark.cpu_test


def _make_client() -> tuple[TestClient, AsyncMock]:
    engine = AsyncMock()
    engine.get_runtime_config.return_value = {
        "max_num_running_seqs": 128,
        "max_num_seqs_capacity": 256,
    }
    engine.update_runtime_config.return_value = {
        "max_num_running_seqs": 64,
        "max_num_seqs_capacity": 256,
    }
    app = FastAPI()
    app.state.engine_client = engine
    attach_router(app)
    return TestClient(app), engine


def test_get_runtime_config():
    client, engine = _make_client()

    response = client.get("/v1/runtime_config")

    assert response.status_code == 200
    assert response.json() == {
        "max_num_running_seqs": 128,
        "max_num_seqs_capacity": 256,
    }
    engine.get_runtime_config.assert_awaited_once_with()


def test_update_runtime_config():
    client, engine = _make_client()

    response = client.patch("/v1/runtime_config", json={"max_num_running_seqs": 64})

    assert response.status_code == 200
    assert response.json()["max_num_running_seqs"] == 64
    engine.get_runtime_config.assert_awaited_once_with()
    engine.update_runtime_config.assert_awaited_once_with(64)


def test_update_runtime_config_reports_capacity_error():
    client, engine = _make_client()

    response = client.patch("/v1/runtime_config", json={"max_num_running_seqs": 512})

    assert response.status_code == 400
    assert "startup capacity" in response.json()["detail"]
    engine.update_runtime_config.assert_not_awaited()


def test_update_runtime_config_rejects_unknown_fields():
    client, _ = _make_client()

    response = client.patch(
        "/v1/runtime_config",
        json={"max_num_running_seqs": 64, "gpu_memory_utilization": 0.5},
    )

    assert response.status_code == 422


def test_update_runtime_config_rejects_boolean():
    client, _ = _make_client()

    response = client.patch("/v1/runtime_config", json={"max_num_running_seqs": True})

    assert response.status_code == 422


def test_update_runtime_config_reports_unsupported_scheduler():
    client, engine = _make_client()
    engine.get_runtime_config.return_value = {}

    response = client.patch("/v1/runtime_config", json={"max_num_running_seqs": 64})

    assert response.status_code == 501
    engine.update_runtime_config.assert_not_awaited()
