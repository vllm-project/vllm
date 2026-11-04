# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.dev.logging.api_router import router


@pytest.fixture
def engine_client() -> AsyncMock:
    client = AsyncMock()
    client.set_log_level.return_value = {"level": "DEBUG", "logger": "vllm"}
    return client


@pytest.fixture
def client(engine_client: AsyncMock) -> TestClient:
    app = FastAPI()
    app.state.engine_client = engine_client
    app.include_router(router)
    return TestClient(app)


def test_set_log_level_global(client: TestClient, engine_client: AsyncMock) -> None:
    response = client.post("/set_log_level", params={"level": "DEBUG"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "level": "DEBUG",
        "logger": "vllm",
    }
    engine_client.set_log_level.assert_awaited_once_with("DEBUG", None)


def test_set_log_level_module(client: TestClient, engine_client: AsyncMock) -> None:
    engine_client.set_log_level.return_value = {
        "level": "WARNING",
        "logger": "vllm.v1.worker",
    }
    response = client.post(
        "/set_log_level",
        params={"level": "WARNING", "logger": "vllm.v1.worker"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "level": "WARNING",
        "logger": "vllm.v1.worker",
    }
    engine_client.set_log_level.assert_awaited_once_with("WARNING", "vllm.v1.worker")


def test_set_log_level_rejects_unknown_level(
    client: TestClient, engine_client: AsyncMock
) -> None:
    response = client.post("/set_log_level", params={"level": "nope"})

    assert response.status_code == 400
    assert "invalid log level" in response.json()["detail"]
    engine_client.set_log_level.assert_not_awaited()
