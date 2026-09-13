# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.profile.api_router import router


@pytest.fixture
def engine_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(engine_client: AsyncMock) -> TestClient:
    app = FastAPI()
    app.state.engine_client = engine_client
    app.include_router(router)
    return TestClient(app)


def test_start_profile_without_body_is_backwards_compatible(
    client: TestClient, engine_client: AsyncMock
) -> None:
    response = client.post("/start_profile")

    assert response.status_code == 200
    engine_client.start_profile.assert_awaited_once_with()


def test_start_profile_forwards_session_overrides(
    client: TestClient, engine_client: AsyncMock
) -> None:
    response = client.post(
        "/start_profile",
        json={
            "profile_prefix": "sharegpt_run-1",
            "delay_iterations": 5000,
            "max_iterations": 20,
        },
    )

    assert response.status_code == 200
    engine_client.start_profile.assert_awaited_once_with("sharegpt_run-1", 5000, 20)


@pytest.mark.parametrize(
    "body",
    [
        {"profile_prefix": "../trace"},
        {"profile_prefix": ""},
        {"delay_iterations": -1},
        {"max_iterations": "20"},
        {"unknown": 1},
    ],
)
def test_start_profile_rejects_invalid_overrides(
    client: TestClient, engine_client: AsyncMock, body: dict
) -> None:
    response = client.post("/start_profile", json=body)

    assert response.status_code == 422
    engine_client.start_profile.assert_not_awaited()
