# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import vllm.envs as envs
from vllm.entrypoints.serve.sleep.api_router import attach_router


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_SERVER_DEV_MODE", True)
    engine = MagicMock()
    engine.collective_rpc = AsyncMock(return_value=None)

    app = FastAPI()
    app.state.engine_client = engine
    attach_router(app)
    return TestClient(app), engine


def test_sleep_ep_ranks_tags_validates_and_dispatches(client):
    test_client, engine = client

    resp = test_client.post(
        "/sleep_ep_ranks_tags",
        json={"sleeping_ep_ranks": [2, 3], "tags": ["expert_weights"]},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "ok": True,
        "sleeping_ep_ranks": [2, 3],
        "tags": ["expert_weights"],
    }
    engine.collective_rpc.assert_awaited_once_with(
        "sleep_ep_ranks_by_tags",
        kwargs={"sleeping_ep_ranks": [2, 3], "tags": ["expert_weights"]},
    )


def test_wake_up_ep_ranks_tags_validates_and_dispatches(client):
    test_client, engine = client

    resp = test_client.post(
        "/wake_up_ep_ranks_tags",
        json={"sleeping_ep_ranks": [2, 3], "tags": ["expert_weights"]},
    )

    assert resp.status_code == 200
    engine.collective_rpc.assert_awaited_once_with(
        "wake_up_ep_ranks",
        kwargs={"sleeping_ep_ranks": [2, 3], "tags": ["expert_weights"]},
    )


@pytest.mark.parametrize(
    "payload, expected_msg",
    [
        ({"tags": ["expert_weights"]}, "sleeping_ep_ranks is required"),
        (
            {"sleeping_ep_ranks": [], "tags": ["expert_weights"]},
            "sleeping_ep_ranks must not be empty",
        ),
        (
            {"sleeping_ep_ranks": [True], "tags": ["expert_weights"]},
            "sleeping_ep_ranks must be a list of integers",
        ),
        (
            {"sleeping_ep_ranks": [-1], "tags": ["expert_weights"]},
            "sleeping_ep_ranks must be non-negative",
        ),
        (
            {"sleeping_ep_ranks": [2, 2], "tags": ["expert_weights"]},
            "sleeping_ep_ranks must not contain duplicates",
        ),
    ],
)
def test_sleep_ep_ranks_tags_rejects_invalid_ranks(client, payload, expected_msg):
    test_client, engine = client

    resp = test_client.post("/sleep_ep_ranks_tags", json=payload)

    assert resp.status_code == 400
    assert expected_msg in resp.json()["detail"]
    engine.collective_rpc.assert_not_called()
