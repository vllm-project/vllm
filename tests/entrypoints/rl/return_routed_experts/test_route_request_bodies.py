# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Malformed bodies on RL dev routes are 400s that never reach the engine."""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.rl.online.api_router import router as rl_router
from vllm.entrypoints.serve.dev.rpc.api_router import router as rpc_router
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler

pytestmark = pytest.mark.cpu_test


class Engine:
    def __init__(self):
        self.calls: list[str] = []

    def __getattr__(self, name):
        async def call(*args, **kwargs):
            self.calls.append(name)

        return call


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(rl_router)
    app.include_router(rpc_router)
    init_exception_handler(app)
    app.state.engine_client = Engine()
    app.state.args = SimpleNamespace(log_error_stack=False)
    return TestClient(app)


def post(client, path, body):
    return client.post(
        path, content=json.dumps(body), headers={"content-type": "application/json"}
    )


@pytest.mark.parametrize(
    "path",
    [
        "/abort_requests",
        "/init_weight_transfer_engine",
        "/update_weights",
        "/collective_rpc",
    ],
)
@pytest.mark.parametrize("body", [[], 1, "x", None])
def test_non_object_body_is_rejected(client, path, body):
    assert post(client, path, body).status_code == 400
    assert client.app.state.engine_client.calls == []


@pytest.mark.parametrize(
    "path,body",
    [
        ("/init_weight_transfer_engine", {}),
        ("/init_weight_transfer_engine", {"init_info": []}),
        ("/init_weight_transfer_engine", {"init_info": 1}),
        ("/update_weights", {}),
        ("/update_weights", {"update_info": 1}),
        ("/update_weights", {"update_info": [1, 2]}),
        ("/update_weights", {"update_info": "x"}),
    ],
)
def test_invalid_field_is_rejected_before_the_engine(client, path, body):
    # Reaching the engine would also abort an in-progress weight update.
    assert post(client, path, body).status_code == 400
    assert client.app.state.engine_client.calls == []


@pytest.mark.parametrize(
    "path,body,engine_call",
    [
        ("/abort_requests", {"request_ids": ["a"]}, "abort"),
        (
            "/init_weight_transfer_engine",
            {"init_info": {}},
            "init_weight_transfer_engine",
        ),
        ("/update_weights", {"update_info": {"names": []}}, "update_weights"),
        ("/update_weights", {"update_info": [{}, {}]}, "update_weights"),
        ("/collective_rpc", {"method": "m"}, "collective_rpc"),
    ],
)
def test_valid_body_reaches_the_engine(client, path, body, engine_call):
    assert post(client, path, body).status_code == 200
    assert client.app.state.engine_client.calls == [engine_call]
